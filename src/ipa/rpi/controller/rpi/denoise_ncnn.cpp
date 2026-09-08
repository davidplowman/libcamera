/* SPDX-License-Identifier: BSD-2-Clause */
/*
 * Copyright (C) 2026, Raspberry Pi Ltd
 *
 * Simple NCNN-based full-frame denoise control algorithm
 */

#include "denoise_ncnn.h"

#include <algorithm>
#include <cmath>
#include <errno.h>
#include <fstream>
#include <map>
#include <sstream>
#include <string.h>
#include <sys/ioctl.h>

#include <linux/dma-buf.h>

#include <libcamera/base/log.h>
#include <libcamera/base/shared_fd.h>
#include <libcamera/base/span.h>

#include "../denoise_status.h"

#if defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>
#endif

using namespace RPiController;
using namespace libcamera;

LOG_DEFINE_CATEGORY(RPiDenoiseNcnn)

#define NAME "rpi.denoise_ncnn"

namespace {

constexpr const char *kInputBlob = "in0";
constexpr const char *kOutputBlob = "out0";

void dmabufSyncStart(const SharedFD &fd)
{
	struct dma_buf_sync dma_sync{};
	dma_sync.flags = DMA_BUF_SYNC_START | DMA_BUF_SYNC_RW;

	::ioctl(fd.get(), DMA_BUF_IOCTL_SYNC, &dma_sync);
}

void dmabufSyncEnd(const SharedFD &fd)
{
	struct dma_buf_sync dma_sync{};
	dma_sync.flags = DMA_BUF_SYNC_END | DMA_BUF_SYNC_RW;

	::ioctl(fd.get(), DMA_BUF_IOCTL_SYNC, &dma_sync);
}

#if defined(__ARM_NEON) && defined(__ARM_FP16_FORMAT_IEEE)
/* 8 raw uint16 samples -> 8 fp16 samples, each divided by 65535. */
inline void packStore8(__fp16 *dst, uint16x8_t v, float32x4_t inv)
{
	float32x4_t lo = vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(v))), inv);
	float32x4_t hi = vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_high_u16(v))), inv);
	vst1_f16(dst, vcvt_f16_f32(lo));
	vst1_f16(dst + 4, vcvt_f16_f32(hi));
}

/*
 * downscale_ pack: top/bot each hold 8 raw uint16 samples of one Bayer
 * colour, one per half-res cell, from two raw rows 2 apart (top: the
 * block's first cell-row, bot: its second). Sums the two rows (the
 * block's vertical pair), then vpaddq_f32 sums each adjacent pair of
 * lanes (the block's horizontal pair) -- together giving, per output
 * lane, the sum of exactly the 4 same-colour raw samples in one 4x4
 * block. 4 fp16 outputs from 8+8 raw-derived inputs, each divided by
 * (65535*4) via invQuarter. vpaddq_f32 is an ARMv8/aarch64 NEON
 * instruction (not available under plain 32-bit ARMv7 NEON); this file
 * already assumes aarch64 in practice (Pi 5 is the only real target),
 * via __fp16 and the __ARM_FP16_FORMAT_IEEE guard this sits under.
 */
inline void packStore4Sum4(__fp16 *dst, uint16x8_t top, uint16x8_t bot, float32x4_t invQuarter)
{
	float32x4_t topLo = vcvtq_f32_u32(vmovl_u16(vget_low_u16(top)));
	float32x4_t topHi = vcvtq_f32_u32(vmovl_u16(vget_high_u16(top)));
	float32x4_t botLo = vcvtq_f32_u32(vmovl_u16(vget_low_u16(bot)));
	float32x4_t botHi = vcvtq_f32_u32(vmovl_u16(vget_high_u16(bot)));
	float32x4_t sumLo = vaddq_f32(topLo, botLo);
	float32x4_t sumHi = vaddq_f32(topHi, botHi);
	float32x4_t sum4 = vpaddq_f32(sumLo, sumHi);
	vst1_f16(dst, vcvt_f16_f32(vmulq_f32(sum4, invQuarter)));
}
#endif

#if defined(__ARM_NEON)
/*
 * 4 fp32 samples, each multiplied by 65535 -> 4 uint16 samples in [0,65535],
 * rounded to nearest and clamped. vcvtaq_u32_f32 (FCVTAU) rounds to nearest
 * and saturates negative/NaN to 0 and overflow to UINT32_MAX; vqmovn_u32
 * then saturates that down to the uint16 range -- so the clamp to [0,65535]
 * falls out of the hardware conversion, no extra compare/select needed.
 */
inline uint16x4_t unpackLoad4(const float *src, float32x4_t scaleMax)
{
	float32x4_t v = vmulq_f32(vld1q_f32(src), scaleMax);
	return vqmovn_u32(vcvtaq_u32_f32(v));
}
#endif

inline uint16_t quantise(float v)
{
	v *= 65535.0f;
	v = v < 0.0f ? 0.0f : (v > 65535.0f ? 65535.0f : v);
	return uint16_t(std::lround(v));
}

} /* namespace */

DenoiseNcnn::DenoiseNcnn(Controller *controller)
	: DenoiseAlgorithm(controller)
{
}

char const *DenoiseNcnn::name() const
{
	return NAME;
}

int DenoiseNcnn::read(const libcamera::ValueNode &params)
{
	param_ = params["param"].get<std::string>("");
	bin_ = params["bin"].get<std::string>("");
	sdn_disable_ = params["sdn_disable"].get<bool>(true);
	tdn_disable_ = params["tdn_disable"].get<bool>(true);
	cdn_disable_ = params["cdn_disable"].get<bool>(true);

	threads_ = params["threads"].get<int>(2);
	downscale_ = params["downscale"].get<bool>(false);

	if (param_.empty() || bin_.empty()) {
		LOG(RPiDenoiseNcnn, Error) << "Both \"param\" and \"bin\" must be supplied";
		return -EINVAL;
	}

	return 0;
}

void DenoiseNcnn::initialise()
{
	net_.opt.num_threads = threads_;
	net_.opt.blob_allocator = &blobPool_;
	net_.opt.workspace_allocator = &workPool_;
	net_.opt.use_vulkan_compute = false;
	net_.opt.use_fp16_arithmetic = true;
	net_.opt.use_fp16_packed = true;
	net_.opt.use_fp16_storage = true;

	if (net_.load_param(param_.c_str()) != 0 || net_.load_model(bin_.c_str()) != 0) {
		LOG(RPiDenoiseNcnn, Error) << "Failed to load NCNN model: " << param_;
		return;
	}

	computeAlignment();

	init_ = true;
}

void DenoiseNcnn::switchMode(CameraMode const &cameraMode, [[maybe_unused]] Metadata *metadata)
{
	width_ = cameraMode.width;
	height_ = cameraMode.height;

	/*
	 * Buffer geometry. packW_/packH_ is the net-plane-pixel region
	 * actually backed by real Bayer data: one pixel per 2x2 raw cell
	 * normally, or one pixel per 4x4 raw block (validW_/validH_ halved
	 * again) when downscale_.
	 *
	 * netW_/netH_ (the padded size actually fed to the network) prefers
	 * fixedPlaneW_/fixedPlaneH_, the network's own exact required plane
	 * size read from its Interp layers' hardcoded targets -- ncnn's
	 * Interp bakes in an absolute pixel target, not a scale factor, so
	 * anything other than the exact size it expects silently corrupts
	 * the output (skip-connection sizes stop matching). This is already
	 * expressed directly in net-plane-pixel units, independent of
	 * downscale_. Fallback: round packW_/packH_ up to a multiple of
	 * alignW_/alignH_ (from computeAlignment()'s stride product) -- only
	 * valid for networks without a fixed-size Interp layer to read
	 * instead. A sensor mode smaller than the plane size is fine, it
	 * just means more zero-filled border relative to the valid
	 * (packW_ x packH_) region; larger isn't (see the clamp below).
	 */
	validW_ = width_ / 2;
	validH_ = height_ / 2;
	unsigned packW = downscale_ ? validW_ / 2 : validW_;
	unsigned packH = downscale_ ? validH_ / 2 : validH_;

	if (fixedPlaneW_ && fixedPlaneH_) {
		netW_ = fixedPlaneW_;
		netH_ = fixedPlaneH_;
	} else {
		auto roundUp = [](unsigned v, unsigned align) {
			return align ? (v + align - 1) / align * align : v;
		};
		netW_ = roundUp(packW, alignW_);
		netH_ = roundUp(packH, alignH_);
	}

	if (packW > netW_ || packH > netH_) {
		LOG(RPiDenoiseNcnn, Error)
			<< "Bayer " << width_ << "x" << height_ << " (plane " << packW << "x" << packH
			<< ") exceeds this network's plane size " << netW_ << "x" << netH_
			<< " -- cropping to fit";
		packW = std::min(packW, netW_);
		packH = std::min(packH, netH_);
	}
	packW_ = packW;
	packH_ = packH;
	validW_ = downscale_ ? packW_ * 2 : packW_;
	validH_ = downscale_ ? packH_ * 2 : packH_;

	buffer_.assign(size_t(4) * netW_ * netH_, __fp16(0.0f));
	if (downscale_)
		upscaleBuffer_.assign(size_t(4) * validW_ * validH_, 0.0f);
	else
		upscaleBuffer_.clear();

	LOG(RPiDenoiseNcnn, Info)
		<< "Bayer " << width_ << "x" << height_
		<< " -> planes " << netW_ << "x" << netH_
		<< " (pack " << packW_ << "x" << packH_ << ", valid " << validW_ << "x" << validH_ << ")";
}

void DenoiseNcnn::setMode([[maybe_unused]] DenoiseMode mode)
{
}

/*
 * Determine the net-plane-pixel alignment this specific network needs,
 * straight from its own .param file: multiply together the stride of
 * every Convolution/ConvolutionDepthWise layer. Skip connections between
 * the encoder and decoder require matching sizes at every downsampled
 * scale, so the plane must be an exact multiple of the network's total
 * downsampling factor. Deeper networks (more stride-2 stages) need a
 * larger multiple; this reads it off the model instead of hardcoding it,
 * since different networks in this family need different values (8 vs 4
 * net-plane pixels, seen in practice). Relies on upsampling being done with
 * Interp rather than a strided Deconvolution -- true of every network
 * this has been tested against -- so this doesn't also need to divide out
 * strides that undo themselves later in the graph.
 */
void DenoiseNcnn::computeAlignment()
{
	std::ifstream f(param_);
	if (!f) {
		LOG(RPiDenoiseNcnn, Warning)
			<< "Couldn't open " << param_ << " to determine alignment, defaulting to "
			<< alignW_ << "x" << alignH_;
		return;
	}

	std::string line;
	std::getline(f, line); /* magic number */
	std::getline(f, line); /* layer/blob counts */

	unsigned strideProdW = 1, strideProdH = 1;
	unsigned maxInterpW = 0, maxInterpH = 0;
	while (std::getline(f, line)) {
		std::istringstream iss(line);
		std::string type, name;
		int nin = 0, nout = 0;
		if (!(iss >> type >> name >> nin >> nout))
			continue;
		if (type != "Convolution" && type != "ConvolutionDepthWise" && type != "Interp")
			continue;

		std::string blob;
		for (int i = 0; i < nin + nout && (iss >> blob); i++)
			;

		std::map<int, int> params;
		std::string tok;
		while (iss >> tok) {
			size_t eq = tok.find('=');
			if (eq == std::string::npos)
				continue;
			params[std::atoi(tok.substr(0, eq).c_str())] = std::atoi(tok.c_str() + eq + 1);
		}

		if (type == "Interp") {
			/*
			 * Params: 3=output height, 4=output width, 5=dynamic
			 * target size (computed from a second input blob at
			 * runtime rather than fixed here -- skip those, they
			 * don't tell us anything about a required plane size).
			 */
			if (params.count(5) && params[5])
				continue;
			maxInterpH = std::max(maxInterpH, unsigned(std::max(0, params[3])));
			maxInterpW = std::max(maxInterpW, unsigned(std::max(0, params[4])));
			continue;
		}

		unsigned sw = std::max(1, params.count(3) ? params[3] : 1);
		unsigned sh = std::max(1, params.count(13) ? params[13] : 1);
		strideProdW *= sw;
		strideProdH *= sh;
	}

	alignW_ = strideProdW;
	alignH_ = strideProdH;

	if (maxInterpW && maxInterpH) {
		/*
		 * The largest fixed Interp target is the shallowest (nearest
		 * full-res) upsample stage, i.e. half the network's required
		 * plane size.
		 */
		fixedPlaneW_ = maxInterpW * 2;
		fixedPlaneH_ = maxInterpH * 2;
		LOG(RPiDenoiseNcnn, Info)
			<< "Detected required plane size " << fixedPlaneW_ << "x" << fixedPlaneH_
			<< " from Interp targets in " << param_
			<< " (stride-based alignment would have given " << alignW_ << "x" << alignH_ << ")";
	} else {
		LOG(RPiDenoiseNcnn, Info)
			<< "Detected required net-plane alignment " << alignW_ << "x" << alignH_
			<< " from " << param_ << " (no fixed Interp targets found)";
	}
}

void DenoiseNcnn::packBayer(const uint16_t *bayer16, unsigned stridePx)
{
	if (downscale_) {
		packBayerDownscale(bayer16, stridePx);
		return;
	}

	const unsigned netW = netW_, netH = netH_, packW = packW_, packH = packH_;
	const size_t plane = size_t(netW) * netH;
	__fp16 *p0 = buffer_.data(); /* R  (x=0,y=0) */
	__fp16 *p1 = p0 + plane; /* G1 (x=1,y=0) */
	__fp16 *p2 = p1 + plane; /* G2 (x=0,y=1) */
	__fp16 *p3 = p2 + plane; /* B  (x=1,y=1) */

	constexpr float invMax = 1.0f / 65535.0f;
#if defined(__ARM_NEON) && defined(__ARM_FP16_FORMAT_IEEE)
	const float32x4_t vInv = vdupq_n_f32(invMax);
#endif

#ifdef _OPENMP
#pragma omp parallel for num_threads(threads_) schedule(static)
#endif
	for (int y_ = 0; y_ < int(packH); ++y_) {
		const unsigned y = unsigned(y_);
		const uint16_t *row0 = bayer16 + size_t(2 * y) * stridePx;
		const uint16_t *row1 = row0 + stridePx;
		__fp16 *d0 = p0 + size_t(y) * netW;
		__fp16 *d1 = p1 + size_t(y) * netW;
		__fp16 *d2 = p2 + size_t(y) * netW;
		__fp16 *d3 = p3 + size_t(y) * netW;

		unsigned x = 0;
#if defined(__ARM_NEON) && defined(__ARM_FP16_FORMAT_IEEE)
		for (; x + 8 <= packW; x += 8) {
			const uint16x8x2_t a = vld2q_u16(row0 + size_t(x) * 2);
			const uint16x8x2_t b = vld2q_u16(row1 + size_t(x) * 2);
			packStore8(d0 + x, a.val[0], vInv);
			packStore8(d1 + x, a.val[1], vInv);
			packStore8(d2 + x, b.val[0], vInv);
			packStore8(d3 + x, b.val[1], vInv);
		}
#endif
		for (; x < packW; ++x) {
			d0[x] = __fp16(float(row0[2 * x]) * invMax);
			d1[x] = __fp16(float(row0[2 * x + 1]) * invMax);
			d2[x] = __fp16(float(row1[2 * x]) * invMax);
			d3[x] = __fp16(float(row1[2 * x + 1]) * invMax);
		}
		/* Zero the right-hand border for this row. */
		for (unsigned c = packW; c < netW; ++c)
			d0[c] = d1[c] = d2[c] = d3[c] = __fp16(0.0f);
	}

	/* Zero the bottom border rows, full width. */
#ifdef _OPENMP
#pragma omp parallel for num_threads(threads_) schedule(static)
#endif
	for (int y_ = int(packH); y_ < int(netH); ++y_) {
		const unsigned y = unsigned(y_);
		memset(p0 + size_t(y) * netW, 0, netW * sizeof(__fp16));
		memset(p1 + size_t(y) * netW, 0, netW * sizeof(__fp16));
		memset(p2 + size_t(y) * netW, 0, netW * sizeof(__fp16));
		memset(p3 + size_t(y) * netW, 0, netW * sizeof(__fp16));
	}
}

/*
 * downscale_ variant: one net-plane pixel per 4x4 raw block (an extra
 * 2x2 box-average on top of the usual 2x2 Bayer-cell pack). Output pixel
 * (x,y) covers raw rows [4y,4y+4) and columns [4x,4x+4); each Bayer
 * colour's 4 samples in that block are averaged independently -- e.g. R
 * at block-relative (0,0),(2,0),(0,2),(2,2).
 */
void DenoiseNcnn::packBayerDownscale(const uint16_t *bayer16, unsigned stridePx)
{
	const unsigned netW = netW_, netH = netH_, packW = packW_, packH = packH_;
	const size_t plane = size_t(netW) * netH;
	__fp16 *p0 = buffer_.data(); /* R  */
	__fp16 *p1 = p0 + plane; /* G1 */
	__fp16 *p2 = p1 + plane; /* G2 */
	__fp16 *p3 = p2 + plane; /* B  */

	constexpr float invMax4 = 1.0f / (65535.0f * 4.0f);
#if defined(__ARM_NEON) && defined(__ARM_FP16_FORMAT_IEEE)
	const float32x4_t vInv4 = vdupq_n_f32(invMax4);
#endif

#ifdef _OPENMP
#pragma omp parallel for num_threads(threads_) schedule(static)
#endif
	for (int y_ = 0; y_ < int(packH); ++y_) {
		const unsigned y = unsigned(y_);
		const uint16_t *row0 = bayer16 + size_t(4 * y) * stridePx;
		const uint16_t *row1 = row0 + stridePx;
		const uint16_t *row2 = row1 + stridePx;
		const uint16_t *row3 = row2 + stridePx;
		__fp16 *d0 = p0 + size_t(y) * netW;
		__fp16 *d1 = p1 + size_t(y) * netW;
		__fp16 *d2 = p2 + size_t(y) * netW;
		__fp16 *d3 = p3 + size_t(y) * netW;

		unsigned x = 0;
#if defined(__ARM_NEON) && defined(__ARM_FP16_FORMAT_IEEE)
		for (; x + 4 <= packW; x += 4) {
			const uint16x8x2_t a0 = vld2q_u16(row0 + size_t(x) * 4); /* R,G1 from row0 */
			const uint16x8x2_t a2 = vld2q_u16(row2 + size_t(x) * 4); /* R,G1 from row2 */
			const uint16x8x2_t b1 = vld2q_u16(row1 + size_t(x) * 4); /* G2,B from row1 */
			const uint16x8x2_t b3 = vld2q_u16(row3 + size_t(x) * 4); /* G2,B from row3 */
			packStore4Sum4(d0 + x, a0.val[0], a2.val[0], vInv4);
			packStore4Sum4(d1 + x, a0.val[1], a2.val[1], vInv4);
			packStore4Sum4(d2 + x, b1.val[0], b3.val[0], vInv4);
			packStore4Sum4(d3 + x, b1.val[1], b3.val[1], vInv4);
		}
#endif
		for (; x < packW; ++x) {
			const unsigned rb = 4 * x;
			float r = float(row0[rb + 0]) + row0[rb + 2] + row2[rb + 0] + row2[rb + 2];
			float g1 = float(row0[rb + 1]) + row0[rb + 3] + row2[rb + 1] + row2[rb + 3];
			float g2 = float(row1[rb + 0]) + row1[rb + 2] + row3[rb + 0] + row3[rb + 2];
			float b = float(row1[rb + 1]) + row1[rb + 3] + row3[rb + 1] + row3[rb + 3];
			d0[x] = __fp16(r * invMax4);
			d1[x] = __fp16(g1 * invMax4);
			d2[x] = __fp16(g2 * invMax4);
			d3[x] = __fp16(b * invMax4);
		}
		/* Zero the right-hand border for this row. */
		for (unsigned c = packW; c < netW; ++c)
			d0[c] = d1[c] = d2[c] = d3[c] = __fp16(0.0f);
	}

	/* Zero the bottom border rows, full width. */
#ifdef _OPENMP
#pragma omp parallel for num_threads(threads_) schedule(static)
#endif
	for (int y_ = int(packH); y_ < int(netH); ++y_) {
		const unsigned y = unsigned(y_);
		memset(p0 + size_t(y) * netW, 0, netW * sizeof(__fp16));
		memset(p1 + size_t(y) * netW, 0, netW * sizeof(__fp16));
		memset(p2 + size_t(y) * netW, 0, netW * sizeof(__fp16));
		memset(p3 + size_t(y) * netW, 0, netW * sizeof(__fp16));
	}
}

bool DenoiseNcnn::runNet(ncnn::Mat &out)
{
	ncnn::Mat in(int(netW_), int(netH_), 4, (void *)buffer_.data(), size_t(2u));

	ncnn::Extractor ex = net_.create_extractor();
	if (ex.input(kInputBlob, in) != 0) {
		LOG(RPiDenoiseNcnn, Error) << "ex.input(" << kInputBlob << ") failed";
		return false;
	}
	if (ex.extract(kOutputBlob, out) != 0) {
		LOG(RPiDenoiseNcnn, Error) << "ex.extract(" << kOutputBlob << ") failed";
		return false;
	}

	/*
	 * out.c < 4 would mean ncnn hasn't unpacked the final blob back to
	 * separate channel planes (elempack > 1) -- catch that here rather
	 * than reading out-of-bounds channels below.
	 */
	return out.c >= 4 && (unsigned)out.w == netW_ && (unsigned)out.h == netH_;
}

void DenoiseNcnn::unpackBayer(uint16_t *bayer16, unsigned stridePx, const ncnn::Mat &out)
{
	if (downscale_) {
		upscale2x(out);
		const size_t plane = size_t(validW_) * validH_;
		interleaveToRaw(bayer16, stridePx, upscaleBuffer_.data() + 0 * plane,
				 upscaleBuffer_.data() + 1 * plane, upscaleBuffer_.data() + 2 * plane,
				 upscaleBuffer_.data() + 3 * plane, validW_);
		return;
	}

	interleaveToRaw(bayer16, stridePx, out.channel(0), out.channel(1), out.channel(2),
			 out.channel(3), netW_);
}

void DenoiseNcnn::interleaveToRaw(uint16_t *bayer16, unsigned stridePx, const float *R,
				   const float *G1, const float *G2, const float *B,
				   unsigned srcStride)
{
	const unsigned validW = validW_, validH = validH_;

#if defined(__ARM_NEON)
	constexpr float scaleMax = 65535.0f;
	const float32x4_t vScaleMax = vdupq_n_f32(scaleMax);
#endif

#ifdef _OPENMP
#pragma omp parallel for num_threads(threads_) schedule(static)
#endif
	for (int y_ = 0; y_ < int(validH); ++y_) {
		const unsigned y = unsigned(y_);
		uint16_t *row0 = bayer16 + size_t(2 * y) * stridePx;
		uint16_t *row1 = row0 + stridePx;
		const float *s0 = R + size_t(y) * srcStride;
		const float *s1 = G1 + size_t(y) * srcStride;
		const float *s2 = G2 + size_t(y) * srcStride;
		const float *s3 = B + size_t(y) * srcStride;

		unsigned x = 0;
#if defined(__ARM_NEON)
		for (; x + 4 <= validW; x += 4) {
			const uint16x4x2_t a{ unpackLoad4(s0 + x, vScaleMax), unpackLoad4(s1 + x, vScaleMax) };
			const uint16x4x2_t b{ unpackLoad4(s2 + x, vScaleMax), unpackLoad4(s3 + x, vScaleMax) };
			vst2_u16(row0 + size_t(x) * 2, a);
			vst2_u16(row1 + size_t(x) * 2, b);
		}
#endif
		for (; x < validW; ++x) {
			row0[2 * x] = quantise(s0[x]);
			row0[2 * x + 1] = quantise(s1[x]);
			row1[2 * x] = quantise(s2[x]);
			row1[2 * x + 1] = quantise(s3[x]);
		}
	}
}

/*
 * downscale_ only: bilinearly upscale each of the network's 4 quarter-res
 * output planes (out.channel(c), netW_ x netH_, valid region packW_ x
 * packH_) by exactly 2x in each dimension into upscaleBuffer_ (validW_ x
 * validH_ == packW_*2 x packH_*2 by construction), ready for
 * interleaveToRaw().
 */
void DenoiseNcnn::upscale2x(const ncnn::Mat &out)
{
	const size_t plane = size_t(validW_) * validH_;
	for (int c = 0; c < 4; c++)
		upscaleChannel2x(out.channel(c), netW_, packW_, packH_,
				  upscaleBuffer_.data() + size_t(c) * plane, validW_, validH_);
}

/*
 * Exact 2x bilinear upscale of one srcW x srcH plane (stride srcStride)
 * into a dstStride x dstH plane (dstStride == srcW*2, dstH == srcH*2):
 * original samples land unchanged on even,even destination pixels;
 * odd,even are the average of horizontal neighbours; even,odd the average
 * of vertical neighbours; odd,odd the average of all 4 diagonal
 * neighbours -- which falls out of first horizontally interpolating each
 * source row, then averaging vertically-adjacent interpolated rows.
 * Edge columns/rows clamp to the last valid sample (no wraparound).
 */
void DenoiseNcnn::upscaleChannel2x(const float *src, unsigned srcStride, unsigned srcW,
				    unsigned srcH, float *dst, unsigned dstStride, unsigned dstH)
{
	auto hInterpRow = [](const float *srcRow, unsigned w, float *dstRow) {
		unsigned j = 0;
#if defined(__ARM_NEON)
		/* +5, not +4: keeps the one-element lookahead load (v1's
		 * last lane, index j+4) inside [0, w-1]. */
		for (; j + 5 <= w; j += 4) {
			float32x4_t v0 = vld1q_f32(srcRow + j);
			float32x4_t v1 = vld1q_f32(srcRow + j + 1);
			float32x4_t avg = vmulq_n_f32(vaddq_f32(v0, v1), 0.5f);
			float32x4x2_t inter{ v0, avg };
			vst2q_f32(dstRow + 2 * j, inter);
		}
#endif
		for (; j < w; j++) {
			float v0 = srcRow[j], v1 = srcRow[std::min(j + 1, w - 1)];
			dstRow[2 * j] = v0;
			dstRow[2 * j + 1] = 0.5f * (v0 + v1);
		}
	};

#ifdef _OPENMP
#pragma omp parallel for num_threads(threads_) schedule(static)
#endif
	for (int sy_ = 0; sy_ < int(srcH); ++sy_) {
		const unsigned sy = unsigned(sy_);
		/* Reused across rows/frames within each OpenMP worker thread,
		 * rather than a fresh heap allocation per row. */
		thread_local std::vector<float> nextRow;
		nextRow.resize(dstStride);
		float *evenRow = dst + size_t(2 * sy) * dstStride;

		hInterpRow(src + size_t(sy) * srcStride, srcW, evenRow);

		if (2 * sy + 1 < dstH) {
			const unsigned sy1 = std::min(sy + 1, srcH - 1);
			hInterpRow(src + size_t(sy1) * srcStride, srcW, nextRow.data());
			float *oddRow = evenRow + dstStride;
			for (unsigned j = 0; j < dstStride; j++)
				oddRow[j] = 0.5f * (evenRow[j] + nextRow[j]);
		}
	}
}

void DenoiseNcnn::prepare(Metadata *imageMetadata)
{
	std::pair<SharedFD, Span<uint8_t>> bayer;
	imageMetadata->get("global.bayer_buffer", bayer);

	if (!init_ || !bayer.second.data() || !width_ || !height_)
		return;

	dmabufSyncStart(bayer.first);

	uint16_t *bayer16 = reinterpret_cast<uint16_t *>(bayer.second.begin());
	const unsigned stridePx = (bayer.second.size() / height_) / sizeof(uint16_t);

	/*
	 * Accept row padding but reject a compressed buffer -- same test
	 * model_denoise.cpp uses: bytes-per-row >= width, not an exact size.
	 */
	const size_t rowBytes = bayer.second.size() / height_;
	const size_t minRow = size_t(width_) * sizeof(uint16_t);
	if (bayer.second.size() % height_ || rowBytes < minRow) {
		static bool warned = false;
		if (!warned) {
			warned = true;
			LOG(RPiDenoiseNcnn, Error)
				<< "raw buffer " << bayer.second.size() << " bytes = " << rowBytes
				<< " B/row, need >= " << minRow << " for " << width_
				<< " px wide. Compressed raw? Skipping denoise_ncnn.";
		}
		dmabufSyncEnd(bayer.first);
		return;
	}

	packBayer(bayer16, stridePx);

	ncnn::Mat out;
	if (!runNet(out)) {
		LOG(RPiDenoiseNcnn, Error) << "NCNN infer failed";
		dmabufSyncEnd(bayer.first);
		return;
	}

	unpackBayer(bayer16, stridePx, out);

	dmabufSyncEnd(bayer.first);

	if (sdn_disable_)
		imageMetadata->erase("sdn.status");
	if (tdn_disable_)
		imageMetadata->erase("tdn.status");
	else {
		TdnStatus tdnStatus{};
		if (imageMetadata->get("tdn.status", tdnStatus) == 0) {
			tdnStatus.noiseConstant /= 2;
			tdnStatus.noiseSlope /= 2;
			imageMetadata->set("tdn.status", tdnStatus);
		}
	}
	if (cdn_disable_)
		imageMetadata->erase("cdn.status");
}

/* Register algorithm with the system. */
static Algorithm *create(Controller *controller)
{
	return (Algorithm *)new DenoiseNcnn(controller);
}
static RegisterAlgorithm reg(NAME, &create);
