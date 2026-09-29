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
 * downscale_ pack: a, b, c, d each hold 8 raw uint16 samples of one Bayer
 * colour, one per consecutive 4x4 raw block along a row (a/b: the two
 * same-colour columns of the block's first same-colour row, c/d: of its
 * second). Summing all four gives, per lane, the sum of exactly the 4
 * same-colour samples in one 4x4 block (i.e. the 2x2 Bayer-plane average,
 * once scaled by invQuarter = 1/(65535*4)). The 8 consecutive blocks are
 * then split into even ones (dstEven, space-to-depth column j=0) and odd
 * ones (dstOdd, j=1), 4 fp16 outputs each. vuzp1q_f32/vuzp2q_f32 are
 * aarch64-only; this file already assumes aarch64 in practice (Pi 5 is the
 * only real target), via __fp16 and the __ARM_FP16_FORMAT_IEEE guard this
 * sits under.
 */
inline void packStore4x2Sum4(__fp16 *dstEven, __fp16 *dstOdd, uint16x8_t a, uint16x8_t b,
			     uint16x8_t c, uint16x8_t d, float32x4_t invQuarter)
{
	uint32x4_t lo = vaddq_u32(vaddl_u16(vget_low_u16(a), vget_low_u16(b)),
				  vaddl_u16(vget_low_u16(c), vget_low_u16(d)));
	uint32x4_t hi = vaddq_u32(vaddl_u16(vget_high_u16(a), vget_high_u16(b)),
				  vaddl_u16(vget_high_u16(c), vget_high_u16(d)));
	float32x4_t loF = vmulq_f32(vcvtq_f32_u32(lo), invQuarter);
	float32x4_t hiF = vmulq_f32(vcvtq_f32_u32(hi), invQuarter);
	vst1_f16(dstEven, vcvt_f16_f32(vuzp1q_f32(loF, hiF)));
	vst1_f16(dstOdd, vcvt_f16_f32(vuzp2q_f32(loF, hiF)));
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

/*
 * The network's input/output is the 4 Bayer planes (R, G1, G2, B, i.e.
 * raw 2x2-cell positions (0,0), (1,0), (0,1), (1,1)) further packed by
 * space_to_depth(2) (PyTorch pixel_unshuffle): channel c*4 + i*2 + j holds
 * Bayer plane c's (row i, column j) sub-position within each 2x2 block of
 * that plane. Without downscale_, that means one network pixel per 4x4
 * raw block, and this returns the channel holding raw offset (dx, dy)
 * within that block.
 */
constexpr unsigned rawToChannel(unsigned dx, unsigned dy)
{
	return 4 * (2 * (dy & 1) + (dx & 1)) + 2 * (dy >> 1) + (dx >> 1);
}

constexpr unsigned kChannels = 16;

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

	if (inChannels_ && inChannels_ != kChannels) {
		LOG(RPiDenoiseNcnn, Error)
			<< param_ << " takes " << inChannels_ << " input channels, but "
			<< kChannels << " (space-to-depth packed Bayer) are required";
		return;
	}

	init_ = true;
}

void DenoiseNcnn::switchMode(CameraMode const &cameraMode, [[maybe_unused]] Metadata *metadata)
{
	width_ = cameraMode.width;
	height_ = cameraMode.height;

	/*
	 * Buffer geometry. The 4 Bayer planes the network sees are half the
	 * raw size normally, or a quarter when downscale_ (2x2 averaged).
	 * space_to_depth(2) halves those again, so packW_/packH_, the
	 * net-plane-pixel region actually backed by real Bayer data, is one
	 * pixel per 4x4 raw block normally, or per 8x8 raw block when
	 * downscale_.
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
	const unsigned bayerDiv = downscale_ ? 4 : 2;
	unsigned packW = width_ / bayerDiv / 2;
	unsigned packH = height_ / bayerDiv / 2;

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
	bayerW_ = packW_ * 2;
	bayerH_ = packH_ * 2;
	validW_ = downscale_ ? bayerW_ * 2 : bayerW_;
	validH_ = downscale_ ? bayerH_ * 2 : bayerH_;

	/*
	 * Must match the cstep ncnn::Mat computes when wrapping buffer_ in
	 * runNet(): each channel starts on a 16-byte boundary, so a plane
	 * whose size isn't a multiple of 8 fp16 samples (e.g. 242x138) is
	 * followed by padding.
	 */
	planeStride_ = (size_t(netW_) * netH_ * sizeof(__fp16) + 15) / 16 * 16 / sizeof(__fp16);

	buffer_.assign(kChannels * planeStride_, __fp16(0.0f));
	if (downscale_) {
		bayerBuffer_.assign(size_t(4) * bayerW_ * bayerH_, 0.0f);
		upscaleBuffer_.assign(size_t(4) * validW_ * validH_, 0.0f);
	} else {
		bayerBuffer_.clear();
		upscaleBuffer_.clear();
	}

	LOG(RPiDenoiseNcnn, Info)
		<< "Bayer " << width_ << "x" << height_
		<< " -> " << kChannels << " planes " << netW_ << "x" << netH_
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

		/*
		 * The first Convolution is the network's input layer: its
		 * weight count (6) is num_output (0) x input channels x
		 * kernel_w (1) x kernel_h (11, defaulting to kernel_w).
		 */
		if (type == "Convolution" && !inChannels_) {
			int kw = params.count(1) ? params[1] : 1;
			int kh = params.count(11) ? params[11] : kw;
			int denom = (params.count(0) ? params[0] : 0) * kw * kh;
			if (denom > 0 && params.count(6) && params[6] % denom == 0)
				inChannels_ = params[6] / denom;
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
		 * The largest fixed Interp target is the shallowest upsample
		 * stage, which restores the full network resolution for the
		 * top-level skip connection, i.e. the network's required plane
		 * size.
		 */
		fixedPlaneW_ = maxInterpW;
		fixedPlaneH_ = maxInterpH;
		LOG(RPiDenoiseNcnn, Info)
			<< "Detected required plane size " << fixedPlaneW_ << "x" << fixedPlaneH_
			<< " from Interp targets in " << param_
			<< " (stride-based alignment would have given " << alignW_ << "x" << alignH_ << ")";
	} else {
		LOG(RPiDenoiseNcnn, Info)
			<< "Detected required net-plane alignment " << alignW_ << "x" << alignH_
			<< " from " << param_ << " (no fixed Interp targets found)";
	}

	if (!inChannels_)
		LOG(RPiDenoiseNcnn, Warning)
			<< "Couldn't determine input channel count from " << param_
			<< ", assuming " << kChannels;
}

/*
 * Zero the net-plane border beyond packW_ x packH_ in every channel, so
 * the network sees black rather than stale data there.
 */
void DenoiseNcnn::zeroBorder()
{
	const unsigned netW = netW_, netH = netH_, packW = packW_, packH = packH_;

	/* Single-threaded, as is packBayer(): it's only the (small) padding. */
	for (int k = 0; k < int(kChannels); ++k) {
		__fp16 *p = buffer_.data() + size_t(k) * planeStride_;
		if (packW < netW) {
			for (unsigned y = 0; y < packH; ++y)
				std::fill(p + size_t(y) * netW + packW, p + size_t(y + 1) * netW,
					  __fp16(0.0f));
		}
		memset(p + size_t(packH) * netW, 0, size_t(netH - packH) * netW * sizeof(__fp16));
	}
}

/*
 * Pack the raw Bayer frame into the network's 16 input planes: the 4
 * Bayer planes, each further packed by space_to_depth(2), so network pixel
 * (x,y) covers raw rows [4y,4y+4) and columns [4x,4x+4), with raw offset
 * (dx,dy) going to channel rawToChannel(dx,dy).
 */
void DenoiseNcnn::packBayer(const uint16_t *bayer16, unsigned stridePx)
{
	if (downscale_) {
		packBayerDownscale(bayer16, stridePx);
		return;
	}

	const unsigned netW = netW_, packW = packW_, packH = packH_;
	__fp16 *const base = buffer_.data();
	const size_t plane = planeStride_;

	constexpr float invMax = 1.0f / 65535.0f;
#if defined(__ARM_NEON) && defined(__ARM_FP16_FORMAT_IEEE)
	const float32x4_t vInv = vdupq_n_f32(invMax);
#endif

	/*
	 * Single-threaded, as is depthToRaw(): this is memory-bound, and
	 * measured slower with more threads.
	 */
	for (unsigned y = 0; y < packH; ++y) {
		for (unsigned dy = 0; dy < 4; ++dy) {
			const uint16_t *row = bayer16 + size_t(4 * y + dy) * stridePx;
			__fp16 *d[4];
			for (unsigned dx = 0; dx < 4; ++dx)
				d[dx] = base + rawToChannel(dx, dy) * plane + size_t(y) * netW;

			unsigned x = 0;
#if defined(__ARM_NEON) && defined(__ARM_FP16_FORMAT_IEEE)
			for (; x + 8 <= packW; x += 8) {
				const uint16x8x4_t v = vld4q_u16(row + size_t(x) * 4);
				packStore8(d[0] + x, v.val[0], vInv);
				packStore8(d[1] + x, v.val[1], vInv);
				packStore8(d[2] + x, v.val[2], vInv);
				packStore8(d[3] + x, v.val[3], vInv);
			}
#endif
			for (; x < packW; ++x) {
				for (unsigned dx = 0; dx < 4; ++dx)
					d[dx][x] = __fp16(float(row[4 * x + dx]) * invMax);
			}
		}
	}

	zeroBorder();
}

/*
 * downscale_ variant: first box-average each Bayer colour over 2x2 (each
 * averaged sample covering a 4x4 raw block), then space_to_depth(2) as
 * usual -- so network pixel (x,y) covers raw rows [8y,8y+8) and columns
 * [8x,8x+8). Channel c*4 + i*2 + j holds colour c averaged over the 4x4
 * raw block at raw rows [8y+4i,8y+4i+4) and columns [8x+4j,8x+4j+4); e.g.
 * R at block-relative (0,0),(2,0),(0,2),(2,2).
 */
void DenoiseNcnn::packBayerDownscale(const uint16_t *bayer16, unsigned stridePx)
{
	const unsigned netW = netW_, packW = packW_, packH = packH_;
	__fp16 *const base = buffer_.data();
	const size_t plane = planeStride_;

	constexpr float invMax4 = 1.0f / (65535.0f * 4.0f);
#if defined(__ARM_NEON) && defined(__ARM_FP16_FORMAT_IEEE)
	const float32x4_t vInv4 = vdupq_n_f32(invMax4);
#endif

#ifdef _OPENMP
#pragma omp parallel for num_threads(threads_) schedule(static)
#endif
	for (int y_ = 0; y_ < int(packH); ++y_) {
		const unsigned y = unsigned(y_);
		for (unsigned i = 0; i < 2; ++i) {
			const uint16_t *rows[4];
			for (unsigned r = 0; r < 4; ++r)
				rows[r] = bayer16 + size_t(8 * y + 4 * i + r) * stridePx;
			/* d[c][j]: colour c, space-to-depth column j, this row i. */
			__fp16 *d[4][2];
			for (unsigned c = 0; c < 4; ++c)
				for (unsigned j = 0; j < 2; ++j)
					d[c][j] = base + (c * 4 + i * 2 + j) * plane + size_t(y) * netW;

			unsigned x = 0;
#if defined(__ARM_NEON) && defined(__ARM_FP16_FORMAT_IEEE)
			for (; x + 4 <= packW; x += 4) {
				/* Lane l of val[dx]: raw column 8x + 4l + dx, i.e. 4x4 block 2x + l. */
				uint16x8x4_t v[4];
				for (unsigned r = 0; r < 4; ++r)
					v[r] = vld4q_u16(rows[r] + size_t(x) * 8);
				for (unsigned c = 0; c < 4; ++c) {
					const unsigned cx = c & 1, cy = c >> 1;
					packStore4x2Sum4(d[c][0] + x, d[c][1] + x,
							 v[cy].val[cx], v[cy].val[cx + 2],
							 v[cy + 2].val[cx], v[cy + 2].val[cx + 2], vInv4);
				}
			}
#endif
			for (; x < packW; ++x) {
				for (unsigned c = 0; c < 4; ++c) {
					const unsigned cx = c & 1, cy = c >> 1;
					for (unsigned j = 0; j < 2; ++j) {
						const unsigned col = 8 * x + 4 * j + cx;
						float sum = float(rows[cy][col]) + rows[cy][col + 2] +
							    rows[cy + 2][col] + rows[cy + 2][col + 2];
						d[c][j][x] = __fp16(sum * invMax4);
					}
				}
			}
		}
	}

	zeroBorder();
}

bool DenoiseNcnn::runNet(ncnn::Mat &out)
{
	ncnn::Mat in(int(netW_), int(netH_), int(kChannels), (void *)buffer_.data(), size_t(2u));
	if (in.cstep != planeStride_) {
		LOG(RPiDenoiseNcnn, Error)
			<< "ncnn plane stride " << in.cstep << " != expected " << planeStride_;
		return false;
	}

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
	 * out.c != 16 would mean ncnn hasn't unpacked the final blob back to
	 * separate channel planes (elempack > 1) -- catch that here rather
	 * than reading out-of-bounds channels below.
	 */
	return (unsigned)out.c == kChannels && out.elempack == 1 && out.elemsize == sizeof(float) &&
	       (unsigned)out.w == netW_ && (unsigned)out.h == netH_;
}

void DenoiseNcnn::unpackBayer(uint16_t *bayer16, unsigned stridePx, const ncnn::Mat &out)
{
	if (downscale_) {
		depthToSpace(out);
		const size_t bayerPlane = size_t(bayerW_) * bayerH_;
		const size_t plane = size_t(validW_) * validH_;
		for (unsigned c = 0; c < 4; c++)
			upscaleChannel2x(bayerBuffer_.data() + c * bayerPlane, bayerW_, bayerW_, bayerH_,
					 upscaleBuffer_.data() + c * plane, validW_, validH_);
		interleaveToRaw(bayer16, stridePx, upscaleBuffer_.data() + 0 * plane,
				upscaleBuffer_.data() + 1 * plane, upscaleBuffer_.data() + 2 * plane,
				upscaleBuffer_.data() + 3 * plane, validW_);
		return;
	}

	depthToRaw(bayer16, stridePx, out);
}

/*
 * Inverse of packBayer(): write each network output pixel's 16 channels
 * straight back over its 4x4 raw block.
 */
void DenoiseNcnn::depthToRaw(uint16_t *bayer16, unsigned stridePx, const ncnn::Mat &out)
{
	const unsigned netW = netW_, packW = packW_, packH = packH_;

#if defined(__ARM_NEON)
	const float32x4_t vScaleMax = vdupq_n_f32(65535.0f);
#endif

	/*
	 * Single-threaded, as is packBayer(): this is memory-bound, and
	 * measured slower with more threads.
	 */
	for (unsigned y = 0; y < packH; ++y) {
		for (unsigned dy = 0; dy < 4; ++dy) {
			uint16_t *row = bayer16 + size_t(4 * y + dy) * stridePx;
			const float *s[4];
			for (unsigned dx = 0; dx < 4; ++dx)
				s[dx] = static_cast<const float *>(out.channel(rawToChannel(dx, dy))) +
					size_t(y) * netW;

			unsigned x = 0;
#if defined(__ARM_NEON)
			for (; x + 4 <= packW; x += 4) {
				const uint16x4x4_t v{ unpackLoad4(s[0] + x, vScaleMax),
						      unpackLoad4(s[1] + x, vScaleMax),
						      unpackLoad4(s[2] + x, vScaleMax),
						      unpackLoad4(s[3] + x, vScaleMax) };
				vst4_u16(row + size_t(x) * 4, v);
			}
#endif
			for (; x < packW; ++x) {
				for (unsigned dx = 0; dx < 4; ++dx)
					row[4 * x + dx] = quantise(s[dx][x]);
			}
		}
	}
}

/*
 * downscale_ only: depth_to_space(2) the network's 16 output channels back
 * into 4 quarter-res Bayer planes (bayerBuffer_, bayerW_ x bayerH_ each),
 * ready for upscaleChannel2x().
 */
void DenoiseNcnn::depthToSpace(const ncnn::Mat &out)
{
	const unsigned netW = netW_, packW = packW_, packH = packH_, bayerW = bayerW_;
	const size_t bayerPlane = size_t(bayerW_) * bayerH_;

#ifdef _OPENMP
#pragma omp parallel for num_threads(threads_) schedule(static)
#endif
	for (int cy_ = 0; cy_ < int(4 * packH); ++cy_) {
		const unsigned c = unsigned(cy_) / packH, y = unsigned(cy_) % packH;
		for (unsigned i = 0; i < 2; ++i) {
			float *dst = bayerBuffer_.data() + c * bayerPlane + size_t(2 * y + i) * bayerW;
			const float *s0 = static_cast<const float *>(out.channel(c * 4 + i * 2 + 0)) +
					  size_t(y) * netW;
			const float *s1 = static_cast<const float *>(out.channel(c * 4 + i * 2 + 1)) +
					  size_t(y) * netW;

			unsigned x = 0;
#if defined(__ARM_NEON)
			for (; x + 4 <= packW; x += 4) {
				const float32x4x2_t v{ vld1q_f32(s0 + x), vld1q_f32(s1 + x) };
				vst2q_f32(dst + 2 * x, v);
			}
#endif
			for (; x < packW; ++x) {
				dst[2 * x] = s0[x];
				dst[2 * x + 1] = s1[x];
			}
		}
	}
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
