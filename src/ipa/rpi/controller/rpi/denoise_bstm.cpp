/* SPDX-License-Identifier: BSD-2-Clause */
/*
 * Copyright (C) 2026, Raspberry Pi Ltd
 *
 * Simple full-frame denoise control algorithm on the BSTM NPU
 */

#include "denoise_bstm.h"

#include <algorithm>
#include <cmath>
#include <errno.h>
#include <sstream>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <type_traits>

#include <linux/dma-buf.h>

#include <libcamera/base/log.h>
#include <libcamera/base/shared_fd.h>
#include <libcamera/base/span.h>

#include <tensorflow/lite/builtin_ops.h>
#include <tensorflow/lite/delegates/external/external_delegate.h>
#include <tensorflow/lite/interpreter_builder.h>
#include <tensorflow/lite/kernels/register.h>

#include "../denoise_status.h"

#if defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>
#endif

using namespace RPiController;
using namespace libcamera;

LOG_DEFINE_CATEGORY(RPiDenoiseBstm)

#define NAME "rpi.denoise_bstm"

namespace {

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

#if defined(__ARM_NEON)
/*
 * 4 fp32 samples v -> 4 uint16 samples v * 65535 + base in [0,65535],
 * rounded to nearest and clamped. base is 0, or with residual_ the raw
 * input samples the network's output is a correction to. vcvtaq_u32_f32
 * (FCVTAU) rounds to nearest and saturates negative/NaN to 0 and overflow
 * to UINT32_MAX; vqmovn_u32 then saturates that down to the uint16 range --
 * so the clamp to [0,65535] falls out of the hardware conversion, no extra
 * compare/select needed. The multiply-add is fused, as in quantise(), so
 * both give identical results.
 */
inline uint16x4_t toRaw4(float32x4_t v, float32x4_t base, float32x4_t scaleMax)
{
	return vqmovn_u32(vcvtaq_u32_f32(vfmaq_f32(base, v, scaleMax)));
}

inline float32x4_t rawToFloat4(uint16x4_t v)
{
	return vcvtq_f32_u32(vmovl_u16(v));
}
#endif

/*
 * Scalar v * scaleMax + base, rounded and clamped to [0,65535] (see
 * toRaw4()). scaleMax is 65535, or 65535 times an int8 output tensor's
 * scale when v is its (q - zero point).
 */
inline uint16_t quantise(float v, float base = 0.0f, float scaleMax = 65535.0f)
{
	v = std::fma(v, scaleMax, base);
	v = v < 0.0f ? 0.0f : (v > 65535.0f ? 65535.0f : v);
	return uint16_t(std::lround(v));
}

/*
 * The network's input/output is the 4 Bayer planes (R, G1, G2, B, i.e.
 * raw 2x2-cell positions (0,0), (1,0), (0,1), (1,1)) further packed by
 * space_to_depth(2) (PyTorch pixel_unshuffle): channel c*4 + i*2 + j holds
 * Bayer plane c's (row i, column j) sub-position within each 2x2 block of
 * that plane. The TFLite tensor is NHWC, so each network pixel's 16
 * channels are consecutive. Without downscale_, a network pixel covers
 * one 4x4 raw block, and this returns the channel holding raw offset
 * (dx, dy) within that block.
 */
constexpr unsigned rawToChannel(unsigned dx, unsigned dy)
{
	return 4 * (2 * (dy & 1) + (dx & 1)) + 2 * (dy >> 1) + (dx >> 1);
}

constexpr unsigned kChannels = 16;

} /* namespace */

DenoiseBstm::DenoiseBstm(Controller *controller)
	: DenoiseAlgorithm(controller)
{
}

DenoiseBstm::~DenoiseBstm()
{
	/* The interpreter still references the delegate until it's gone. */
	interpreter_.reset();
	if (tfDelegate_)
		TfLiteExternalDelegateDelete(tfDelegate_);
}

char const *DenoiseBstm::name() const
{
	return NAME;
}

int DenoiseBstm::read(const libcamera::ValueNode &params)
{
	model_ = params["model"].get<std::string>("");
	delegate_ = params["delegate"].get<std::string>(RPI_TEFLON_PATH);
	floatFormat_ = params["float_format"].get<std::string>("bf16");
	sdn_disable_ = params["sdn_disable"].get<bool>(true);
	tdn_disable_ = params["tdn_disable"].get<bool>(true);
	cdn_disable_ = params["cdn_disable"].get<bool>(true);

	threads_ = params["threads"].get<int>(2);
	downscale_ = params["downscale"].get<bool>(false);
	demo_ = params["demo"].get<bool>(false);
	residual_ = params["residual"].get<bool>(false);

	if (model_.empty()) {
		LOG(RPiDenoiseBstm, Error) << "\"model\" must be supplied";
		return -EINVAL;
	}

	if (floatFormat_ != "fp32" && floatFormat_ != "bf16" && floatFormat_ != "fp16") {
		LOG(RPiDenoiseBstm, Error)
			<< "\"float_format\" must be \"fp32\", \"bf16\" or \"fp16\", not \""
			<< floatFormat_ << "\"";
		return -EINVAL;
	}

	return 0;
}

/*
 * A model input or output must be [1, H, W, 16], float32 or int8. An int8
 * tensor must be quantised per tensor (a single scale and zero point),
 * which are returned in scale and zp.
 */
bool DenoiseBstm::checkTensor(const TfLiteTensor *t, const char *what, bool &isInt8,
			      float &scale, int &zp)
{
	if (!t || !t->dims || t->dims->size != 4 || t->dims->data[0] != 1 ||
	    t->dims->data[3] != int(kChannels)) {
		LOG(RPiDenoiseBstm, Error)
			<< model_ << ": " << what << " must be [1, H, W, " << kChannels << "]";
		return false;
	}

	if (t->type == kTfLiteFloat32) {
		isInt8 = false;
		return true;
	}

	if (t->type == kTfLiteInt8) {
		const auto *q = static_cast<const TfLiteAffineQuantization *>(t->quantization.params);
		if (t->quantization.type != kTfLiteAffineQuantization || !q || !q->scale ||
		    q->scale->size != 1 || !(t->params.scale > 0.0f)) {
			LOG(RPiDenoiseBstm, Error)
				<< model_ << ": int8 " << what << " must have a single (per-tensor) scale";
			return false;
		}
		isInt8 = true;
		scale = t->params.scale;
		zp = t->params.zero_point;
		return true;
	}

	LOG(RPiDenoiseBstm, Error)
		<< model_ << ": " << what << " must be float32 or int8, not " << TfLiteTypeGetName(t->type);
	return false;
}

void DenoiseBstm::initialise()
{
	flatbuffer_ = tflite::FlatBufferModel::BuildFromFile(model_.c_str());
	if (!flatbuffer_) {
		LOG(RPiDenoiseBstm, Error) << "Failed to load TFLite model: " << model_;
		return;
	}

	tflite::ops::builtin::BuiltinOpResolver resolver;
	tflite::InterpreterBuilder(*flatbuffer_, resolver)(&interpreter_);
	if (!interpreter_) {
		LOG(RPiDenoiseBstm, Error) << "Failed to build TFLite interpreter for " << model_;
		return;
	}
	interpreter_->SetNumThreads(threads_);

	/* Read by the bstm driver when the delegate opens the NPU. */
	setenv("BSTM_FLOAT", floatFormat_.c_str(), 0);

	TfLiteExternalDelegateOptions options = TfLiteExternalDelegateOptionsDefault(delegate_.c_str());
	tfDelegate_ = TfLiteExternalDelegateCreate(&options);
	if (!tfDelegate_) {
		LOG(RPiDenoiseBstm, Error) << "Failed to create the Teflon delegate from " << delegate_;
		return;
	}

	/* This is where the driver compiles the model (or finds it in its disk cache). */
	if (interpreter_->ModifyGraphWithDelegate(tfDelegate_) != kTfLiteOk) {
		LOG(RPiDenoiseBstm, Error) << "Failed to apply the Teflon delegate to " << model_;
		return;
	}

	if (interpreter_->AllocateTensors() != kTfLiteOk) {
		LOG(RPiDenoiseBstm, Error) << "Failed to allocate tensors for " << model_;
		return;
	}

	unsigned delegated = 0;
	const std::vector<int> &plan = interpreter_->execution_plan();
	for (int node : plan) {
		const auto *nr = interpreter_->node_and_registration(node);
		if (nr && nr->second.builtin_code == kTfLiteBuiltinDelegate)
			delegated++;
	}
	if (!delegated)
		LOG(RPiDenoiseBstm, Warning) << "No part of " << model_ << " runs on the NPU";
	else if (delegated != plan.size())
		LOG(RPiDenoiseBstm, Warning)
			<< model_ << ": " << plan.size() - delegated << " of " << plan.size()
			<< " execution-plan nodes run on the CPU";

	if (interpreter_->inputs().size() != 1 || interpreter_->outputs().size() != 1) {
		LOG(RPiDenoiseBstm, Error) << model_ << " must have exactly one input and one output";
		return;
	}
	const TfLiteTensor *in = interpreter_->input_tensor(0);
	const TfLiteTensor *out = interpreter_->output_tensor(0);
	if (!checkTensor(in, "input", inInt8_, inScale_, inZp_) ||
	    !checkTensor(out, "output", outInt8_, outScale_, outZp_))
		return;
	if (out->dims->data[1] != in->dims->data[1] || out->dims->data[2] != in->dims->data[2]) {
		LOG(RPiDenoiseBstm, Error) << model_ << ": output size must match input size";
		return;
	}

	netH_ = in->dims->data[1];
	netW_ = in->dims->data[2];

	LOG(RPiDenoiseBstm, Info)
		<< "Loaded " << model_ << " (" << kChannels << "x" << netH_ << "x" << netW_
		<< "), " << delegated << "/" << plan.size() << " nodes on the NPU, BSTM_FLOAT="
		<< getenv("BSTM_FLOAT");
	auto describe = [](bool isInt8, float scale, int zp) {
		std::ostringstream os;
		if (isInt8)
			os << "int8 (scale " << scale << ", zero point " << zp << ")";
		else
			os << "float32";
		return os.str();
	};
	LOG(RPiDenoiseBstm, Info)
		<< "Input " << describe(inInt8_, inScale_, inZp_)
		<< ", output " << describe(outInt8_, outScale_, outZp_);

	init_ = true;
}

void DenoiseBstm::switchMode(CameraMode const &cameraMode, [[maybe_unused]] Metadata *metadata)
{
	width_ = cameraMode.width;
	height_ = cameraMode.height;

	if (!init_)
		return;

	/*
	 * Same geometry as denoise_ncnn.cpp: the 4 Bayer planes are half the
	 * raw size normally, or a quarter when downscale_ (2x2 averaged), and
	 * space_to_depth(2) halves those again. The model's input size is
	 * fixed, so anything smaller is zero-padded and anything larger is
	 * cropped (and left untouched).
	 */
	const unsigned bayerDiv = downscale_ ? 4 : 2;
	unsigned packW = width_ / bayerDiv / 2;
	unsigned packH = height_ / bayerDiv / 2;

	if (packW > netW_ || packH > netH_) {
		LOG(RPiDenoiseBstm, Error)
			<< "Bayer " << width_ << "x" << height_ << " (network " << packW << "x" << packH
			<< ") exceeds this network's size " << netW_ << "x" << netH_
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

	if (downscale_) {
		bayerBuffer_.assign(size_t(4) * bayerW_ * bayerH_, 0.0f);
		upscaleBuffer_.assign(size_t(4) * validW_ * validH_, 0.0f);
	} else {
		bayerBuffer_.clear();
		upscaleBuffer_.clear();
	}
	if (downscale_ && (inInt8_ || outInt8_))
		scratch_.assign(size_t(netW_) * netH_ * kChannels, 0.0f);
	else
		scratch_.clear();

	LOG(RPiDenoiseBstm, Info)
		<< "Bayer " << width_ << "x" << height_
		<< " -> " << kChannels << "x" << netH_ << "x" << netW_
		<< " (pack " << packW_ << "x" << packH_ << ", valid " << validW_ << "x" << validH_ << ")";
}

void DenoiseBstm::setMode([[maybe_unused]] DenoiseMode mode)
{
}

/*
 * Fill the network pixels beyond packW_ x packH_ with the encoding of 0.0
 * (0.0f, or an int8 input tensor's zero point), so the network sees black
 * rather than stale data there.
 */
template<typename T>
void DenoiseBstm::zeroBorder(T *in, T zero)
{
	const size_t rowLen = size_t(netW_) * kChannels;

	if (packW_ < netW_) {
		for (unsigned y = 0; y < packH_; ++y)
			std::fill(in + y * rowLen + size_t(packW_) * kChannels, in + (y + 1) * rowLen, zero);
	}
	std::fill(in + packH_ * rowLen, in + netH_ * rowLen, zero);
}

void DenoiseBstm::packBayer(const uint16_t *bayer16, unsigned stridePx, float *in)
{
	if (downscale_)
		packBayerDownscale(bayer16, stridePx, in);
	else
		packBayerFull(bayer16, stridePx, in);
}

/*
 * Pack the raw Bayer frame into the network's NHWC input: network pixel
 * (x,y) covers raw rows [4y,4y+4) and columns [4x,4x+4), raw offset
 * (dx,dy) going to channel rawToChannel(dx,dy). Each sample is raw / 65535
 * for a float tensor, or quantised as TFLite would for an int8 one:
 * round(raw / 65535 / scale) (half away from zero) + zero point, saturated.
 */
template<typename T>
void DenoiseBstm::packBayerFull(const uint16_t *bayer16, unsigned stridePx, T *in)
{
	constexpr bool isInt8 = std::is_same_v<T, int8_t>;
	const unsigned netW = netW_, packW = packW_, packH = packH_;
	const float k = isInt8 ? 1.0f / (65535.0f * inScale_) : 1.0f / 65535.0f;
	const int zp = inZp_;
	auto encode = [&](uint16_t raw) -> T {
		const float v = float(raw) * k;
		if constexpr (isInt8)
			return int8_t(std::clamp(int(std::lround(v)) + zp, -128, 127));
		else
			return v;
	};

#if defined(__ARM_NEON)
	const float32x4_t vK = vdupq_n_f32(k);
	const int32x4_t vZp = vdupq_n_s32(zp);
	auto toFloat = [&](uint16x4_t v) { return vmulq_f32(vcvtq_f32_u32(vmovl_u16(v)), vK); };
	[[maybe_unused]] auto toInt8x4 = [&](float32x4_t v) {
		return vqmovn_s32(vqaddq_s32(vcvtaq_s32_f32(v), vZp));
	};
#endif

	/*
	 * Single-threaded, as is depthToRaw(): this is memory-bound, and
	 * measured slower with more threads.
	 */
	for (unsigned y = 0; y < packH; ++y) {
		T *d = in + size_t(y) * netW * kChannels;
		const uint16_t *rows[4];
		for (unsigned dy = 0; dy < 4; ++dy)
			rows[dy] = bayer16 + size_t(4 * y + dy) * stridePx;

		unsigned x0 = 0;
#if defined(__ARM_NEON)
		/*
		 * Two network pixels (8 raw samples per row) at a time. Raw row dy
		 * of a pixel holds, in dx order, channels (see rawToChannel()):
		 * dy=0: 0,4,1,5   dy=1: 8,12,9,13   dy=2: 2,6,3,7   dy=3: 10,14,11,15
		 * so unzipping rows 0 and 2 gives channels 0-3 and 4-7, and rows 1
		 * and 3 give channels 8-11 and 12-15.
		 */
		for (; x0 + 2 <= packW; x0 += 2) {
			uint16x8_t r[4];
			for (unsigned dy = 0; dy < 4; ++dy)
				r[dy] = vld1q_u16(rows[dy] + 4 * x0);
			for (unsigned k2 = 0; k2 < 2; ++k2) {
				float32x4_t f[4];
				for (unsigned dy = 0; dy < 4; ++dy)
					f[dy] = toFloat(k2 ? vget_high_u16(r[dy]) : vget_low_u16(r[dy]));
				const float32x4x4_t c{ vuzp1q_f32(f[0], f[2]), vuzp2q_f32(f[0], f[2]),
						       vuzp1q_f32(f[1], f[3]), vuzp2q_f32(f[1], f[3]) };
				T *p = d + size_t(x0 + k2) * kChannels;
				if constexpr (isInt8) {
					const int16x8_t lo = vcombine_s16(toInt8x4(c.val[0]), toInt8x4(c.val[1]));
					const int16x8_t hi = vcombine_s16(toInt8x4(c.val[2]), toInt8x4(c.val[3]));
					vst1q_s8(p, vcombine_s8(vqmovn_s16(lo), vqmovn_s16(hi)));
				} else {
					vst1q_f32_x4(p, c);
				}
			}
		}
#endif
		for (unsigned dy = 0; dy < 4; ++dy) {
			const uint16_t *row = rows[dy];
			for (unsigned x = x0; x < packW; ++x) {
				T *p = d + size_t(x) * kChannels;
				const uint16_t *r = row + 4 * x;
				for (unsigned dx = 0; dx < 4; ++dx)
					p[rawToChannel(dx, dy)] = encode(r[dx]);
			}
		}
	}

	zeroBorder(in, isInt8 ? T(zp) : T(0));
}

/*
 * downscale_ variant: first box-average each Bayer colour over 2x2 (each
 * averaged sample covering a 4x4 raw block), then space_to_depth(2) as
 * usual -- so network pixel (x,y) covers raw rows [8y,8y+8) and columns
 * [8x,8x+8). Channel c*4 + i*2 + j holds colour c averaged over the 4x4
 * raw block at raw rows [8y+4i,8y+4i+4) and columns [8x+4j,8x+4j+4); e.g.
 * R at block-relative (0,0),(2,0),(0,2),(2,2).
 */
void DenoiseBstm::packBayerDownscale(const uint16_t *bayer16, unsigned stridePx, float *in)
{
	const unsigned netW = netW_, packW = packW_, packH = packH_;
	constexpr float invMax4 = 1.0f / (65535.0f * 4.0f);

#ifdef _OPENMP
#pragma omp parallel for num_threads(threads_) schedule(static)
#endif
	for (int y_ = 0; y_ < int(packH); ++y_) {
		const unsigned y = unsigned(y_);
		float *d = in + size_t(y) * netW * kChannels;
		for (unsigned i = 0; i < 2; ++i) {
			const uint16_t *rows[4];
			for (unsigned r = 0; r < 4; ++r)
				rows[r] = bayer16 + size_t(8 * y + 4 * i + r) * stridePx;
			for (unsigned x = 0; x < packW; ++x) {
				float *p = d + size_t(x) * kChannels;
				for (unsigned j = 0; j < 2; ++j) {
					const unsigned col = 8 * x + 4 * j;
					for (unsigned c = 0; c < 4; ++c) {
						const unsigned cx = col + (c & 1), cy = c >> 1;
						float sum = float(rows[cy][cx]) + rows[cy][cx + 2] +
							    rows[cy + 2][cx] + rows[cy + 2][cx + 2];
						p[c * 4 + i * 2 + j] = sum * invMax4;
					}
				}
			}
		}
	}

	zeroBorder(in, 0.0f);
}

void DenoiseBstm::unpackBayer(uint16_t *bayer16, unsigned stridePx, const float *out)
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

/* Pack the frame into the model's input tensor, float32 or int8. */
void DenoiseBstm::packInput(const uint16_t *bayer16, unsigned stridePx)
{
	TfLiteTensor *t = interpreter_->input_tensor(0);

	if (!inInt8_)
		packBayer(bayer16, stridePx, t->data.f);
	else if (!downscale_)
		packBayerFull(bayer16, stridePx, t->data.int8);
	else {
		/* Not speed-critical: pack as float, then quantise. */
		packBayerDownscale(bayer16, stridePx, scratch_.data());
		quantiseInput(scratch_.data(), t->data.int8);
	}
}

/* Write the model's output tensor, float32 or int8, back over the frame. */
void DenoiseBstm::unpackOutput(uint16_t *bayer16, unsigned stridePx)
{
	const TfLiteTensor *t = interpreter_->output_tensor(0);

	if (!outInt8_)
		unpackBayer(bayer16, stridePx, t->data.f);
	else if (!downscale_)
		depthToRaw(bayer16, stridePx, static_cast<const int8_t *>(t->data.int8));
	else {
		/* Not speed-critical: dequantise, then unpack as float. */
		dequantiseOutput(t->data.int8, scratch_.data());
		unpackBayer(bayer16, stridePx, scratch_.data());
	}
}

/* float -> int8 input tensor, as in packBayerFull(). */
void DenoiseBstm::quantiseInput(const float *v, int8_t *q)
{
	const size_t n = size_t(netW_) * netH_ * kChannels;
	const float k = 1.0f / inScale_;

	for (size_t i = 0; i < n; ++i)
		q[i] = int8_t(std::clamp(int(std::lround(v[i] * k)) + inZp_, -128, 127));
}

/* int8 output tensor -> float: (q - zero point) * scale. */
void DenoiseBstm::dequantiseOutput(const int8_t *q, float *v)
{
	const size_t n = size_t(netW_) * netH_ * kChannels;

	for (size_t i = 0; i < n; ++i)
		v[i] = float(int(q[i]) - outZp_) * outScale_;
}

/*
 * Inverse of packBayer(): write each network output pixel's 16 channels
 * straight back over its 4x4 raw block. A float output v is written as
 * v * 65535; an int8 output q is dequantised first, (q - zero point) *
 * scale, with the scale folded into the same fused multiply-add.
 */
template<typename T>
void DenoiseBstm::depthToRaw(uint16_t *bayer16, unsigned stridePx, const T *out)
{
	constexpr bool isInt8 = std::is_same_v<T, int8_t>;
	const unsigned netW = netW_, packW = packW_, packH = packH_;
	/* demo_: write back the right half only. */
	const unsigned xStart = demo_ ? packW / 2 : 0;
	const float scaleMax = isInt8 ? outScale_ * 65535.0f : 65535.0f;
	const int zp = outZp_;
	auto decode = [&](T q) -> float {
		if constexpr (isInt8)
			return float(int(q) - zp);
		else
			return q;
	};

#if defined(__ARM_NEON)
	const float32x4_t vScaleMax = vdupq_n_f32(scaleMax);
	const int16x8_t vZp = vdupq_n_s16(int16_t(zp));
	/* One network pixel's 16 channels as float32x4 x 4 (channels 0-3 ... 12-15). */
	auto loadPixel = [&](const T *p) -> float32x4x4_t {
		if constexpr (isInt8) {
			const int8x16_t q = vld1q_s8(p);
			/* q - zp is within [-255, 255]: exact in int16. */
			const int16x8_t lo = vsubq_s16(vmovl_s8(vget_low_s8(q)), vZp);
			const int16x8_t hi = vsubq_s16(vmovl_s8(vget_high_s8(q)), vZp);
			auto cvt = [](int16x4_t v) { return vcvtq_f32_s32(vmovl_s16(v)); };
			return { cvt(vget_low_s16(lo)), cvt(vget_high_s16(lo)),
				 cvt(vget_low_s16(hi)), cvt(vget_high_s16(hi)) };
		} else {
			return vld1q_f32_x4(p);
		}
	};
	/*
	 * One network pixel's 16 channels, loaded as 4 registers (channels
	 * 0-3, 4-7, 8-11, 12-15), hold raw row dy's 4 samples, in dx order, as:
	 * dy=0: zip1(c0-3, c4-7) = 0,4,1,5    dy=1: zip1(c8-11, c12-15) = 8,12,9,13
	 * dy=2: zip2(c0-3, c4-7) = 2,6,3,7    dy=3: zip2(c8-11, c12-15) = 10,14,11,15
	 * (see rawToChannel()).
	 */
	auto rowSamples = [](const float32x4x4_t &v, unsigned dy) {
		return dy == 0	 ? vzip1q_f32(v.val[0], v.val[1])
		       : dy == 1 ? vzip1q_f32(v.val[2], v.val[3])
		       : dy == 2 ? vzip2q_f32(v.val[0], v.val[1])
				 : vzip2q_f32(v.val[2], v.val[3]);
	};
	const float32x4_t vZero = vdupq_n_f32(0.0f);
#endif
	const bool residual = residual_;

	/*
	 * Single-threaded, as is packBayer(): this is memory-bound, and
	 * measured slower with more threads.
	 */
	for (unsigned y = 0; y < packH; ++y) {
		const T *s = out + size_t(y) * netW * kChannels;
		uint16_t *rows[4];
		for (unsigned dy = 0; dy < 4; ++dy)
			rows[dy] = bayer16 + size_t(4 * y + dy) * stridePx;

		unsigned x0 = xStart;
#if defined(__ARM_NEON)
		/* Two network pixels (8 raw samples per row) at a time. */
		for (; x0 + 2 <= packW; x0 += 2) {
			const T *p = s + size_t(x0) * kChannels;
			const float32x4x4_t a = loadPixel(p);
			const float32x4x4_t b = loadPixel(p + kChannels);
			for (unsigned dy = 0; dy < 4; ++dy) {
				uint16_t *r = rows[dy] + 4 * x0;
				float32x4_t baseA = vZero, baseB = vZero;
				if (residual) {
					/* Still the network's input: we write in place. */
					const uint16x8_t in = vld1q_u16(r);
					baseA = rawToFloat4(vget_low_u16(in));
					baseB = rawToFloat4(vget_high_u16(in));
				}
				vst1q_u16(r, vcombine_u16(toRaw4(rowSamples(a, dy), baseA, vScaleMax),
							  toRaw4(rowSamples(b, dy), baseB, vScaleMax)));
			}
		}
#endif
		for (unsigned dy = 0; dy < 4; ++dy) {
			uint16_t *row = rows[dy];
			for (unsigned x = x0; x < packW; ++x) {
				const T *p = s + size_t(x) * kChannels;
				uint16_t *r = row + 4 * x;
				for (unsigned dx = 0; dx < 4; ++dx)
					r[dx] = quantise(decode(p[rawToChannel(dx, dy)]),
							 residual ? float(r[dx]) : 0.0f, scaleMax);
			}
		}
	}
}

/*
 * downscale_ only: depth_to_space(2) the network's NHWC output back into
 * 4 quarter-res Bayer planes (bayerBuffer_, bayerW_ x bayerH_ each), ready
 * for upscaleChannel2x().
 */
void DenoiseBstm::depthToSpace(const float *out)
{
	const unsigned netW = netW_, packW = packW_, packH = packH_, bayerW = bayerW_;
	const size_t bayerPlane = size_t(bayerW_) * bayerH_;
	float *const dst = bayerBuffer_.data();

#ifdef _OPENMP
#pragma omp parallel for num_threads(threads_) schedule(static)
#endif
	for (int y_ = 0; y_ < int(packH); ++y_) {
		const unsigned y = unsigned(y_);
		const float *s = out + size_t(y) * netW * kChannels;
		for (unsigned x = 0; x < packW; ++x) {
			const float *p = s + size_t(x) * kChannels;
			for (unsigned c = 0; c < 4; ++c)
				for (unsigned i = 0; i < 2; ++i) {
					float *d = dst + c * bayerPlane + size_t(2 * y + i) * bayerW + 2 * x;
					d[0] = p[c * 4 + i * 2];
					d[1] = p[c * 4 + i * 2 + 1];
				}
		}
	}
}

void DenoiseBstm::interleaveToRaw(uint16_t *bayer16, unsigned stridePx, const float *R,
				  const float *G1, const float *G2, const float *B,
				  unsigned srcStride)
{
	const unsigned validW = validW_, validH = validH_;

	const bool residual = residual_;

#if defined(__ARM_NEON)
	const float32x4_t vScaleMax = vdupq_n_f32(65535.0f);
	const float32x4_t vZero = vdupq_n_f32(0.0f);
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

		/* demo_: write back the right half only. */
		unsigned x = demo_ ? validW / 2 : 0;
#if defined(__ARM_NEON)
		for (; x + 4 <= validW; x += 4) {
			float32x4_t b0 = vZero, b1 = vZero, b2 = vZero, b3 = vZero;
			if (residual) {
				/* Still the network's input: we write in place. */
				const uint16x4x2_t in0 = vld2_u16(row0 + size_t(x) * 2);
				const uint16x4x2_t in1 = vld2_u16(row1 + size_t(x) * 2);
				b0 = rawToFloat4(in0.val[0]);
				b1 = rawToFloat4(in0.val[1]);
				b2 = rawToFloat4(in1.val[0]);
				b3 = rawToFloat4(in1.val[1]);
			}
			const uint16x4x2_t a{ toRaw4(vld1q_f32(s0 + x), b0, vScaleMax),
					      toRaw4(vld1q_f32(s1 + x), b1, vScaleMax) };
			const uint16x4x2_t b{ toRaw4(vld1q_f32(s2 + x), b2, vScaleMax),
					      toRaw4(vld1q_f32(s3 + x), b3, vScaleMax) };
			vst2_u16(row0 + size_t(x) * 2, a);
			vst2_u16(row1 + size_t(x) * 2, b);
		}
#endif
		for (; x < validW; ++x) {
			uint16_t *r0 = row0 + 2 * x, *r1 = row1 + 2 * x;
			r0[0] = quantise(s0[x], residual ? float(r0[0]) : 0.0f);
			r0[1] = quantise(s1[x], residual ? float(r0[1]) : 0.0f);
			r1[0] = quantise(s2[x], residual ? float(r1[0]) : 0.0f);
			r1[1] = quantise(s3[x], residual ? float(r1[1]) : 0.0f);
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
void DenoiseBstm::upscaleChannel2x(const float *src, unsigned srcStride, unsigned srcW,
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

void DenoiseBstm::prepare(Metadata *imageMetadata)
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
			LOG(RPiDenoiseBstm, Error)
				<< "raw buffer " << bayer.second.size() << " bytes = " << rowBytes
				<< " B/row, need >= " << minRow << " for " << width_
				<< " px wide. Compressed raw? Skipping denoise_bstm.";
		}
		dmabufSyncEnd(bayer.first);
		return;
	}

	packInput(bayer16, stridePx);

	if (interpreter_->Invoke() != kTfLiteOk) {
		LOG(RPiDenoiseBstm, Error) << "NPU inference failed";
		dmabufSyncEnd(bayer.first);
		return;
	}

	unpackOutput(bayer16, stridePx);

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
	return (Algorithm *)new DenoiseBstm(controller);
}
static RegisterAlgorithm reg(NAME, &create);
