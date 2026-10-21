/* SPDX-License-Identifier: BSD-2-Clause */
/*
 * Copyright (C) 2026, Raspberry Pi Ltd
 *
 * Recurrent full-frame denoise control algorithm on the BSTM NPU
 */

#include "denoise_bstm.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <errno.h>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>

#include <linux/dma-buf.h>

#include <libcamera/base/log.h>
#include <libcamera/base/shared_fd.h>
#include <libcamera/base/span.h>

#include <tensorflow/lite/builtin_ops.h>
#include <tensorflow/lite/delegates/external/external_delegate.h>
#include <tensorflow/lite/interpreter_builder.h>
#include <tensorflow/lite/kernels/register.h>

#include "../denoise_status.h"
#include "../device_status.h"

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

/*
 * The network's x and prev inputs, and its r output, are the 4 Bayer planes
 * (raw 2x2-cell positions (0,0), (1,0), (0,1), (1,1)) further packed by
 * space_to_depth(2): channel c*4 + i*2 + j holds Bayer plane c's (row i,
 * column j) sub-position within each 2x2 block of that plane. The tensors
 * are NHWC, so each network pixel's 16 channels are consecutive, and a
 * network pixel covers one 4x4 raw block. This returns the channel holding
 * raw offset (dx, dy) within that block.
 */
constexpr unsigned rawToChannel(unsigned dx, unsigned dy)
{
	return 4 * (2 * (dy & 1) + (dx & 1)) + 2 * (dy >> 1) + (dx >> 1);
}

constexpr unsigned kChannels = 16;

/* 12-bit raw data, padded with zeros at the bottom: tables are indexed by raw >> 4. */
constexpr unsigned kRawShift = 4;
constexpr unsigned kTableSize = 65536 >> kRawShift;

using Clock = std::chrono::steady_clock;

double msSince(Clock::time_point t)
{
	return std::chrono::duration<double, std::milli>(Clock::now() - t).count();
}

/*
 * The current frame's square root for network pixel x, in channel order,
 * looked up from the raw values (rows[dy] being raw row 4y + dy).
 */
/*
 * Scale the classic PISP denoise stage's status (tag) by factor, using
 * scale(status, factor), or with a factor of 0 erase it so the stage doesn't
 * run at all on a frame we've already denoised. 1 leaves it untouched.
 */
template<typename T, typename F>
void scaleStatus(Metadata *imageMetadata, const char *tag, double factor, F scale)
{
	if (factor == 0.0) {
		imageMetadata->erase(tag);
		return;
	}
	if (factor == 1.0)
		return;

	T status;
	if (imageMetadata->get(tag, status) == 0) {
		scale(status, factor);
		imageMetadata->set(tag, status);
	}
}

inline void gatherSqrt(uint16_t *const rows[4], unsigned x, const float *table, float *xf)
{
	for (unsigned dy = 0; dy < 4; ++dy) {
		const uint16_t *r = rows[dy] + 4 * x;
		for (unsigned dx = 0; dx < 4; ++dx)
			xf[rawToChannel(dx, dy)] = table[r[dx] >> kRawShift];
	}
}

#if defined(__ARM_NEON)
/*
 * Square network pixel x's 16 channels (in [0, 1]) back to linear, scale to
 * 16 bits and write them over its 4x4 raw block. FCVTAU rounds to nearest
 * and saturates, and vqmovn_u32 saturates again to uint16, so no clamp is
 * needed. The 4 channel registers hold raw row dy's samples, in dx order,
 * as (see rawToChannel()):
 * dy=0: zip1(c0-3, c4-7) = 0,4,1,5    dy=1: zip1(c8-11, c12-15) = 8,12,9,13
 * dy=2: zip2(c0-3, c4-7) = 2,6,3,7    dy=3: zip2(c8-11, c12-15) = 10,14,11,15
 */
inline void writeRaw(uint16_t *const rows[4], unsigned x, const float32x4x4_t &out)
{
	const float32x4_t vRawMax = vdupq_n_f32(65535.0f);
	float32x4_t lin[4];
	for (unsigned i = 0; i < 4; ++i)
		lin[i] = vmulq_f32(out.val[i], vmulq_f32(out.val[i], vRawMax));
	const float32x4_t rowv[4] = { vzip1q_f32(lin[0], lin[1]), vzip1q_f32(lin[2], lin[3]),
				      vzip2q_f32(lin[0], lin[1]), vzip2q_f32(lin[2], lin[3]) };
	for (unsigned dy = 0; dy < 4; ++dy)
		vst1_u16(rows[dy] + 4 * x, vqmovn_u32(vcvtaq_u32_f32(rowv[dy])));
}
#else
inline void writeRaw(uint16_t *const rows[4], unsigned x, const float *out)
{
	for (unsigned dy = 0; dy < 4; ++dy) {
		uint16_t *r = rows[dy] + 4 * x;
		for (unsigned dx = 0; dx < 4; ++dx) {
			const float o = out[rawToChannel(dx, dy)];
			r[dx] = uint16_t(std::lround(std::min(o * (o * 65535.0f), 65535.0f)));
		}
	}
}
#endif

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
	sdnFactor_ = params["sdn_factor"].get<double>(0.0);
	tdnFactor_ = params["tdn_factor"].get<double>(0.0);
	cdnFactor_ = params["cdn_factor"].get<double>(0.0);

	threads_ = params["threads"].get<int>(2);
	cpu_ = params["cpu"].get<bool>(false);
	cpuThreads_ = params["cpu_threads"].get<int>(3);
	demo_ = params["demo"].get<bool>(false);
	resetRatio_ = params["reset_ratio"].get<double>(2.0);

	dumpDir_ = params["dump_dir"].get<std::string>("");
	dumpFrames_ = params["dump_frames"].get<unsigned>(0);

	if (model_.empty()) {
		LOG(RPiDenoiseBstm, Error) << "\"model\" must be supplied";
		return -EINVAL;
	}

	for (const char *stage : { "sdn", "tdn", "cdn" }) {
		const std::string old = std::string(stage) + "_disable";
		if (params.contains(old))
			LOG(RPiDenoiseBstm, Warning)
				<< "\"" << old << "\" is no longer supported and is ignored; use \""
				<< stage << "_factor\" (0 disables the stage, 1 leaves it as it is)";
	}

	if (sdnFactor_ < 0.0 || tdnFactor_ < 0.0 || cdnFactor_ < 0.0) {
		LOG(RPiDenoiseBstm, Error)
			<< "\"sdn_factor\", \"tdn_factor\" and \"cdn_factor\" must not be negative";
		return -EINVAL;
	}

	if (floatFormat_ != "bf16" && floatFormat_ != "fp16") {
		LOG(RPiDenoiseBstm, Error)
			<< "\"float_format\" must be \"bf16\" or \"fp16\", not \"" << floatFormat_ << "\"";
		return -EINVAL;
	}

	if (!(resetRatio_ > 1.0)) {
		LOG(RPiDenoiseBstm, Error) << "\"reset_ratio\" must be greater than 1";
		return -EINVAL;
	}

	return 0;
}

/*
 * A model input or output must be [1, H, W, channels], float32 or int8. An
 * int8 tensor must be quantised per tensor (a single scale and zero point),
 * which are returned in scale and zp.
 */
bool DenoiseBstm::checkTensor(const TfLiteTensor *t, const char *what, int channels,
				      bool &isInt8, float &scale, int &zp)
{
	if (!t || !t->dims || t->dims->size != 4 || t->dims->data[0] != 1 ||
	    t->dims->data[3] != channels) {
		LOG(RPiDenoiseBstm, Error)
			<< model_ << ": " << what << " must be [1, H, W, " << channels << "]";
		return false;
	}

	if (t->type == kTfLiteFloat32) {
		isInt8 = false;
		return true;
	}

	if (t->type != kTfLiteInt8) {
		LOG(RPiDenoiseBstm, Error)
			<< model_ << ": " << what << " must be float32 or int8, not "
			<< TfLiteTypeGetName(t->type);
		return false;
	}

	const auto *q = static_cast<const TfLiteAffineQuantization *>(t->quantization.params);
	if (t->quantization.type != kTfLiteAffineQuantization || !q || !q->scale ||
	    q->scale->size != 1 || !(t->params.scale > 0.0f)) {
		LOG(RPiDenoiseBstm, Error)
			<< model_ << ": " << what << " must have a single (per-tensor) scale";
		return false;
	}

	isInt8 = true;
	scale = t->params.scale;
	zp = t->params.zero_point;
	return true;
}

void DenoiseBstm::initialise()
{
	flatbuffer_ = tflite::FlatBufferModel::BuildFromFile(model_.c_str());
	if (!flatbuffer_) {
		LOG(RPiDenoiseBstm, Error) << "Failed to load TFLite model: " << model_;
		return;
	}

	/*
	 * On the CPU, BuiltinOpResolverWithXNNPACK asks TFLite to apply its
	 * default XNNPACK delegate, for float and int8 alike, which is about 3x
	 * faster for this network. Debian's libtensorflow-lite is built without
	 * XNNPACK, though (its MaybeCreateXNNPACKDelegate() returns nullptr),
	 * so there this runs on TFLite's builtin kernels, and the log shows 0
	 * nodes delegated. The resolver must outlive AllocateTensors(), where
	 * a default delegate would be applied.
	 */
	std::unique_ptr<tflite::ops::builtin::BuiltinOpResolver> resolver;
	if (cpu_)
		resolver = std::make_unique<tflite::ops::builtin::BuiltinOpResolverWithXNNPACK>();
	else
		resolver = std::make_unique<tflite::ops::builtin::BuiltinOpResolver>();
	tflite::InterpreterBuilder(*flatbuffer_, *resolver)(&interpreter_);
	if (!interpreter_) {
		LOG(RPiDenoiseBstm, Error) << "Failed to build TFLite interpreter for " << model_;
		return;
	}
	/* This also sets the threads of TFLite's default CPU (XNNPACK) delegate. */
	interpreter_->SetNumThreads(cpu_ ? cpuThreads_ : threads_);

	if (!cpu_) {
		/*
		 * Read by the bstm driver when the delegate opens the NPU, so it
		 * must be set before then; the inputs' type is already known.
		 */
		if (!interpreter_->inputs().empty() &&
		    interpreter_->tensor(interpreter_->inputs()[0])->type == kTfLiteFloat32)
			setenv("BSTM_FLOAT", floatFormat_.c_str(), 0);

		TfLiteExternalDelegateOptions options =
			TfLiteExternalDelegateOptionsDefault(delegate_.c_str());
		tfDelegate_ = TfLiteExternalDelegateCreate(&options);
		if (!tfDelegate_) {
			LOG(RPiDenoiseBstm, Error)
				<< "Failed to create the Teflon delegate from " << delegate_;
			return;
		}

		/* This is where the driver compiles the model (or finds it in its disk cache). */
		if (interpreter_->ModifyGraphWithDelegate(tfDelegate_) != kTfLiteOk) {
			LOG(RPiDenoiseBstm, Error)
				<< "Failed to apply the Teflon delegate to " << model_;
			return;
		}
	}

	if (interpreter_->AllocateTensors() != kTfLiteOk) {
		LOG(RPiDenoiseBstm, Error) << "Failed to allocate tensors for " << model_;
		return;
	}

	/*
	 * With cpu_, any delegate nodes are TFLite's own default CPU delegate
	 * (XNNPACK), applied by AllocateTensors(); otherwise they're the NPU.
	 */
	unsigned delegated = 0;
	const std::vector<int> &plan = interpreter_->execution_plan();
	for (int node : plan) {
		const auto *nr = interpreter_->node_and_registration(node);
		if (nr && nr->second.builtin_code == kTfLiteBuiltinDelegate)
			delegated++;
	}
	if (!cpu_ && !delegated)
		LOG(RPiDenoiseBstm, Warning) << "No part of " << model_ << " runs on the NPU";
	else if (!cpu_ && delegated != plan.size())
		LOG(RPiDenoiseBstm, Warning)
			<< model_ << ": " << plan.size() - delegated << " of " << plan.size()
			<< " execution-plan nodes run on the CPU";

	if (interpreter_->inputs().size() != 2 || interpreter_->outputs().size() != 2) {
		LOG(RPiDenoiseBstm, Error)
			<< model_ << " must have exactly two inputs (x, prev) and two outputs (r, a)";
		return;
	}

	/*
	 * The inputs have the same shape, so they're taken in the order the
	 * interpreter lists them, x then prev (convert_temporal.py --verify
	 * checks this). The outputs are told apart by their last dimension.
	 */
	inX_ = interpreter_->inputs()[0];
	inPrev_ = interpreter_->inputs()[1];
	for (int idx : interpreter_->outputs()) {
		const TfLiteTensor *t = interpreter_->tensor(idx);
		const int c = t && t->dims && t->dims->size == 4 ? t->dims->data[3] : 0;
		if (c == int(kChannels))
			outR_ = idx;
		else if (c == 1)
			outA_ = idx;
	}
	if (outR_ < 0 || outA_ < 0) {
		LOG(RPiDenoiseBstm, Error)
			<< model_ << ": outputs must be r [1, H, W, 16] and a [1, H, W, 1]";
		return;
	}

	const TfLiteTensor *x = interpreter_->tensor(inX_);
	const TfLiteTensor *prev = interpreter_->tensor(inPrev_);
	const TfLiteTensor *r = interpreter_->tensor(outR_);
	const TfLiteTensor *a = interpreter_->tensor(outA_);
	bool int8[4];
	if (!checkTensor(x, "input x", kChannels, int8[0], xScale_, xZp_) ||
	    !checkTensor(prev, "input prev", kChannels, int8[1], prevScale_, prevZp_) ||
	    !checkTensor(r, "output r", kChannels, int8[2], rScale_, rZp_) ||
	    !checkTensor(a, "output a", 1, int8[3], aScale_, aZp_))
		return;
	if (int8[1] != int8[0] || int8[2] != int8[0] || int8[3] != int8[0]) {
		LOG(RPiDenoiseBstm, Error)
			<< model_ << ": inputs and outputs must be all int8 or all float32";
		return;
	}
	isFloat_ = !int8[0];

	netH_ = x->dims->data[1];
	netW_ = x->dims->data[2];
	for (const TfLiteTensor *t : { prev, r, a }) {
		if (unsigned(t->dims->data[1]) != netH_ || unsigned(t->dims->data[2]) != netW_) {
			LOG(RPiDenoiseBstm, Error)
				<< model_ << ": all inputs and outputs must be the same size";
			return;
		}
	}

	buildTables();

	LOG(RPiDenoiseBstm, Info)
		<< "Loaded " << model_ << " (" << kChannels << "x" << netH_ << "x" << netW_
		<< "), " << delegated << "/" << plan.size()
		<< (cpu_ ? " nodes delegated to XNNPACK, all on the CPU (" : " nodes on the NPU")
		<< (cpu_ ? std::to_string(cpuThreads_) + " threads)" : "");
	if (isFloat_ && !cpu_)
		LOG(RPiDenoiseBstm, Info)
			<< "float32 model, BSTM_FLOAT=" << getenv("BSTM_FLOAT");
	else if (isFloat_)
		LOG(RPiDenoiseBstm, Info) << "float32 model";
	else
		LOG(RPiDenoiseBstm, Info)
			<< "int8 model: x (" << xScale_ << ", " << xZp_ << "), prev (" << prevScale_
			<< ", " << prevZp_ << "), r (" << rScale_ << ", " << rZp_ << "), a (" << aScale_
			<< ", " << aZp_ << ")";

	init_ = true;
}

/*
 * The input tables don't depend on the model except through x's
 * quantisation; the sigmoid table belongs to a's scale and zero point. All
 * are computed in double, then rounded once. A float model needs only
 * xfTable_: its a output isn't limited to 256 values, so the sigmoid is
 * computed directly.
 */
void DenoiseBstm::buildTables()
{
	xfTable_.resize(kTableSize);
	for (unsigned i = 0; i < kTableSize; i++)
		xfTable_[i] = float(std::sqrt(double(i << kRawShift) / 65535.0));

	if (isFloat_) {
		xqTable_.clear();
		alphaTable_.clear();
		return;
	}

	xqTable_.resize(kTableSize);
	for (unsigned i = 0; i < kTableSize; i++) {
		const double v = std::sqrt(double(i << kRawShift) / 65535.0);
		xqTable_[i] = int8_t(std::clamp(int(std::lround(v / xScale_)) + xZp_, -128, 127));
	}

	alphaTable_.resize(256);
	for (int q = -128; q <= 127; q++)
		alphaTable_[q + 128] = float(1.0 / (1.0 + std::exp(-double(q - aZp_) * aScale_)));
}

void DenoiseBstm::switchMode(CameraMode const &cameraMode, [[maybe_unused]] Metadata *metadata)
{
	width_ = cameraMode.width;
	height_ = cameraMode.height;
	active_ = false;

	if (!init_)
		return;

	/*
	 * One network pixel per 4x4 raw block. The model's size is fixed, so
	 * a smaller frame is padded (with the inputs' encoding of zero) and a
	 * larger one cropped, the rest being left untouched.
	 */
	unsigned packW = width_ / 4;
	unsigned packH = height_ / 4;
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
	if (!packW_ || !packH_)
		return;

	if (isFloat_)
		prev_.clear();
	else
		prev_.assign(size_t(packW_) * packH_ * kChannels, 0);

	/*
	 * TFLite never writes its input tensors, so whatever lies outside the
	 * packed region stays as filled here for as long as this mode lasts.
	 */
	const size_t n = size_t(netW_) * netH_ * kChannels;
	TfLiteTensor *x = interpreter_->tensor(inX_);
	if (isFloat_)
		std::fill(x->data.f, x->data.f + n, 0.0f);
	else
		std::fill(x->data.int8, x->data.int8 + n, int8_t(xZp_));

	dumpCount_ = 0;
	active_ = true;
	reset();

	LOG(RPiDenoiseBstm, Info)
		<< "Bayer " << width_ << "x" << height_ << " -> " << kChannels << "x" << netH_
		<< "x" << netW_ << " (pack " << packW_ << "x" << packH_ << ")";
}

void DenoiseBstm::setMode([[maybe_unused]] DenoiseMode mode)
{
}

/*
 * Restart from frame 0: the network's prev input becomes real zeros (for
 * int8, its zero point, -128, not int8 0), and the next blend ignores alpha.
 */
void DenoiseBstm::reset()
{
	const size_t n = size_t(netW_) * netH_ * kChannels;
	TfLiteTensor *prev = interpreter_->tensor(inPrev_);
	if (isFloat_)
		std::fill(prev->data.f, prev->data.f + n, 0.0f);
	else
		std::fill(prev->data.int8, prev->data.int8 + n, int8_t(prevZp_));

	firstFrame_ = true;
	lastExposure_ = 0.0;
}

/*
 * Restart on a large jump in exposure time x analogue gain, which the
 * network was never trained across. Returns true if we reset.
 */
bool DenoiseBstm::checkContinuity(Metadata *imageMetadata)
{
	DeviceStatus deviceStatus;
	if (imageMetadata->get("device.status", deviceStatus) != 0)
		return false;

	const double exposure = deviceStatus.exposureTime.get<std::micro>() * deviceStatus.analogueGain;
	if (!(exposure > 0.0))
		return false;

	bool jump = false;
	if (lastExposure_ > 0.0) {
		const double ratio = exposure / lastExposure_;
		jump = ratio > resetRatio_ || ratio < 1.0 / resetRatio_;
	}
	if (jump) {
		LOG(RPiDenoiseBstm, Debug)
			<< "Exposure x gain jumped from " << lastExposure_ << " to " << exposure
			<< " -- restarting";
		reset();
	}
	lastExposure_ = exposure;

	return jump;
}

/*
 * Pack the raw Bayer frame into the network's x input: network pixel (x,y)
 * covers raw rows [4y,4y+4) and columns [4x,4x+4), raw offset (dx,dy) going
 * to channel rawToChannel(dx,dy), each sample being its square root from
 * table: quantised (xqTable_) for an int8 model, or in float (xfTable_).
 *
 * The 16 scalar table lookups per network pixel make this compute-bound
 * rather than memory-bound, so it is threaded (2.1 ms -> 1.4 ms with 2
 * threads on a Pi 5).
 */
template<typename T>
void DenoiseBstm::packInput(const uint16_t *bayer16, unsigned stridePx, const T *table,
				    T *in)
{
	const unsigned netW = netW_, packW = packW_, packH = packH_;

#ifdef _OPENMP
#pragma omp parallel for num_threads(threads_) schedule(static)
#endif
	for (int y_ = 0; y_ < int(packH); ++y_) {
		const unsigned y = unsigned(y_);
		T *d = in + size_t(y) * netW * kChannels;
		const uint16_t *rows[4];
		for (unsigned dy = 0; dy < 4; ++dy)
			rows[dy] = bayer16 + size_t(4 * y + dy) * stridePx;

		for (unsigned x = 0; x < packW; ++x) {
			T *p = d + size_t(x) * kChannels;
			for (unsigned dy = 0; dy < 4; ++dy) {
				const uint16_t *r = rows[dy] + 4 * x;
				for (unsigned dx = 0; dx < 4; ++dx)
					p[rawToChannel(dx, dy)] = table[r[dx] >> kRawShift];
			}
		}
	}
}

/*
 * Complete the frame from the network's outputs, all in float and in
 * square-root space, for each network pixel and channel k:
 *
 *   s    = clamp(xf + r, 0, 1)
 *   out  = s                                   on the first frame
 *   out  = prev + alpha * (s - prev)           otherwise
 *
 * with xf the current frame's square root (looked up again from the raw
 * values, which are still in the buffer), and alpha = sigmoid(a), one per
 * network pixel. out is kept for the next frame, and written back over the
 * raw buffer squared, as round(out^2 * 65535).
 *
 * int8 model: r and a are dequantised, (q - zero point) * scale, and alpha
 * comes from alphaTable_. out is kept in prev_ (as 16-bit fixed point,
 * round(out * 65535)), and written quantised to the network's prev input
 * ready for the next frame.
 */
void DenoiseBstm::blendInt8(uint16_t *bayer16, unsigned stridePx)
{
	const unsigned netW = netW_, packW = packW_, packH = packH_;
	/* demo_: write back the right half only (but update everything else). */
	const unsigned xStart = demo_ ? packW / 2 : 0;
	const bool first = firstFrame_;
	const float *xfTable = xfTable_.data();
	const float *alphaTable = alphaTable_.data();
	const int8_t *rOut = interpreter_->tensor(outR_)->data.int8;
	const int8_t *aOut = interpreter_->tensor(outA_)->data.int8;
	int8_t *prevIn = interpreter_->tensor(inPrev_)->data.int8;
	uint16_t *prevBuf = prev_.data();
	const float rScale = rScale_;
	const int rZp = rZp_;
	const float prevInvScale = 1.0f / prevScale_;
	const int prevZp = prevZp_;

#if defined(__ARM_NEON)
	const float32x4_t vRScale = vdupq_n_f32(rScale);
	const int16x8_t vRZp = vdupq_n_s16(int16_t(rZp));
	const float32x4_t vZero = vdupq_n_f32(0.0f);
	const float32x4_t vOne = vdupq_n_f32(1.0f);
	const float32x4_t vPrevInvScale = vdupq_n_f32(prevInvScale);
	const int32x4_t vPrevZp = vdupq_n_s32(prevZp);
	const float32x4_t vRawMax = vdupq_n_f32(65535.0f);
	const float32x4_t vInvMax = vdupq_n_f32(1.0f / 65535.0f);
#endif

#ifdef _OPENMP
#pragma omp parallel for num_threads(threads_) schedule(static)
#endif
	for (int y_ = 0; y_ < int(packH); ++y_) {
		const unsigned y = unsigned(y_);
		const int8_t *rRow = rOut + size_t(y) * netW * kChannels;
		const int8_t *aRow = aOut + size_t(y) * netW;
		int8_t *pqRow = prevIn + size_t(y) * netW * kChannels;
		uint16_t *pRow = prevBuf + size_t(y) * packW * kChannels;
		uint16_t *rows[4];
		for (unsigned dy = 0; dy < 4; ++dy)
			rows[dy] = bayer16 + size_t(4 * y + dy) * stridePx;

		for (unsigned x = 0; x < packW; ++x) {
			alignas(16) float xf[kChannels];
			gatherSqrt(rows, x, xfTable, xf);
			const float alpha = alphaTable[aRow[x] + 128];
			uint16_t *p = pRow + size_t(x) * kChannels;
			int8_t *pq = pqRow + size_t(x) * kChannels;

#if defined(__ARM_NEON)
			/* r - zr is within [-255, 255]: exact in int16. */
			const int8x16_t rq = vld1q_s8(rRow + size_t(x) * kChannels);
			const int16x8_t rLo = vsubq_s16(vmovl_s8(vget_low_s8(rq)), vRZp);
			const int16x8_t rHi = vsubq_s16(vmovl_s8(vget_high_s8(rq)), vRZp);
			auto cvt = [](int16x4_t v) { return vcvtq_f32_s32(vmovl_s16(v)); };
			const float32x4_t rf[4] = { cvt(vget_low_s16(rLo)), cvt(vget_high_s16(rLo)),
						    cvt(vget_low_s16(rHi)), cvt(vget_high_s16(rHi)) };
			const float32x4x4_t xv = vld1q_f32_x4(xf);
			float32x4x4_t pv;
			if (!first) {
				const uint16x8x2_t pu = vld1q_u16_x2(p);
				auto cvtPrev = [&](uint16x4_t v) {
					return vmulq_f32(vcvtq_f32_u32(vmovl_u16(v)), vInvMax);
				};
				pv = { cvtPrev(vget_low_u16(pu.val[0])), cvtPrev(vget_high_u16(pu.val[0])),
				       cvtPrev(vget_low_u16(pu.val[1])), cvtPrev(vget_high_u16(pu.val[1])) };
			}
			const float32x4_t vAlpha = vdupq_n_f32(alpha);

			float32x4x4_t out;
			for (unsigned i = 0; i < 4; ++i) {
				/* The residual's dequantising multiply is fused with the skip add. */
				float32x4_t s = vfmaq_f32(xv.val[i], rf[i], vRScale);
				s = vminq_f32(vmaxq_f32(s, vZero), vOne);
				out.val[i] = first ? s
						   : vfmaq_f32(pv.val[i], vAlpha, vsubq_f32(s, pv.val[i]));
			}
			/* Kept to 16 bits: out is in [0, 1], so no clamp is needed (see writeRaw()). */
			uint16x4_t pu[4];
			for (unsigned i = 0; i < 4; ++i)
				pu[i] = vqmovn_u32(vcvtaq_u32_f32(vmulq_f32(out.val[i], vRawMax)));
			vst1q_u16_x2(p, uint16x8x2_t{ vcombine_u16(pu[0], pu[1]), vcombine_u16(pu[2], pu[3]) });

			/*
			 * The next frame's prev input, quantised as TFLite would:
			 * round half away from zero (FCVTAS), plus the zero point,
			 * saturated to int8.
			 */
			int16x4_t q16[4];
			for (unsigned i = 0; i < 4; ++i)
				q16[i] = vqmovn_s32(vqaddq_s32(vcvtaq_s32_f32(vmulq_f32(out.val[i], vPrevInvScale)),
							       vPrevZp));
			vst1q_s8(pq, vcombine_s8(vqmovn_s16(vcombine_s16(q16[0], q16[1])),
						 vqmovn_s16(vcombine_s16(q16[2], q16[3]))));
#else
			const int8_t *rq = rRow + size_t(x) * kChannels;
			float out[kChannels];
			for (unsigned k = 0; k < kChannels; ++k) {
				float s = std::fma(float(int(rq[k]) - rZp), rScale, xf[k]);
				s = std::clamp(s, 0.0f, 1.0f);
				const float pk = float(p[k]) * (1.0f / 65535.0f);
				out[k] = first ? s : std::fma(alpha, s - pk, pk);
				p[k] = uint16_t(std::lround(out[k] * 65535.0f));
				pq[k] = int8_t(std::clamp(int(std::lround(out[k] * prevInvScale)) + prevZp,
							  -128, 127));
			}
#endif

			if (x >= xStart)
				writeRaw(rows, x, out);
		}
	}
}

/*
 * As blendInt8(), for a float model: r and a need no dequantising, alpha is
 * computed directly (one exp per network pixel), and out is kept in the
 * network's float prev input itself, which is read back next frame. TFLite
 * never writes its inputs, and the Teflon delegate converts them to
 * bf16/fp16 without touching the originals.
 */
void DenoiseBstm::blendFloat(uint16_t *bayer16, unsigned stridePx)
{
	const unsigned netW = netW_, packW = packW_, packH = packH_;
	/* demo_: write back the right half only (but update everything else). */
	const unsigned xStart = demo_ ? packW / 2 : 0;
	const bool first = firstFrame_;
	const float *xfTable = xfTable_.data();
	const float *rOut = interpreter_->tensor(outR_)->data.f;
	const float *aOut = interpreter_->tensor(outA_)->data.f;
	float *prevIn = interpreter_->tensor(inPrev_)->data.f;

#if defined(__ARM_NEON)
	const float32x4_t vZero = vdupq_n_f32(0.0f);
	const float32x4_t vOne = vdupq_n_f32(1.0f);
#endif

#ifdef _OPENMP
#pragma omp parallel for num_threads(threads_) schedule(static)
#endif
	for (int y_ = 0; y_ < int(packH); ++y_) {
		const unsigned y = unsigned(y_);
		const float *rRow = rOut + size_t(y) * netW * kChannels;
		const float *aRow = aOut + size_t(y) * netW;
		float *pRow = prevIn + size_t(y) * netW * kChannels;
		uint16_t *rows[4];
		for (unsigned dy = 0; dy < 4; ++dy)
			rows[dy] = bayer16 + size_t(4 * y + dy) * stridePx;

		for (unsigned x = 0; x < packW; ++x) {
			alignas(16) float xf[kChannels];
			gatherSqrt(rows, x, xfTable, xf);
			const float alpha = 1.0f / (1.0f + std::exp(-aRow[x]));
			const float *r = rRow + size_t(x) * kChannels;
			float *p = pRow + size_t(x) * kChannels;

#if defined(__ARM_NEON)
			const float32x4x4_t xv = vld1q_f32_x4(xf);
			const float32x4x4_t rv = vld1q_f32_x4(r);
			float32x4x4_t pv;
			if (!first)
				pv = vld1q_f32_x4(p);
			const float32x4_t vAlpha = vdupq_n_f32(alpha);

			float32x4x4_t out;
			for (unsigned i = 0; i < 4; ++i) {
				float32x4_t s = vaddq_f32(xv.val[i], rv.val[i]);
				s = vminq_f32(vmaxq_f32(s, vZero), vOne);
				out.val[i] = first ? s
						   : vfmaq_f32(pv.val[i], vAlpha, vsubq_f32(s, pv.val[i]));
			}
			vst1q_f32_x4(p, out);
#else
			float out[kChannels];
			for (unsigned k = 0; k < kChannels; ++k) {
				const float s = std::clamp(xf[k] + r[k], 0.0f, 1.0f);
				out[k] = first ? s : std::fma(alpha, s - p[k], p[k]);
				p[k] = out[k];
			}
#endif

			if (x >= xStart)
				writeRaw(rows, x, out);
		}
	}
}

/* Debug only: see dumpDir_. */
void DenoiseBstm::dumpFrame(const uint16_t *bayer16, unsigned stridePx, const char *what)
{
	std::ostringstream name;
	name << dumpDir_ << "/" << what << "_" << std::setw(4) << std::setfill('0') << dumpCount_
	     << ".raw";
	std::ofstream f(name.str(), std::ios::binary);
	for (unsigned y = 0; y < height_; ++y)
		f.write(reinterpret_cast<const char *>(bayer16 + size_t(y) * stridePx),
			width_ * sizeof(uint16_t));
	if (!f)
		LOG(RPiDenoiseBstm, Error) << "Failed to write " << name.str();
}

void DenoiseBstm::prepare(Metadata *imageMetadata)
{
	std::pair<SharedFD, std::span<uint8_t>> bayer;
	imageMetadata->get("global.bayer_buffer", bayer);

	if (!active_ || !bayer.second.data() || !width_ || !height_)
		return;

	/*
	 * Accept row padding but reject a compressed buffer: bytes-per-row
	 * >= width, not an exact size.
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
		return;
	}

	checkContinuity(imageMetadata);

	const Clock::time_point tStart = Clock::now();
	dmabufSyncStart(bayer.first);
	const double syncStartMs = msSince(tStart);

	uint16_t *bayer16 = reinterpret_cast<uint16_t *>(bayer.second.data());
	const unsigned stridePx = rowBytes / sizeof(uint16_t);
	const bool dump = !dumpDir_.empty() && dumpCount_ < dumpFrames_;

	if (dump) {
		dumpFrame(bayer16, stridePx, "in");
		std::ofstream(dumpDir_ + "/frames.txt", dumpCount_ ? std::ios::app : std::ios::trunc)
			<< dumpCount_ << " " << (firstFrame_ ? 1 : 0) << "\n";
	}

	const Clock::time_point t0 = Clock::now();
	if (isFloat_)
		packInput(bayer16, stridePx, xfTable_.data(), interpreter_->tensor(inX_)->data.f);
	else
		packInput(bayer16, stridePx, xqTable_.data(), interpreter_->tensor(inX_)->data.int8);
	const double packMs = msSince(t0);

	const Clock::time_point t1 = Clock::now();
	if (interpreter_->Invoke() != kTfLiteOk) {
		LOG(RPiDenoiseBstm, Error) << "NPU inference failed";
		dmabufSyncEnd(bayer.first);
		reset();
		return;
	}
	const double invokeMs = msSince(t1);

	const Clock::time_point t2 = Clock::now();
	if (isFloat_)
		blendFloat(bayer16, stridePx);
	else
		blendInt8(bayer16, stridePx);
	const double blendMs = msSince(t2);
	firstFrame_ = false;

	if (dump) {
		dumpFrame(bayer16, stridePx, "out");
		dumpCount_++;
	}

	const Clock::time_point tEnd = Clock::now();
	dmabufSyncEnd(bayer.first);
	const double syncEndMs = msSince(tEnd);

	LOG(RPiDenoiseBstm, Debug)
		<< "sync " << syncStartMs << " + " << syncEndMs << " ms, pack " << packMs
		<< " ms, network " << invokeMs << " ms, blend " << blendMs << " ms, total "
		<< msSince(tStart) << " ms";

	/*
	 * The classic stages' noise thresholds scale with the noise model's
	 * constant and slope. CDN's threshold is deviation * slope + constant,
	 * so scaling the noise model scales the threshold itself.
	 */
	scaleStatus<SdnStatus>(imageMetadata, "sdn.status", sdnFactor_, [](SdnStatus &st, double f) {
		st.noiseConstant *= f;
		st.noiseSlope *= f;
	});
	scaleStatus<TdnStatus>(imageMetadata, "tdn.status", tdnFactor_, [](TdnStatus &st, double f) {
		st.noiseConstant *= f;
		st.noiseSlope *= f;
	});
	scaleStatus<CdnStatus>(imageMetadata, "cdn.status", cdnFactor_, [](CdnStatus &st, double f) {
		st.threshold *= f;
	});
}

/* Register algorithm with the system. */
static Algorithm *create(Controller *controller)
{
	return (Algorithm *)new DenoiseBstm(controller);
}
static RegisterAlgorithm reg(NAME, &create);
