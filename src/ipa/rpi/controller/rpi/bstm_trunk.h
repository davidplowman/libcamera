/* SPDX-License-Identifier: BSD-2-Clause */
/*
 * bstm_trunk.h - run the bilateral-grid predictor on the BCM2712 BSTM NPU.
 *
 * WHAT THIS REPLACES. m9_datafix_lrgb.param is 63 layers and splits as a clean
 * prefix/suffix: layers `convdw_48` .. `conv_46` (53 of them) are the low-res
 * predictor and are all NPU-supported, while `GuideBlur5` / `BilateralAssembleLR`
 * are hand-written full-resolution custom layers with no NPU equivalent. This
 * layer stands in for those 53, so the crop/pool prefix and the full-res tail
 * keep running in ncnn exactly as before and runNet() does not change at all.
 *
 * WHY IT IS WORTH IT. Measured on a Pi 5 at 121x68x9 -> 121x68x16, same weights:
 *
 *     ncnn fp16   1 thread   7.12 ms      4 threads  6.40 ms
 *     BSTM bf16                                      1.78 ms
 *
 * The NPU graph is one partition -- the TFLite delegate reports "Replacing 47
 * out of 47 node(s)" -- so this is a single DMA in, one execute, one DMA out,
 * not an interleaved round trip. bf16 is what makes it worth doing: int8 and
 * bf16 both cost 1 cycle/MAC on BSTM where fp32 costs 9, and the measured gap
 * is 210k cycles fp32 against 50k bf16.
 *
 * PRECISION. The compiled graph is bf16 internally with fp32 in/out. Against
 * the fp32 reference it scores 0 mismatches in 131648 values, max abs diff
 * 0.040 on an output whose |max| is ~17.8.
 *
 * LAYOUT. ncnn is planar CHW; BSTM tensors are NHWC. The transpose is done here
 * on 148 KB in and 263 KB out. Packing and fp16 storage are deliberately NOT
 * declared, so ncnn hands us plain unpacked fp32 and inserts its own
 * Cast/Packing either side rather than us guessing at an elempack layout.
 *
 * PROCESS MODEL. One BSTM context per process, refcounted by BstmRuntime, held
 * for the lifetime of the net. Loading is client mode, so `bstorm-daemon` must
 * be running -- see bstm-bf16-needs-x86 notes in the runbook. If the context
 * cannot be created the layer refuses to load rather than silently producing
 * wrong pixels; ModelDenoise then falls back to the all-ncnn param.
 */
#pragma once

#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <ncnn/layer.h>
#include <ncnn/mat.h>

extern "C" {
#include <bstorm_common.h>
#include <bstorm_init.h>
#include <bstorm_operations.h>
}

namespace RPiController {

/*
 * Passed to the layer as ncnn userdata, because ncnn's ParamDict carries only
 * ints/floats and the .bstm path has to reach the layer somehow.
 */
struct BstmTrunkConfig {
	std::string model_path;
};

/*
 * The BSTM context is a process-wide hardware handle: creating a second one
 * while the first lives fails in BSTM_HW_Open. Only one net loads at a time
 * here, but the refcount keeps that true even if that stops being so.
 */
class BstmRuntime
{
public:
	static BstmRuntime &instance()
	{
		static BstmRuntime rt;
		return rt;
	}

	/* Returns nullptr if the NPU is unavailable; caller must not retry blindly. */
	const struct bstorm_context *acquire()
	{
		std::lock_guard<std::mutex> lock(mutex_);
		if (refs_ == 0) {
			struct bstorm_context_config cfg;
			bstorm_context_config_Default(&cfg);
			cfg.engine.allow.bstm = true;
			context_ = bstorm_context_create(&cfg);
			if (!context_)
				return nullptr;
			struct bstorm_context_status status;
			bstorm_result rc = bstorm_context_get_status(context_, &status);
			if (BSTORM_IS_ERROR(rc) || !status.engine.bstm.enabled) {
				bstorm_context_destroy(context_);
				context_ = nullptr;
				return nullptr;
			}
		}
		refs_++;
		return context_;
	}

	void release()
	{
		std::lock_guard<std::mutex> lock(mutex_);
		if (refs_ > 0 && --refs_ == 0 && context_) {
			bstorm_context_destroy(context_);
			context_ = nullptr;
		}
	}

private:
	BstmRuntime() = default;
	~BstmRuntime() = default;
	BstmRuntime(const BstmRuntime &) = delete;
	BstmRuntime &operator=(const BstmRuntime &) = delete;

	std::mutex mutex_;
	struct bstorm_context *context_ = nullptr;
	unsigned refs_ = 0;
};

class BstmTrunk : public ncnn::Layer
{
public:
	BstmTrunk()
	{
		one_blob_only = true;
		support_inplace = false;
		/*
		 * Take plain fp32 CHW. The NPU wants fp32 NHWC at its boundary
		 * anyway, so accepting ncnn's packed fp16 here would mean
		 * unpacking AND converting -- let ncnn's own Cast/Packing do it.
		 */
		support_packing = false;
		support_fp16_storage = false;
		support_bf16_storage = false;
	}

	~BstmTrunk() override = default;

	int load_param(const ncnn::ParamDict &pd) override
	{
		out_c_ = pd.get(0, 16);
		in_c_ = pd.get(1, 9);
		return 0;
	}

	int create_pipeline(const ncnn::Option & /*opt*/) override
	{
		const BstmTrunkConfig *cfg = static_cast<const BstmTrunkConfig *>(userdata);
		if (!cfg || cfg->model_path.empty())
			return -1;

		context_ = BstmRuntime::instance().acquire();
		if (!context_)
			return -1;
		holds_context_ = true;

		struct bstorm_bstm_precompiled_model_config mcfg;
		bstorm_bstm_precompiled_model_config_default(&mcfg);
		/*
		 * Direct access maps the payload buffers into this process, so
		 * the per-frame cost is the CHW<->HWC transpose only, with no
		 * extra copy through write_input/read_output.
		 */
		mcfg.payload_access = bstorm_bstm_payload_access_direct;
		model_ = bstorm_bstm_precompiled_model_load(context_, cfg->model_path.c_str(), &mcfg);
		if (!model_)
			return -1;

		bstorm_result rc = bstorm_bstm_precompiled_get_model_info(model_, &info_);
		if (BSTORM_IS_ERROR(rc))
			return -1;
		if (info_.input.count != 1 || info_.output.count != 1)
			return -1;
		direct_ = (mcfg.payload_access == bstorm_bstm_payload_access_direct);
		if (!direct_) {
			in_staging_.resize(info_.input.blobs[0].size / sizeof(float));
			out_staging_.resize(info_.output.blobs[0].size / sizeof(float));
		}
		return 0;
	}

	int destroy_pipeline(const ncnn::Option & /*opt*/) override
	{
		if (model_) {
			bstorm_bstm_precompiled_model_release(model_);
			model_ = nullptr;
		}
		if (holds_context_) {
			BstmRuntime::instance().release();
			holds_context_ = false;
			context_ = nullptr;
		}
		return 0;
	}

	int forward(const ncnn::Mat &bottom, ncnn::Mat &top,
		    const ncnn::Option &opt) const override
	{
		if (!model_)
			return -1;

		const int w = bottom.w, h = bottom.h, c = bottom.c;
		if (c != in_c_)
			return -1;

		const size_t in_vals = size_t(w) * h * c;
		const size_t out_vals = size_t(w) * h * out_c_;
		if (info_.input.blobs[0].size != in_vals * sizeof(float) ||
		    info_.output.blobs[0].size != out_vals * sizeof(float))
			return -1;

		float *in_hwc = direct_ ? static_cast<float *>(info_.input.blobs[0].payload)
					: const_cast<float *>(in_staging_.data());

		/*
		 * CHW -> HWC. Walk each ncnn channel with its own pointer:
		 * Mat::channel() honours cstep, which is padded for alignment
		 * and is NOT w*h, so a flat index into Mat::data would drift.
		 */
		for (int ch = 0; ch < c; ch++) {
			const float *src = bottom.channel(ch);
			float *dst = in_hwc + ch;
			for (int i = 0; i < w * h; i++)
				dst[size_t(i) * c] = src[i];
		}
		if (!direct_) {
			bstorm_result rc = bstorm_bstm_precompiled_write_input(
				model_, 0, in_staging_.data(), info_.input.blobs[0].size);
			if (BSTORM_IS_ERROR(rc))
				return -1;
		}

		bstorm_result rc = bstorm_bstm_precompiled_exec(context_, model_, nullptr);
		if (BSTORM_IS_ERROR(rc))
			return -1;

		top.create(w, h, out_c_, size_t(4u), opt.blob_allocator);
		if (top.empty())
			return -100;

		const float *out_hwc;
		if (direct_) {
			out_hwc = static_cast<const float *>(info_.output.blobs[0].payload);
		} else {
			rc = bstorm_bstm_precompiled_read_output(
				model_, 0, out_staging_.data(), info_.output.blobs[0].size);
			if (BSTORM_IS_ERROR(rc))
				return -1;
			out_hwc = out_staging_.data();
		}

		/* HWC -> CHW, same cstep caveat. */
		for (int ch = 0; ch < out_c_; ch++) {
			float *dst = top.channel(ch);
			const float *src = out_hwc + ch;
			for (int i = 0; i < w * h; i++)
				dst[i] = src[size_t(i) * out_c_];
		}
		return 0;
	}

private:
	int out_c_ = 16;
	int in_c_ = 9;
	const struct bstorm_context *context_ = nullptr;
	struct bstorm_bstm_precompiled_model *model_ = nullptr;
	struct bstorm_bstm_precompiled_model_info info_ {};
	bool direct_ = false;
	bool holds_context_ = false;
	mutable std::vector<float> in_staging_;
	mutable std::vector<float> out_staging_;
};

/*
 * Own creator rather than DEFINE_LAYER_CREATOR, because that macro drops the
 * userdata pointer and userdata is how the .bstm path reaches the layer.
 */
static inline ::ncnn::Layer *BstmTrunk_layer_creator(void *userdata)
{
	BstmTrunk *layer = new BstmTrunk;
	layer->userdata = userdata;
	return layer;
}

static inline void BstmTrunk_layer_destroyer(::ncnn::Layer *layer, void * /*userdata*/)
{
	delete layer;
}

} /* namespace RPiController */
