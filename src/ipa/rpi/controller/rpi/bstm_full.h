/* SPDX-License-Identifier: BSD-2-Clause */
/*
 * bstm_full.h - run the ENTIRE denoiser on the BSTM NPU, one tile at a time.
 *
 * This is the successor to bstm_trunk.h. That offloaded only the predictor,
 * which is ~3% of the frame, so it saved 3.6 ms of ~206 ms. Here the whole
 * graph -- predictor, guided-filter tail and all -- is one precompiled bf16
 * BSTM program, and ncnn is out of the inference path entirely.
 *
 * WHAT MADE THE WHOLE MODEL EXPORTABLE (see export_bstm_full.py):
 *  - box_blur_r's cumsum integral images -> reflect-pad + AVERAGE_POOL_2D.
 *  - F.pad(mode="reflect") -> SLICE + CONCATENATION.
 *  - The single 8x bilinear upsample -> three chained 2x steps, because
 *    bstorm_compiler_get_supported.c:385 supports ResizeBilinear for EXACTLY
 *    2x: `input.W * 2 == output.W && input.H * 2 == output.H`. That one rule is
 *    why an 8x resize was always rejected, and why tiles must be multiples of 8.
 * Result: 93 of 93 nodes delegated, ONE partition.
 *
 * WHY TILES, AND WHY SMALL ONES. A full-resolution 4-channel bf16 tensor is
 * ~1 MB against a 2 MB on-chip scratchpad, so a whole-frame inference cannot
 * hold even one op's input and output and spills every layer to DRAM. Measured
 * on this Pi at 968x548-equivalent coverage:
 *
 *     tile       DMA/px   ns/px
 *     484x274     876 B     170
 *     242x137     490 B      95
 *     128x64       74 B     127   <- with the true bilinear chain
 *
 * Tiling is NOT a resolution reduction: every pixel is still processed, just in
 * pieces. ModelDenoise::switchMode lays out the 2D grid.
 *
 * LAYOUT. ncnn/feed_chw_ is planar CHW; BSTM is NHWC. The transpose is done
 * here. Payload access is direct, so there is no extra copy beyond that.
 */
#pragma once

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <ncnn/mat.h>

#include "bstm_trunk.h"   /* BstmRuntime: one refcounted context per process */

namespace RPiController {

class BstmFullModel
{
public:
	BstmFullModel() = default;
	~BstmFullModel() { release(); }

	BstmFullModel(const BstmFullModel &) = delete;
	BstmFullModel &operator=(const BstmFullModel &) = delete;

	/*
	 * in_c/w/h must match the compiled program exactly -- the .bstm is built
	 * for one tile size. A mismatch is rejected here rather than allowed to
	 * read past a payload buffer.
	 */
	bool load(const std::string &path, unsigned in_c, unsigned w, unsigned h,
		  unsigned out_c)
	{
		release();
		context_ = BstmRuntime::instance().acquire();
		if (!context_)
			return false;
		holds_ = true;

		struct bstorm_bstm_precompiled_model_config cfg;
		bstorm_bstm_precompiled_model_config_default(&cfg);
		cfg.payload_access = bstorm_bstm_payload_access_direct;
		model_ = bstorm_bstm_precompiled_model_load(context_, path.c_str(), &cfg);
		if (!model_)
			return false;
		if (BSTORM_IS_ERROR(bstorm_bstm_precompiled_get_model_info(model_, &info_)))
			return false;
		if (info_.input.count < 1 || info_.input.count > 2 || info_.output.count != 1)
			return false;

		const size_t want_out = size_t(w) * h * out_c * sizeof(float);
		if (info_.output.blobs[0].size != want_out)
			return false;
		if (info_.input.count == 1) {
			if (info_.input.blobs[0].size != size_t(w) * h * in_c * sizeof(float))
				return false;
		} else {
			/* Split model: [image slot, block means]. */
			if (info_.input.blobs[0].size != want_out ||
			    info_.input.blobs[1].size != size_t(w) * h * in_img_c_ * sizeof(float))
				return false;
		}

		in_c_ = in_c; out_c_ = out_c; w_ = w; h_ = h;
		return true;
	}

	/*
	 * Space-to-depth variant: the compiled program runs on a gw x gh grid with
	 * the r x r block folded into channels, so in_c*r*r channels go in and
	 * out_c*r*r come out. Nothing full-resolution exists inside the graph --
	 * which is the point, since 4-channel full-res work sustains 0.14 GMAC/s
	 * on this chip against 373 GMAC/s wide-and-deep.
	 */
	bool loadS2D(const std::string &path, unsigned in_c, unsigned full_w,
		     unsigned full_h, unsigned r, unsigned out_c)
	{
		block_ = r;
		full_w_ = full_w;
		full_h_ = full_h;
		gw_ = full_w / r;
		gh_ = full_h / r;
		in_img_c_ = in_c;
		out_img_c_ = out_c;
		return load(path, in_c * r * r, gw_, gh_, out_c * r * r);
	}

	/* For the merged path: the caller fills this itself, block-ordered. */
	float *inputPayload() { return static_cast<float *>(info_.input.blobs[0].payload); }

	/*
	 * Split-input fold: writes ONLY the image slot into payload 0 (oc*sub
	 * channels) and the 8x8 block means of ALL in_img_c_ planes into payload 1
	 * (in_img_c_ channels at the s2d grid).
	 *
	 * The tail consumes only the image slot; the other slots and the gain
	 * plane exist purely to be block-averaged for the predictor. Sending the
	 * whole feed meant writing 576 channels (19 MB) when 256 (8.4 MB) suffice.
	 * The block means come almost free: this pass already reads all nine
	 * planes, so it accumulates them on the way past.
	 */
	template <typename T>
	bool foldSplit(const T *chw, float *img, float *low)
	{
		if (!model_ || block_ == 0 || !img || !low)
			return false;
		const unsigned r = block_, sub = r * r;
		const size_t plane = size_t(full_w_) * full_h_;
		const unsigned CIMG = out_img_c_ * sub;   /* image slot only */
		const float inv = 1.0f / float(sub);

		/*
		 * PLANE-OUTER, NOT BLOCK-OUTER. The obvious loop order -- for each
		 * block, gather all in_img_c_ channels -- makes 8 tiny reads per
		 * channel per block from nine separate ~2 MB planes, i.e. 72
		 * scattered quarter-cache-line bursts per block. Measured 13.4 ms
		 * to move ~18 MB (~1.4 GB/s), against ~5.5 ms for the merged
		 * single-input fold doing MORE bytes.
		 *
		 * Streaming one plane at a time makes the source reads fully
		 * sequential along each row: for fixed (c, by, dy) the address
		 * advances with bx. Destination writes are strided by CIMG but go
		 * out in contiguous r-float runs, which is the cheap direction.
		 */
		std::memset(low, 0, size_t(gw_) * gh_ * in_img_c_ * sizeof(float));

		for (unsigned c = 0; c < in_img_c_; ++c) {
			const T *src = chw + size_t(c) * plane;
			const bool keep = (c < out_img_c_);
			/*
			 * Parallel over BLOCK ROWS. Race-free by construction:
			 * a given `by` owns its own img and low rows, and the
			 * only accumulation (into low) is indexed by (by,bx,c).
			 * Threading the merged fold was measured slower before,
			 * but that one read scattered slot memory; this one
			 * streams a single plane per pass.
			 */
#ifdef _OPENMP
			#pragma omp parallel for num_threads(threads_) schedule(static)
#endif
			for (int byi = 0; byi < int(gh_); ++byi) {
				const unsigned by = unsigned(byi);
				float *lrow = low + size_t(by) * gw_ * in_img_c_ + c;
				for (unsigned dy = 0; dy < r; ++dy) {
					const T *s = src + (size_t(by) * r + dy) * full_w_;
					if (keep) {
						float *d = img + size_t(by) * gw_ * CIMG
							   + c * sub + dy * r;
						for (unsigned bx = 0; bx < gw_; ++bx) {
							const T *sb = s + size_t(bx) * r;
							float acc = 0.0f;
							for (unsigned dx = 0; dx < r; ++dx) {
								const float v = float(sb[dx]);
								d[dx] = v;
								acc += v;
							}
							lrow[size_t(bx) * in_img_c_] += acc;
							d += CIMG;
						}
					} else {
						for (unsigned bx = 0; bx < gw_; ++bx) {
							const T *sb = s + size_t(bx) * r;
							float acc = 0.0f;
							for (unsigned dx = 0; dx < r; ++dx)
								acc += float(sb[dx]);
							lrow[size_t(bx) * in_img_c_] += acc;
						}
					}
				}
			}
		}
		/* the block means were accumulated as sums; scale once at the end */
		{
			const size_t n = size_t(gw_) * gh_ * in_img_c_;
			for (size_t i = 0; i < n; ++i)
				low[i] *= inv;
		}
		return true;
	}

	float *inputPayload(unsigned i)
	{
		return (i < info_.input.count)
			       ? static_cast<float *>(info_.input.blobs[i].payload)
			       : nullptr;
	}
	unsigned inputCount() const { return info_.input.count; }

	/* Exec + unfold only; input is assumed already written by the caller. */
	bool execUnfold(ncnn::Mat &out)
	{
		if (!model_ || block_ == 0)
			return false;
		/*
		 * Split the timer: `net` bundles the NPU execute with a CPU
		 * unfold (block order -> raster). Overlapping CPU with NPU is
		 * only worth doing if we know which half is which -- the unfold
		 * is CPU and therefore CANNOT be hidden behind the NPU.
		 */
		auto e0 = std::chrono::steady_clock::now();
		if (BSTORM_IS_ERROR(bstorm_bstm_precompiled_exec(context_, model_, nullptr)))
			return false;
		auto e1 = std::chrono::steady_clock::now();
		const bool ok = unfold(out);
		auto e2 = std::chrono::steady_clock::now();
		ms_exec_ = std::chrono::duration<double, std::milli>(e1 - e0).count();
		ms_unfold_ = std::chrono::duration<double, std::milli>(e2 - e1).count();
		return ok;
	}

	double lastExecMs() const { return ms_exec_; }
	double lastUnfoldMs() const { return ms_unfold_; }

	bool valid() const { return model_ != nullptr; }
	unsigned block() const { return block_; }
	/* Rows the s2d grid cannot reach: full_h - gh*r. Left untouched. */
	unsigned uncoveredRows() const { return full_h_ - gh_ * block_; }
	unsigned tileW() const { return w_; }
	unsigned tileH() const { return h_; }

	/* chw: in_c planes of w*h, planar. out: ncnn Mat w x h x out_c, fp32. */
	template <typename T>
	bool run(const T *chw, ncnn::Mat &out)
	{
		if (!model_)
			return false;
		float *in = static_cast<float *>(info_.input.blobs[0].payload);
		const size_t plane = size_t(w_) * h_;
		for (unsigned c = 0; c < in_c_; c++) {
			const T *src = chw + size_t(c) * plane;
			float *dst = in + c;
			for (size_t i = 0; i < plane; i++)
				dst[i * in_c_] = float(src[i]);
		}
		if (BSTORM_IS_ERROR(bstorm_bstm_precompiled_exec(context_, model_, nullptr)))
			return false;

		out.create(int(w_), int(h_), int(out_c_), size_t(4u));
		if (out.empty())
			return false;
		const float *o = static_cast<const float *>(info_.output.blobs[0].payload);
		for (unsigned c = 0; c < out_c_; c++) {
			float *dst = out.channel(c);
			const float *src = o + c;
			for (size_t i = 0; i < plane; i++)
				dst[i] = src[i * out_c_];
		}
		return true;
	}

	/*
	 * chw: in_img_c_ planes of full_w_ x full_h_, planar, raster order.
	 * The NPU wants them in BLOCK order, so this is the same copy the raster
	 * path does with different addressing -- no extra passes over memory.
	 */
	template <typename T>
	bool runS2D(const T *chw, ncnn::Mat &out)
	{
		if (!model_ || block_ == 0)
			return false;
		const unsigned r = block_, sub = r * r;
		const size_t plane = size_t(full_w_) * full_h_;
		const unsigned CIN = in_img_c_ * sub;
		float *in = static_cast<float *>(info_.input.blobs[0].payload);

		/*
		 * Raster -> block order. Parallel over block ROWS: each thread owns a
		 * disjoint band of the destination, so no synchronisation is needed.
		 * Worth doing -- single-threaded this pass alone cost ~11 ms, which is
		 * most of what the NPU saves.
		 */
		/*
		 * Raster -> block order, DESTINATION-SEQUENTIAL.
		 *
		 * The obvious nest (channel, then block) writes 8-float runs that land
		 * CIN floats apart, so every run touches a fresh cache line and the CPU
		 * does read-modify-write on all of them: measured 8.40 ms even across 4
		 * threads, against 2.58 ms for the reverse direction which writes
		 * contiguously. Walking blocks on the OUTSIDE makes the destination one
		 * long sequential fill -- a block's whole CIN floats are adjacent -- and
		 * moves the scattering to the READ side, which is much cheaper.
		 */
		/*
		 * NOT parallelised, deliberately. Measured in the live IPA: the fold is
		 * 6.22 ms single-threaded and 7.84 ms across 4 OpenMP threads, and the
		 * unfold 2.06 vs 2.61. It moves ~28 MB with almost no arithmetic, so it
		 * is memory-bound; spawn and contention cost more than the extra cores
		 * return. Do not "optimise" this by adding a pragma back.
		 */
		for (int by = 0; by < int(gh_); by++) {
			for (unsigned bx = 0; bx < gw_; bx++) {
				float *d = in + (size_t(by) * gw_ + bx) * CIN;
				for (unsigned c = 0; c < in_img_c_; c++) {
					const T *src = chw + size_t(c) * plane;
					for (unsigned dy = 0; dy < r; dy++) {
						const T *s = src + (size_t(by) * r + dy) * full_w_
							    + bx * r;
						for (unsigned dx = 0; dx < r; dx++)
							*d++ = float(s[dx]);
					}
				}
			}
		}

		if (BSTORM_IS_ERROR(bstorm_bstm_precompiled_exec(context_, model_, nullptr)))
			return false;

		return unfoldT<T>(out, chw);
	}

	bool unfold(ncnn::Mat &out) { return unfoldT<float>(out, nullptr); }

	template <typename T>
	bool unfoldT(ncnn::Mat &out, const T *chw)
	{
		const unsigned r = block_, sub = r * r;
		const size_t plane = size_t(full_w_) * full_h_;
		const unsigned COUT = out_img_c_ * sub;
		out.create(int(full_w_), int(full_h_), int(out_img_c_), size_t(4u));
		if (out.empty())
			return false;
		const float *o = static_cast<const float *>(info_.output.blobs[0].payload);
		for (unsigned c = 0; c < out_img_c_; c++) {
			float *dst = out.channel(c);
			for (int by = 0; by < int(gh_); by++) {
				for (unsigned dy = 0; dy < r; dy++) {
					float *row = dst + (size_t(by) * r + dy) * full_w_;
					for (unsigned bx = 0; bx < gw_; bx++) {
						const float *s = o + (size_t(by) * gw_ + bx) * COUT
								 + c * sub + dy * r;
						float *d = row + bx * r;
						for (unsigned dx = 0; dx < r; dx++)
							d[dx] = s[dx];
					}
				}
			}
			/*
			 * 548 = 8*68 + 4, so the bottom four packed rows have no block.
			 * Copy the input through rather than leaving the Mat
			 * uninitialised -- undenoised looks right, garbage does not.
			 */
			if (chw) {
				const T *src = chw + size_t(c) * plane;
				for (unsigned y = gh_ * r; y < full_h_; y++) {
					const T *s = src + size_t(y) * full_w_;
					float *d = dst + size_t(y) * full_w_;
					for (unsigned x = 0; x < full_w_; x++)
						d[x] = float(s[x]);
				}
			}
		}
		return true;
	}

private:
	void release()
	{
		if (model_) {
			bstorm_bstm_precompiled_model_release(model_);
			model_ = nullptr;
		}
		if (holds_) {
			BstmRuntime::instance().release();
			holds_ = false;
			context_ = nullptr;
		}
	}

	const struct bstorm_context *context_ = nullptr;
	struct bstorm_bstm_precompiled_model *model_ = nullptr;
	struct bstorm_bstm_precompiled_model_info info_ {};
	bool holds_ = false;
	unsigned in_c_ = 0, out_c_ = 0, w_ = 0, h_ = 0;
	unsigned block_ = 0, gw_ = 0, gh_ = 0, full_w_ = 0, full_h_ = 0;
	unsigned in_img_c_ = 0, out_img_c_ = 0;
	mutable unsigned calls_ = 0;
	double ms_exec_ = 0.0, ms_unfold_ = 0.0;
	int threads_ = 4;

public:
	void setThreads(int n) { threads_ = n > 0 ? n : 1; }

private:
};

} /* namespace RPiController */
