/* SPDX-License-Identifier: BSD-2-Clause */
/*
 * lf_locks.h - fused low-frequency chroma + luma locks and the output clamp.
 *
 * WHAT THIS REPLACES. h2's exported tail ends in ~20 full-resolution layers
 * (2 Padding, 2 Pooling, ~12 BinaryOp, Crop, Clip) implementing
 * nsa/bilateral_grid.py's lf_chroma_lock -> lf_luma_lock -> clamp. ncnn threads
 * elementwise work over CHANNELS, and elempack collapses a 4-channel full-res
 * Mat to c == 1, so every one of those layers runs SINGLE-THREADED over 2.1M
 * elements -- ~15 ms each. Measured: thread count barely moves the graph
 * (1188 ms at threads=1 vs 1124 ms at threads=4) and disabling the packing
 * layout makes it worse (1182 ms). Fusing and threading over ROWS is the fix,
 * exactly as guide_blur5.h and bilateral_assemble_lr.h do for m9.
 *
 * THE MATH, verbatim from nsa/bilateral_grid.py (chroma FIRST, then luma, then
 * clamp -- the luma lock reads the CHROMA-LOCKED output):
 *
 *   go = 0.5*(o1+o2)              gr = 0.5*(r1+r2)
 *   dcr = box_r((o0-go) - (r0-gr))
 *   dcb = box_r((o3-go) - (r3-gr))
 *   o0 -= dcr ; o3 -= dcb
 *
 *   lo = 0.25*(o0+o1+o2+o3)       lr = 0.25*(r0+r1+r2+r3)
 *   d  = box_r(lo - lr)
 *   o[0..3] -= d
 *
 *   out = clamp(o, 0, 1)
 *
 * `ref` MUST be the memory slot, not the live frame -- the docstrings are
 * explicit that one frame's LF chroma noise is not negligible.
 *
 * The box is a plain (2r+1)^2 mean with REPLICATE edges, matching box_blur_r.
 * Two sequential stages, because the luma difference depends on the result of
 * the chroma stage; each stage is independently row-parallel.
 */
#pragma once

#include <algorithm>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <ncnn/layer.h>
#include <ncnn/mat.h>

namespace RPiController {

class LfLocks : public ncnn::Layer
{
public:
	LfLocks()
	{
		one_blob_only = false;
		support_inplace = false;
		support_packing = false;    /* plain fp32 planes; the win is threading */
		support_fp16_storage = false;
		support_bf16_storage = false;
	}

	int load_param(const ncnn::ParamDict &pd) override
	{
		rc_ = pd.get(0, 1);   /* chroma radius */
		rl_ = pd.get(1, 1);   /* luma radius   */
		clip_ = pd.get(2, 1);
		return 0;
	}

	/* (2r+1)^2 box mean of `src` into `dst`, replicate edges, parallel by row. */
	static void box(const float *src, float *dst, int w, int h, int r, [[maybe_unused]] int nthr)
	{
		if (r <= 0) {
			std::copy(src, src + size_t(w) * h, dst);
			return;
		}
		std::vector<float> col(size_t(w) * h);
		const float invW = 1.0f / float(2 * r + 1);
		/* horizontal pass */
#ifdef _OPENMP
		#pragma omp parallel for num_threads(nthr) schedule(static)
#endif
		for (int y = 0; y < h; y++) {
			const float *s = src + size_t(y) * w;
			float *d = col.data() + size_t(y) * w;
			float acc = 0.0f;
			for (int i = -r; i <= r; i++)
				acc += s[std::min(std::max(i, 0), w - 1)];
			for (int x = 0; x < w; x++) {
				d[x] = acc * invW;
				const int add = std::min(x + r + 1, w - 1);
				const int sub = std::min(std::max(x - r, 0), w - 1);
				acc += s[add] - s[sub];
			}
		}
		/* vertical pass */
#ifdef _OPENMP
		#pragma omp parallel for num_threads(nthr) schedule(static)
#endif
		for (int x = 0; x < w; x++) {
			float acc = 0.0f;
			for (int i = -r; i <= r; i++)
				acc += col[size_t(std::min(std::max(i, 0), h - 1)) * w + x];
			for (int y = 0; y < h; y++) {
				dst[size_t(y) * w + x] = acc * invW;
				const int add = std::min(y + r + 1, h - 1);
				const int sub = std::min(std::max(y - r, 0), h - 1);
				acc += col[size_t(add) * w + x] - col[size_t(sub) * w + x];
			}
		}
	}

	int forward(const std::vector<ncnn::Mat> &bots, std::vector<ncnn::Mat> &outs,
		    const ncnn::Option &opt) const override
	{
		if (bots.size() != 2)
			return -1;
		const ncnn::Mat &o = bots[0], &rf = bots[1];
		const int w = o.w, h = o.h;
		if (o.c < 4 || rf.c < 4 || rf.w != w || rf.h != h)
			return -1;

		ncnn::Mat &out = outs[0];
		out.create(w, h, 4, 4u, opt.blob_allocator);
		if (out.empty())
			return -100;

		const size_t n = size_t(w) * h;
		const int nthr = std::max(1, opt.num_threads);
		std::vector<float> t0(n), t1(n), bl(n);

		const float *o0 = o.channel(0), *o1 = o.channel(1),
			    *o2 = o.channel(2), *o3 = o.channel(3);
		const float *r0 = rf.channel(0), *r1 = rf.channel(1),
			    *r2 = rf.channel(2), *r3 = rf.channel(3);
		float *d0 = out.channel(0), *d1 = out.channel(1),
		      *d2 = out.channel(2), *d3 = out.channel(3);

		/* stage 1: chroma lock. t0 = R deviation, t1 = B deviation. */
		if (rc_ > 0) {
#ifdef _OPENMP
			#pragma omp parallel for num_threads(nthr) schedule(static)
#endif
			for (int i = 0; i < int(n); i++) {
				const float go = 0.5f * (o1[i] + o2[i]);
				const float gr = 0.5f * (r1[i] + r2[i]);
				t0[i] = (o0[i] - go) - (r0[i] - gr);
				t1[i] = (o3[i] - go) - (r3[i] - gr);
			}
			box(t0.data(), bl.data(), w, h, rc_, nthr);
#ifdef _OPENMP
			#pragma omp parallel for num_threads(nthr) schedule(static)
#endif
			for (int i = 0; i < int(n); i++)
				d0[i] = o0[i] - bl[i];
			box(t1.data(), bl.data(), w, h, rc_, nthr);
#ifdef _OPENMP
			#pragma omp parallel for num_threads(nthr) schedule(static)
#endif
			for (int i = 0; i < int(n); i++)
				d3[i] = o3[i] - bl[i];
		} else {
			std::copy(o0, o0 + n, d0);
			std::copy(o3, o3 + n, d3);
		}
		std::copy(o1, o1 + n, d1);
		std::copy(o2, o2 + n, d2);

		/* stage 2: luma lock, on the chroma-locked output. */
		if (rl_ > 0) {
#ifdef _OPENMP
			#pragma omp parallel for num_threads(nthr) schedule(static)
#endif
			for (int i = 0; i < int(n); i++) {
				const float lo = 0.25f * (d0[i] + d1[i] + d2[i] + d3[i]);
				const float lr = 0.25f * (r0[i] + r1[i] + r2[i] + r3[i]);
				t0[i] = lo - lr;
			}
			box(t0.data(), bl.data(), w, h, rl_, nthr);
		} else {
			std::fill(bl.begin(), bl.end(), 0.0f);
		}

#ifdef _OPENMP
		#pragma omp parallel for num_threads(nthr) schedule(static)
#endif
		for (int i = 0; i < int(n); i++) {
			const float b = bl[i];
			float v0 = d0[i] - b, v1 = d1[i] - b;
			float v2 = d2[i] - b, v3 = d3[i] - b;
			if (clip_) {
				v0 = v0 < 0.0f ? 0.0f : (v0 > 1.0f ? 1.0f : v0);
				v1 = v1 < 0.0f ? 0.0f : (v1 > 1.0f ? 1.0f : v1);
				v2 = v2 < 0.0f ? 0.0f : (v2 > 1.0f ? 1.0f : v2);
				v3 = v3 < 0.0f ? 0.0f : (v3 > 1.0f ? 1.0f : v3);
			}
			d0[i] = v0; d1[i] = v1; d2[i] = v2; d3[i] = v3;
		}
		return 0;
	}

private:
	int rc_ = 1, rl_ = 1, clip_ = 1;
};

::ncnn::Layer *LfLocks_layer_creator(void *userdata);
DEFINE_LAYER_CREATOR(LfLocks)

} /* namespace RPiController */
