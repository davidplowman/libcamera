/* SPDX-License-Identifier: BSD-2-Clause */
/*
 * input_shrink.h - fused per-octave Wiener shrink on the memory slot.
 *
 * WHAT THIS REPLACES. h2's front end exports nsa/bilateral_grid.py's
 * _apply_input_shrink as ~58 full-resolution layers: 12 box blurs plus ~34
 * BinaryOp, 8 Clip, 4 Reduction. Every one of them is a separate ncnn layer
 * that reads and writes a 968x548x4 tensor through DRAM, and with the packing
 * layout each runs single-threaded because elempack makes Mat.c == 1. Measured:
 * that region is ~250 ms of h2's frame. Everything here is done in one pass
 * over row bands, so the intermediates never reach memory.
 *
 * THE MATH, verbatim from _apply_input_shrink (input_shrink = 4, so
 * _CS_RADII[:5] = 0,1,2,4,8 -> four octaves):
 *
 *   var0 = sigma^2
 *   lo_prev = live ; acc = 0
 *   for j, (r0, r1) in ((0,1), (1,2), (2,4), (4,8)):
 *       lo_next = box_r1(live)                 <- of LIVE, not of lo_prev
 *       band    = lo_prev - lo_next
 *       var_n   = var0 * (1/(2r0+1)^2 - 1/(2r1+1)^2)
 *       var_b   = box_max(2*r1,2)( mean_c(band^2) )
 *       keep    = clamp(1 - lam[j]*var_n/max(var_b,1e-12), 0, 1)
 *       keep    = box_gs(keep)                 <- gate_smooth = 2
 *       acc    += keep * band                  <- ONE keep field for all 4 planes
 *       lo_prev = lo_next
 *   out = acc + lo_prev
 *
 * ONE keep field for all four planes is deliberate and load-bearing: the
 * docstring records that per-plane gates attenuate R, G1, G2 and B differently
 * and thereby CREATE chroma error. Do not "improve" it into a per-plane gate.
 *
 * lam = exp(input_shrink_logit) * _shrink_scale(cond); for h2 shrink_gain_ref is
 * unset so the scale is exactly 1.0 and lam is the four constants below.
 * shrink_keep_min and shrink_chroma_scale are both unset, so neither the kmin
 * floor nor the luma/chroma split applies -- if a future checkpoint sets them
 * this layer must grow to match, so the params are read from the .param rather
 * than hard-coded.
 *
 * bots[1] is var0 = sigma^2, ALREADY SQUARED. The exported graph computes it
 * as (mem-sigma linear plane / 16)^2 and a later layer (_apply_fullres_box's
 * div_59) consumes another copy of it, so the squaring must stay OUTSIDE this
 * layer -- fusing it in would delete a blob the tail still needs.
 */
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <ncnn/layer.h>
#include <ncnn/mat.h>

namespace RPiController {

class InputShrink : public ncnn::Layer
{
public:
	InputShrink()
	{
		one_blob_only = false;
		support_inplace = false;
		support_packing = false;
		support_fp16_storage = false;
		support_bf16_storage = false;
	}

	int load_param(const ncnn::ParamDict &pd) override
	{
		octaves_ = pd.get(0, 4);
		gate_smooth_ = pd.get(1, 2);
		lam_ = pd.get(2, ncnn::Mat());
		return 0;
	}

	static inline int refl(int i, int n)
	{
		if (n == 1)
			return 0;
		while (i < 0 || i >= n)
			i = (i < 0) ? -i : 2 * n - 2 - i;
		return i;
	}

	/* separable O(1) box mean, reflect edges, threaded over rows */
	static void box(const float *src, float *dst, float *mid, float *acc,
			int w, int h, int r, [[maybe_unused]] int nthr)
	{
		if (r <= 0) {
			std::copy(src, src + size_t(w) * h, dst);
			return;
		}
		const float inv = 1.0f / float(2 * r + 1);
#ifdef _OPENMP
		#pragma omp parallel for num_threads(nthr) schedule(static)
#endif
		for (int y = 0; y < h; y++) {
			const float *s = src + size_t(y) * w;
			float *m = mid + size_t(y) * w;
			float a = 0.0f;
			for (int i = -r; i <= r; i++)
				a += s[refl(i, w)];
			for (int x = 0; x < w; x++) {
				m[x] = a;
				a += s[refl(x + r + 1, w)] - s[refl(x - r, w)];
			}
		}
		const float inv2 = inv * inv;
		std::fill(acc, acc + w, 0.0f);
		for (int i = -r; i <= r; i++) {
			const float *m = mid + size_t(refl(i, h)) * w;
			for (int x = 0; x < w; x++)
				acc[x] += m[x];
		}
		for (int y = 0; y < h; y++) {
			float *d = dst + size_t(y) * w;
			for (int x = 0; x < w; x++)
				d[x] = acc[x] * inv2;
			const float *ma = mid + size_t(refl(y + r + 1, h)) * w;
			const float *ms = mid + size_t(refl(y - r, h)) * w;
			for (int x = 0; x < w; x++)
				acc[x] += ma[x] - ms[x];
		}
	}

	int forward(const std::vector<ncnn::Mat> &bots, std::vector<ncnn::Mat> &outs,
		    const ncnn::Option &opt) const override
	{
		if (bots.size() != 2)
			return -1;
		const ncnn::Mat &live = bots[0], &sig = bots[1];   /* sig = var0 */
		const int w = live.w, h = live.h;
		/* one-shot shape dump: a wrong assumption here is a segfault, and the
		 * failure gives no other clue about which bottom was wrong */
		static bool dumped = false;
		if (!dumped) {
			dumped = true;
			fprintf(stderr, "InputShrink: live %dx%dx%d ep=%d cstep=%zu | "
					"sig %dx%dx%d ep=%d cstep=%zu | lam=%d\n",
				live.w, live.h, live.c, live.elempack, live.cstep,
				sig.w, sig.h, sig.c, sig.elempack, sig.cstep,
				lam_.w);
		}
		if (live.c < 4 || sig.c < 1 || sig.w != w || sig.h != h)
			return -1;
		if (live.cstep < size_t(w) * h || sig.cstep < size_t(w) * h)
			return -1;
		const int J = std::min(octaves_, 4);
		static const int R0[4] = { 0, 1, 2, 4 };
		static const int R1[4] = { 1, 2, 4, 8 };

		ncnn::Mat &out = outs[0];
		out.create(w, h, 4, 4u, opt.blob_allocator);
		if (out.empty())
			return -100;

		const size_t n = size_t(w) * h;
		const int nthr = std::max(1, opt.num_threads);
		std::vector<float> mid(n), col(size_t(w), 0.0f);   /* (n) alone is a function decl */
		std::vector<float> loPrev(4 * n), loNext(4 * n), band(4 * n);
		std::vector<float> tmp(n), varb(n), keep(n), keeps(n), accum(4 * n, 0.0f);

		for (int c = 0; c < 4; c++)
			std::copy((const float *)live.channel(c),
				  (const float *)live.channel(c) + n,
				  loPrev.data() + c * n);

		const float *sp = sig.channel(0);
		const float *lam = lam_.empty() ? nullptr : (const float *)lam_;

		for (int j = 0; j < J; j++) {
			for (int c = 0; c < 4; c++)
				box(live.channel(c), loNext.data() + c * n, mid.data(),
				    col.data(), w, h, R1[j], nthr);

			const float n0 = float((2 * R0[j] + 1) * (2 * R0[j] + 1));
			const float n1 = float((2 * R1[j] + 1) * (2 * R1[j] + 1));
			const float vk = 1.0f / n0 - 1.0f / n1;
			const float lj = lam ? lam[j] : 1.0f;

			/* band, and the channel-mean of band^2 in the same sweep */
#ifdef _OPENMP
			#pragma omp parallel for num_threads(nthr) schedule(static)
#endif
			for (int i = 0; i < int(n); i++) {
				float s = 0.0f;
				for (int c = 0; c < 4; c++) {
					const float b = loPrev[c * n + i] - loNext[c * n + i];
					band[c * n + i] = b;
					s += b * b;
				}
				tmp[i] = 0.25f * s;
			}
			box(tmp.data(), varb.data(), mid.data(), col.data(), w, h,
			    std::max(2 * R1[j], 2), nthr);

#ifdef _OPENMP
			#pragma omp parallel for num_threads(nthr) schedule(static)
#endif
			for (int i = 0; i < int(n); i++) {
				const float vn = sp[i] * vk;   /* sp is var0, pre-squared */
				const float vb = varb[i] > 1e-12f ? varb[i] : 1e-12f;
				float k = 1.0f - lj * vn / vb;
				keep[i] = k < 0.0f ? 0.0f : (k > 1.0f ? 1.0f : k);
			}
			const float *kp = keep.data();
			if (gate_smooth_ > 0) {
				box(keep.data(), keeps.data(), mid.data(), col.data(),
				    w, h, gate_smooth_, nthr);
				kp = keeps.data();
			}

#ifdef _OPENMP
			#pragma omp parallel for num_threads(nthr) schedule(static)
#endif
			for (int i = 0; i < int(n); i++) {
				const float k = kp[i];
				for (int c = 0; c < 4; c++)
					accum[c * n + i] += k * band[c * n + i];
			}
			loPrev.swap(loNext);
		}

		for (int c = 0; c < 4; c++) {
			float *d = out.channel(c);
			const float *a = accum.data() + c * n, *l = loPrev.data() + c * n;
#ifdef _OPENMP
			#pragma omp parallel for num_threads(nthr) schedule(static)
#endif
			for (int i = 0; i < int(n); i++)
				d[i] = a[i] + l[i];
		}
		return 0;
	}

private:
	int octaves_ = 4;
	int gate_smooth_ = 2;
	ncnn::Mat lam_;
};

::ncnn::Layer *InputShrink_layer_creator(void *userdata);
DEFINE_LAYER_CREATOR(InputShrink)

} /* namespace RPiController */
