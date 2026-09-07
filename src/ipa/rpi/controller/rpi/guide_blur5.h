/* SPDX-License-Identifier: BSD-2-Clause */
/*
 * guide_blur5.h - fused reflect-pad + 5x5 average blur for the bilateral grid net.
 *
 * Replaces the Padding(2,2,2,2,REFLECT) + Pooling(avg,5x5,stride 1) pair that
 * produces `base` from the guide.
 *
 * WHY. Measured on the Pi, that pair costs 13.1 ms of a 34 ms forward -- the
 * single most expensive thing in the graph, more than seven times the entire
 * convolution trunk. Two reasons, and the fix addresses both:
 *
 *  1. IT DOES NOT THREAD. The tensor is 4-channel and ncnn packs it elempack=4,
 *     so Mat.c == 1 and ncnn`s channel-parallel omp loop in Pooling has exactly
 *     one iteration. Measured 13.1 ms at 1 thread, 14.8 ms at 4 -- adding cores
 *     makes it slower. This layer splits over ROWS, which is the axis that
 *     actually has 548 of something.
 *
 *  2. IT IS NOT SEPARABLE. ncnn evaluates the box non-separably: 25 taps per
 *     output pixel. A 5x5 box is rank 1, so 5 horizontal + 5 vertical adds give
 *     the same answer for 10 taps. The horizontal sums are reused across the
 *     five output rows that need them via a 5-row ring, so the amortised cost
 *     is one horizontal pass plus one vertical pass.
 *
 * It also deletes the Padding layer outright: reflection is done by index
 * arithmetic on the fly, so the 4.4 MB padded intermediate is never written.
 *
 * ARITHMETIC. Sums accumulate in float32 and are scaled by 1/25 once at the
 * end, which is algebraically what avg-pool-over-25 does. Summation ORDER
 * differs from ncnn`s, so results agree to float rounding rather than bitwise;
 * verified end-to-end against the unfused graph.
 *
 * LAYOUT. Same contract as bilateral_assemble_lr.h: 4 channels interleaved at
 * elempack=4, so one pixel is four contiguous values and a horizontal step of
 * one pixel is a step of four elements. Declaring packing/fp16 support keeps
 * ncnn from inserting Cast/Packing layers around us.
 */
#pragma once

#include <algorithm>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#if defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>
#endif

#include <ncnn/layer.h>
#include <ncnn/mat.h>

namespace RPiController {

class GuideBlur5 : public ncnn::Layer
{
public:
	GuideBlur5()
	{
		one_blob_only = true;
		support_inplace = false;
		support_packing = true;
		support_fp16_storage = true;
		support_bf16_storage = false;
	}

	int load_param(const ncnn::ParamDict &pd) override
	{
		radius_ = pd.get(0, 2);
		return 0;
	}

	int forward(const ncnn::Mat &bottom, ncnn::Mat &top,
		    const ncnn::Option &opt) const override
	{
		const int w = bottom.w, h = bottom.h;
		const int ep = bottom.elempack;
		const size_t es = bottom.elemsize / ep;

		/* the flat interleaved walk below assumes 4 channels at elempack 4 */
		if (bottom.c * ep != 4 || ep != 4 || radius_ != 2)
			return -1;
		if (es != 2 && es != 4)
			return -1;
		if (w < 3 || h < 3)
			return -1;

		top.create_like(bottom, opt.blob_allocator);
		if (top.empty())
			return -100;

		const int nthreads = std::max(1, opt.num_threads);
		const size_t lw = size_t(w) * 4;         /* floats per line buffer */
		/*
		 * Five line buffers plus their row tags per thread, allocated ONCE
		 * per forward. bilateral_assemble_lr.h learned this the hard way:
		 * allocating per row cost ~1100 mallocs a frame and wiped the cache
		 * the fusion exists to exploit.
		 */
		std::vector<float> scratch(size_t(nthreads) * 5u * lw);
		std::vector<int> tags(size_t(nthreads) * 5u, -1);

#ifdef _OPENMP
		#pragma omp parallel for num_threads(nthreads) schedule(static)
#endif
		for (int y = 0; y < h; y++) {
#ifdef _OPENMP
			const int tid = omp_get_thread_num();
#else
			const int tid = 0;
#endif
			float *bufs = scratch.data() + size_t(tid) * 5u * lw;
			int *tag = tags.data() + size_t(tid) * 5u;

			const float *rows[5];
			for (int k = -2; k <= 2; k++) {
				const int rr = reflect(y + k, h);
				const int slot = ((rr % 5) + 5) % 5;
				float *buf = bufs + size_t(slot) * lw;
				if (tag[slot] != rr) {
					if (es == 2)
						hsum<__fp16>((const __fp16 *)bottom.channel(0)
							     + size_t(rr) * lw, buf, w);
					else
						hsum<float>((const float *)bottom.channel(0)
							    + size_t(rr) * lw, buf, w);
					tag[slot] = rr;
				}
				rows[k + 2] = buf;
			}

			if (es == 2)
				vsum<__fp16>(rows, (__fp16 *)top.channel(0) + size_t(y) * lw, lw);
			else
				vsum<float>(rows, (float *)top.channel(0) + size_t(y) * lw, lw);
		}

		return 0;
	}

private:
	/* ncnn Padding type 2: padded[i] = src[|i - pad|], mirrored without
	 * repeating the edge sample (padding.cpp:195). */
	static inline int reflect(int i, int n)
	{
		if (i < 0)
			i = -i;
		if (i >= n)
			i = 2 * (n - 1) - i;
		if (i < 0)
			i = 0;
		return i;
	}

	/* Horizontal sum of 5 pixels, per channel lane. No division here -- the
	 * single 1/25 happens once, in vsum. */
	template<typename T>
	static inline void hsum(const T *src, float *dst, int w)
	{
		/* left edge: reflected pixel indices */
		for (int x = 0; x < 2 && x < w; x++)
			sumTap<T>(src, dst + size_t(x) * 4, x, w);
		const int xhi = std::max(2, w - 2);
		/* interior: all five taps are in range, so step contiguously */
		int x = 2;
#if __ARM_NEON
		for (; x < xhi; x++) {
			const T *p = src + (size_t(x) - 2) * 4;
			float32x4_t a = ld4<T>(p);
			a = vaddq_f32(a, ld4<T>(p + 4));
			a = vaddq_f32(a, ld4<T>(p + 8));
			a = vaddq_f32(a, ld4<T>(p + 12));
			a = vaddq_f32(a, ld4<T>(p + 16));
			vst1q_f32(dst + size_t(x) * 4, a);
		}
#endif
		for (; x < xhi; x++)
			sumTap<T>(src, dst + size_t(x) * 4, x, w);
		for (x = xhi; x < w; x++)
			sumTap<T>(src, dst + size_t(x) * 4, x, w);
	}

	template<typename T>
	static inline void sumTap(const T *src, float *dst, int x, int w)
	{
		float acc[4] = { 0, 0, 0, 0 };
		for (int k = -2; k <= 2; k++) {
			const T *p = src + size_t(reflect(x + k, w)) * 4;
			for (int c = 0; c < 4; c++)
				acc[c] += float(p[c]);
		}
		for (int c = 0; c < 4; c++)
			dst[c] = acc[c];
	}

	/* Vertical sum of the five horizontal-sum rows, then the single 1/25. */
	template<typename T>
	static inline void vsum(const float *const rows[5], T *dst, size_t n)
	{
		const float inv = 1.0f / 25.0f;
		size_t i = 0;
#if __ARM_NEON
		const float32x4_t vInv = vdupq_n_f32(inv);
		for (; i + 4 <= n; i += 4) {
			float32x4_t a = vld1q_f32(rows[0] + i);
			a = vaddq_f32(a, vld1q_f32(rows[1] + i));
			a = vaddq_f32(a, vld1q_f32(rows[2] + i));
			a = vaddq_f32(a, vld1q_f32(rows[3] + i));
			a = vaddq_f32(a, vld1q_f32(rows[4] + i));
			st4<T>(dst + i, vmulq_f32(a, vInv));
		}
#endif
		for (; i < n; i++)
			dst[i] = T((rows[0][i] + rows[1][i] + rows[2][i] +
				    rows[3][i] + rows[4][i]) * inv);
	}

	template<typename T> static inline float32x4_t ld4(const T *p);
	template<typename T> static inline void st4(T *p, float32x4_t v);

	int radius_ = 2;
};

#if __ARM_NEON
template<> inline float32x4_t GuideBlur5::ld4<float>(const float *p)
{
	return vld1q_f32(p);
}
template<> inline void GuideBlur5::st4<float>(float *p, float32x4_t v)
{
	vst1q_f32(p, v);
}
#if defined(__ARM_FP16_FORMAT_IEEE)
template<> inline float32x4_t GuideBlur5::ld4<__fp16>(const __fp16 *p)
{
	return vcvt_f32_f16(vld1_f16(p));
}
template<> inline void GuideBlur5::st4<__fp16>(__fp16 *p, float32x4_t v)
{
	vst1_f16(p, vcvt_f16_f32(v));
}
#endif
#endif

::ncnn::Layer *GuideBlur5_layer_creator(void *userdata);
DEFINE_LAYER_CREATOR(GuideBlur5)

} /* namespace RPiController */
