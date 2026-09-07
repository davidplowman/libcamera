/* SPDX-License-Identifier: BSD-2-Clause */
/*
 * box_blur_r.h - O(1)-per-pixel box blur, any radius, threaded over rows.
 *
 * WHAT THIS REPLACES. h2's graph contains TWELVE full-resolution box blurs
 * exported as Padding + Pooling pairs, with kernels 3,5,5,5,9,5,9,17,5,17,33,5
 * -- the multi-scale pyramid. ncnn's Pooling evaluates the window naively, so
 * the 33x33 costs 1089 multiply-adds per pixel over 968x548, and elempack
 * collapses the 4-channel Mat to c == 1 so it runs SINGLE-THREADED. Measured:
 * the subgraph containing these is 982 ms of h2's 1117 ms net.
 *
 * A box mean is separable AND incremental: a horizontal running sum then a
 * vertical one costs ~4 adds per pixel REGARDLESS of radius. For the 33x33 that
 * is ~270x less arithmetic, and the row loops thread properly because they are
 * over rows, not channels.
 *
 * EDGES ARE REFLECT, matching the Padding layers this replaces (ncnn pad type
 * 2) and torch's box_blur_r as the exporter builds it. Reflect here is the
 * no-repeat form: index -1 maps to 1, index n maps to n-2.
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

class BoxBlurR : public ncnn::Layer
{
public:
	BoxBlurR()
	{
		one_blob_only = true;
		support_inplace = false;
		support_packing = false;   /* plain fp32 planes; the win is threading */
		support_fp16_storage = false;
		support_bf16_storage = false;
	}

	int load_param(const ncnn::ParamDict &pd) override
	{
		radius_ = pd.get(0, 1);
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

	int forward(const ncnn::Mat &bottom, ncnn::Mat &top,
		    const ncnn::Option &opt) const override
	{
		const int w = bottom.w, h = bottom.h, c = bottom.c;
		const int r = radius_;
		if (w < 1 || h < 1)
			return -1;
		top.create(w, h, c, 4u, opt.blob_allocator);
		if (top.empty())
			return -100;
		if (r <= 0) {
			for (int q = 0; q < c; q++) {
				const float *s0 = bottom.channel(q);
				std::copy(s0, s0 + size_t(w) * h, (float *)top.channel(q));
			}
			return 0;
		}

		const float inv = 1.0f / float(2 * r + 1);
		[[maybe_unused]] const int nthr = std::max(1, opt.num_threads);

		/*
		 * The vertical pass keeps a running sum PER COLUMN in a row-length
		 * accumulator and walks y outermost, so every access is sequential.
		 * Column-outermost (stride-w loads) misses on every element. The
		 * scratch buffers are hoisted out of the channel loop; allocating
		 * 2 MB per channel per call was itself a measurable cost.
		 */
		std::vector<float> mid(size_t(w) * h);
		std::vector<float> acc(size_t(w), 0.0f);

		for (int q = 0; q < c; q++) {
			const float *src = bottom.channel(q);
			float *dst = top.channel(q);

#ifdef _OPENMP
			#pragma omp parallel for num_threads(nthr) schedule(static)
#endif
			for (int y = 0; y < h; y++) {
				const float *s = src + size_t(y) * w;
				float *m = mid.data() + size_t(y) * w;
				float a = 0.0f;
				for (int i = -r; i <= r; i++)
					a += s[refl(i, w)];
				for (int x = 0; x < w; x++) {
					m[x] = a;
					a += s[refl(x + r + 1, w)] - s[refl(x - r, w)];
				}
			}

			const float inv2 = inv * inv;
			std::fill(acc.begin(), acc.end(), 0.0f);
			for (int i = -r; i <= r; i++) {
				const float *m = mid.data() + size_t(refl(i, h)) * w;
				for (int x = 0; x < w; x++)
					acc[x] += m[x];
			}
			for (int y = 0; y < h; y++) {
				float *d = dst + size_t(y) * w;
				for (int x = 0; x < w; x++)
					d[x] = acc[x] * inv2;
				const float *ma = mid.data() + size_t(refl(y + r + 1, h)) * w;
				const float *ms = mid.data() + size_t(refl(y - r, h)) * w;
				for (int x = 0; x < w; x++)
					acc[x] += ma[x] - ms[x];
			}
		}
		return 0;
	}

private:
	int radius_ = 1;
};

::ncnn::Layer *BoxBlurR_layer_creator(void *userdata);
DEFINE_LAYER_CREATOR(BoxBlurR)

} /* namespace RPiController */
