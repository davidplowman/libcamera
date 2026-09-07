/* SPDX-License-Identifier: BSD-2-Clause */
/*
 * bilateral_assemble_lr.h - guided-filter assemble with the coefficient
 * upsample folded in.
 *
 * Replaces Slice + 4x Interp + 3x Sigmoid + the eleven elementwise layers
 * (19 layers) with a single pass computing
 *
 *   a,b,m,w = bilinear(coeff16, x, y)        sigmoid on a, m, w (not b)
 *   out     = clip((1-w)*((a+aMin)*base + b + m*(detail+det)) + w*guide, lo, hi)
 *
 * WHY THE UPSAMPLE IS THE POINT. The coefficients only exist at 121x68 -- four
 * groups of four channels, ~264 KB. The four Interp layers inflated them to
 * 968x548 and wrote them to DRAM so the next layer could read them back: 34 MB
 * per frame of traffic to carry 264 KB of information, on a net whose total is
 * ~207 MB. The three full-resolution Sigmoids cost another 25 MB. Both vanish
 * if the interpolation happens into an L1 line buffer and is consumed
 * immediately, which is what this does.
 *
 * SEMANTICS ARE PRESERVED. sigmoid is applied AFTER interpolation, exactly as
 * the graph did. Doing it at low resolution first would be ~50x cheaper and is
 * tempting, but sigmoid(lerp(x)) != lerp(sigmoid(x)) and that is a silent
 * change to the model, so it is not done here.
 *
 * LAYOUT. The full-resolution tensors are 4-channel, so ncnn packs them
 * elempack=4: the four channels of a pixel are contiguous. The line buffers are
 * built in that same interleaved order, which makes the assemble a flat
 * elementwise walk (see bilateral_assemble.h) rather than a strided one.
 */
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdio>
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

class BilateralAssembleLR : public ncnn::Layer
{
public:
	BilateralAssembleLR()
	{
		one_blob_only = false;
		support_inplace = false;
		support_packing = true;
		support_fp16_storage = true;
		support_bf16_storage = false;
	}

	int load_param(const ncnn::ParamDict &pd) override
	{
		aMin_ = pd.get(0, 0.1f);
		loQ_ = pd.get(1, 0.0f);
		hiQ_ = pd.get(2, 1.0f);
		return 0;
	}

	/* bottoms: 0 coeff16 (low res), 1 det, 2 base, 3 detail, 4 guide */
	int forward(const std::vector<ncnn::Mat> &bots, std::vector<ncnn::Mat> &outs,
		    const ncnn::Option &opt) const override
	{
		/*
		 * Two accepted forms:
		 *   5 bottoms  coeff16, det, base, detail, guide   (as exported)
		 *   3 bottoms  coeff16, base, guide                (short form)
		 *
		 * The short form exists because both tensors it drops are free.
		 * `detail` is guide - base, and this layer already holds guide and
		 * base in registers, so recomputing it costs one subtract and saves
		 * materialising, writing and re-reading a full-resolution tensor
		 * (4.2 MB each way) plus the BinaryOp that produced it. `det` is the
		 * detail-branch output, and when the net has no detail branch the
		 * fuser had to synthesise a full-resolution ZERO tensor purely
		 * because this layer refused a shorter call -- 3 more full-res
		 * traversals to add nothing.
		 */
		const bool shortForm = (bots.size() == 3);
		if (bots.size() != 5 && !shortForm)
			return -1;

		const ncnn::Mat &lr = bots[0];
		const ncnn::Mat &base = bots[shortForm ? 1 : 2];
		const ncnn::Mat &g = bots[shortForm ? 2 : 4];
		const ncnn::Mat *det = shortForm ? nullptr : &bots[1];
		const ncnn::Mat *detail = shortForm ? nullptr : &bots[3];

		const int ow = g.w, oh = g.h;
		const size_t es = g.elemsize / g.elempack;

		/* the flat walk below assumes 4 interleaved channels per pixel */
		for (size_t k = 1; k < bots.size(); k++) {
			const ncnn::Mat &t = bots[k];
			if (t.w != ow || t.h != oh || t.c * t.elempack != 4 ||
			    t.elempack != 4 || t.elemsize / t.elempack != es)
				return -1;
		}
		const int lw = lr.w, lh = lr.h, lep = lr.elempack;
		if (lr.c * lep != 16 || lr.elemsize / lep != es)
			return -1;

		ncnn::Mat &out = outs[0];
		out.create_like(g, opt.blob_allocator);
		if (out.empty())
			return -100;

#if defined(__ARM_NEON) && defined(__ARM_FP16_FORMAT_IEEE)
		/* packed fp16 coefficients: four channels of a group are
		 * contiguous lanes, so the whole layer vectorises. */
		if (es == 2 && lep >= 4 && (lep % 4) == 0)
			return forwardFast(lr, det, base, detail, g, out, lep, opt);
#endif

		/*
		 * Fallback path only (fp32, or a packing this layer cannot walk).
		 * Rather than thread the null cases through the scalar kernels,
		 * materialise what the 5-bottom form would have been.
		 */
		ncnn::Mat detailTmp, detTmp;
		if (shortForm) {
			detailTmp.create_like(g, opt.workspace_allocator);
			detTmp.create_like(g, opt.workspace_allocator);
			if (detailTmp.empty() || detTmp.empty())
				return -100;
			const size_t n = size_t(ow) * oh * 4;
			if (es == 4) {
				const float *pg = g.channel(0), *pb = base.channel(0);
				float *pd = detailTmp.channel(0);
				for (size_t i = 0; i < n; i++)
					pd[i] = pg[i] - pb[i];
				float *pz = detTmp.channel(0);
				for (size_t i = 0; i < n; i++)
					pz[i] = 0.0f;
			} else {
				const __fp16 *pg = (const __fp16 *)g.channel(0);
				const __fp16 *pb = (const __fp16 *)base.channel(0);
				__fp16 *pd = (__fp16 *)detailTmp.channel(0);
				for (size_t i = 0; i < n; i++)
					pd[i] = __fp16(float(pg[i]) - float(pb[i]));
				__fp16 *pz = (__fp16 *)detTmp.channel(0);
				for (size_t i = 0; i < n; i++)
					pz[i] = __fp16(0.0f);
			}
			detail = &detailTmp;
			det = &detTmp;
		}

		/*
		 * x0/wx depend only on the geometry, so hoist them out of the row
		 * loop. align_corners=1, matching Interp's 6=1.
		 */
		const float sx = (ow > 1) ? float(lw - 1) / float(ow - 1) : 0.0f;
		const float sy = (oh > 1) ? float(lh - 1) / float(oh - 1) : 0.0f;
		std::vector<int> x0(ow);
		std::vector<float> wx(ow);
		for (int x = 0; x < ow; x++) {
			float fx = x * sx;
			int i = int(fx);
			if (i > lw - 2) i = std::max(0, lw - 2);
			x0[x] = i;
			wx[x] = (lw > 1) ? fx - float(i) : 0.0f;
		}

		const int nthreads = std::max(1, opt.num_threads);
		/*
		 * Scratch is allocated ONCE per forward and indexed by thread, not
		 * per row. Allocating inside the row loop cost ~1100 mallocs of
		 * ~70 KB each per frame and wiped the cache the fusion exists to
		 * exploit -- it made the deeper fusion measure no faster than the
		 * shallow one.
		 */
		const size_t vstride = size_t(16) * lw;
		const size_t lstride = size_t(4) * ow * 4;
		std::vector<float> scratch(size_t(nthreads) * (vstride + lstride));

#ifdef _OPENMP
		#pragma omp parallel for num_threads(nthreads)
#endif
		for (int y = 0; y < oh; y++) {
#ifdef _OPENMP
			const int tid = omp_get_thread_num();
#else
			const int tid = 0;
#endif
			float *vrow = scratch.data() + size_t(tid) * (vstride + lstride);
			float *line = vrow + vstride;

			float fy = y * sy;
			int y0 = int(fy);
			if (y0 > lh - 2) y0 = std::max(0, lh - 2);
			const float wy = (lh > 1) ? fy - float(y0) : 0.0f;
			const int y1 = std::min(y0 + 1, lh - 1);

			for (int gch = 0; gch < 16; gch++) {
				const int pl = gch / lep, lane = gch % lep;
				float *dst = vrow + size_t(gch) * lw;
				if (es == 4) {
					const float *p = (const float *)lr.channel(pl);
					for (int i = 0; i < lw; i++) {
						const float v0 = p[(size_t(y0) * lw + i) * lep + lane];
						const float v1 = p[(size_t(y1) * lw + i) * lep + lane];
						dst[i] = v0 + (v1 - v0) * wy;
					}
				} else {
					const __fp16 *p = (const __fp16 *)lr.channel(pl);
					for (int i = 0; i < lw; i++) {
						const float v0 = float(p[(size_t(y0) * lw + i) * lep + lane]);
						const float v1 = float(p[(size_t(y1) * lw + i) * lep + lane]);
						dst[i] = v0 + (v1 - v0) * wy;
					}
				}
			}

			/*
			 * Horizontal blend into the interleaved lines.
			 *
			 * x is the OUTER loop and the four channels are the vector
			 * lanes, so each pixel is one 4-lane FMA and one contiguous
			 * store. Written the other way round (c outer, x inner) the
			 * store is dl[x*4 + c] -- stride 4, one scalar float at a
			 * time, 8.5M of them per frame. Same arithmetic, but the
			 * lane layout already matches ncnn's elempack=4, so there is
			 * no reason to walk it strided.
			 */
			for (int grp = 0; grp < 4; grp++) {
				float *dl = line + size_t(grp) * ow * 4;
				const float *v0 = vrow + size_t(grp * 4 + 0) * lw;
				const float *v1 = vrow + size_t(grp * 4 + 1) * lw;
				const float *v2 = vrow + size_t(grp * 4 + 2) * lw;
				const float *v3 = vrow + size_t(grp * 4 + 3) * lw;
				for (int x = 0; x < ow; x++) {
					const int i = x0[x];
					const float w = wx[x];
#if __ARM_NEON
					const float32x4_t lo = {
						v0[i], v1[i], v2[i], v3[i]
					};
					const float32x4_t hi = {
						v0[i + 1], v1[i + 1], v2[i + 1], v3[i + 1]
					};
					vst1q_f32(dl + size_t(x) * 4,
						  vfmaq_n_f32(lo, vsubq_f32(hi, lo), w));
#else
					dl[size_t(x) * 4 + 0] = v0[i] + (v0[i + 1] - v0[i]) * w;
					dl[size_t(x) * 4 + 1] = v1[i] + (v1[i + 1] - v1[i]) * w;
					dl[size_t(x) * 4 + 2] = v2[i] + (v2[i + 1] - v2[i]) * w;
					dl[size_t(x) * 4 + 3] = v3[i] + (v3[i + 1] - v3[i]) * w;
#endif
				}
			}

			float *la = line;
			float *lb = line + size_t(1) * ow * 4;
			float *lm = line + size_t(2) * ow * 4;
			float *lw_ = line + size_t(3) * ow * 4;
			sigmoidInPlace(la, size_t(ow) * 4);
			sigmoidInPlace(lm, size_t(ow) * 4);
			sigmoidInPlace(lw_, size_t(ow) * 4);

			const size_t n = size_t(ow) * 4;
			const size_t off = size_t(y) * ow * 4;
			if (es == 4)
				assembleRow((const float *)base.channel(0) + off,
						   (const float *)detail->channel(0) + off,
						   (const float *)det->channel(0) + off,
						   (const float *)g.channel(0) + off,
						   la, lb, lm, lw_,
						   (float *)out.channel(0) + off, n);
			else
				assembleRow((const __fp16 *)base.channel(0) + off,
						    (const __fp16 *)detail->channel(0) + off,
						    (const __fp16 *)det->channel(0) + off,
						    (const __fp16 *)g.channel(0) + off,
						    la, lb, lm, lw_,
						    (__fp16 *)out.channel(0) + off, n);
		}

		return 0;
	}

private:

	/*
	 * FAST PATH -- fp16 with a packed coefficient tensor (elempack 4 or 8).
	 *
	 * The general path above costs 18.4 ms of a 37 ms forward against a
	 * 2.2 MB/frame... 21 MB/frame footprint whose bandwidth floor is 2.2 ms.
	 * Eight times the floor, for two structural reasons, both fixed here:
	 *
	 *  1. THE VERTICAL LERP WAS SCALAR. It walked one channel lane at a
	 *     time with stride `lep`. But ncnn packs the coefficient tensor so
	 *     that the FOUR CHANNELS OF ONE GROUP ARE CONTIGUOUS LANES within a
	 *     pixel -- group g lives at lane offset (4g mod lep) of plane
	 *     (4g div lep). So a group's four channels are one 4-wide load, and
	 *     the lerp is one FMA, for any lep that is a multiple of 4.
	 *
	 *  2. vrow WAS CHANNEL-MAJOR, so the horizontal blend had to gather
	 *     v0[i], v1[i], v2[i], v3[i] -- eight scalar loads per output pixel
	 *     to build two vectors. Storing vrow PIXEL-major (i*4 + c) makes
	 *     those two contiguous vector loads instead.
	 *
	 * With the layout fixed, the horizontal blend, the three sigmoids and
	 * the assemble all fuse into one pass that never leaves registers. That
	 * also deletes the `line` buffer: 4 groups x ow x 4 floats is 62 KB per
	 * row, previously written once and read four times.
	 *
	 * Semantics are unchanged: sigmoid is still applied AFTER interpolation
	 * (sigmoid(lerp(x)) != lerp(sigmoid(x))), still to a, m and w but not b.
	 */
	int forwardFast(const ncnn::Mat &lr, const ncnn::Mat *det,
			const ncnn::Mat &base, const ncnn::Mat *detail,
			const ncnn::Mat &g, ncnn::Mat &out, int lep,
			const ncnn::Option &opt) const
	{
		const int ow = g.w, oh = g.h;
		const int lw = lr.w, lh = lr.h;

		const float sx = (ow > 1) ? float(lw - 1) / float(ow - 1) : 0.0f;
		const float sy = (oh > 1) ? float(lh - 1) / float(oh - 1) : 0.0f;
		std::vector<int> x0(ow);
		std::vector<float> wx(ow);
		for (int x = 0; x < ow; x++) {
			float fx = x * sx;
			int i = int(fx);
			if (i > lw - 2) i = std::max(0, lw - 2);
			x0[x] = i;
			wx[x] = (lw > 1) ? fx - float(i) : 0.0f;
		}

		const int nthreads = std::max(1, opt.num_threads);
		/* 4 groups x lw pixels x 4 channels, pixel-major */
		const size_t vstride = size_t(4) * size_t(lw) * 4;
		std::vector<float> scratch(size_t(nthreads) * vstride);

		const float32x4_t vAMin = vdupq_n_f32(aMin_);
		const float32x4_t vOne = vdupq_n_f32(1.0f);
		const float32x4_t vLo = vdupq_n_f32(loQ_);
		const float32x4_t vHi = vdupq_n_f32(hiQ_);

#ifdef _OPENMP
		#pragma omp parallel for num_threads(nthreads) schedule(static)
#endif
		for (int y = 0; y < oh; y++) {
#ifdef _OPENMP
			const int tid = omp_get_thread_num();
#else
			const int tid = 0;
#endif
			float *vrow = scratch.data() + size_t(tid) * vstride;

			float fy = y * sy;
			int y0 = int(fy);
			if (y0 > lh - 2) y0 = std::max(0, lh - 2);
			const float wy = (lh > 1) ? fy - float(y0) : 0.0f;
			const int y1 = std::min(y0 + 1, lh - 1);
			const float32x4_t vWy = vdupq_n_f32(wy);

			for (int grp = 0; grp < 4; grp++) {
				const int pl = (4 * grp) / lep;
				const int off = (4 * grp) % lep;
				const __fp16 *p0 = (const __fp16 *)lr.channel(pl)
						   + size_t(y0) * lw * lep + off;
				const __fp16 *p1 = (const __fp16 *)lr.channel(pl)
						   + size_t(y1) * lw * lep + off;
				float *dst = vrow + size_t(grp) * lw * 4;
				for (int i = 0; i < lw; i++) {
					const float32x4_t a =
						vcvt_f32_f16(vld1_f16(p0 + size_t(i) * lep));
					const float32x4_t b =
						vcvt_f32_f16(vld1_f16(p1 + size_t(i) * lep));
					vst1q_f32(dst + size_t(i) * 4,
						  vfmaq_f32(a, vsubq_f32(b, a), vWy));
				}
			}

			const float *vA = vrow;
			const float *vB = vrow + size_t(1) * lw * 4;
			const float *vM = vrow + size_t(2) * lw * 4;
			const float *vW = vrow + size_t(3) * lw * 4;

			const size_t off = size_t(y) * ow * 4;
			const __fp16 *pbase = (const __fp16 *)base.channel(0) + off;
			const __fp16 *pdet1 = detail ? (const __fp16 *)detail->channel(0) + off
						     : nullptr;
			const __fp16 *pdet0 = det ? (const __fp16 *)det->channel(0) + off
						  : nullptr;
			const __fp16 *pg = (const __fp16 *)g.channel(0) + off;
			__fp16 *po = (__fp16 *)out.channel(0) + off;

			for (int x = 0; x < ow; x++) {
				const size_t i = size_t(x0[x]) * 4;
				const float32x4_t vwx = vdupq_n_f32(wx[x]);
				const float32x4_t va = lerp4(vA + i, vwx);
				const float32x4_t vb = lerp4(vB + i, vwx);
				const float32x4_t vm = lerp4(vM + i, vwx);
				const float32x4_t vw = lerp4(vW + i, vwx);

				const float32x4_t a = vaddq_f32(sigmoid4(va), vAMin);
				const float32x4_t m = sigmoid4(vm);
				const float32x4_t w = sigmoid4(vw);

				const size_t px = size_t(x) * 4;
				const float32x4_t bs = vcvt_f32_f16(vld1_f16(pbase + px));
				const float32x4_t gg = vcvt_f32_f16(vld1_f16(pg + px));
				/* detail = guide - base, already in registers */
				float32x4_t dd = pdet1
					? vcvt_f32_f16(vld1_f16(pdet1 + px))
					: vsubq_f32(gg, bs);
				if (pdet0)
					dd = vaddq_f32(dd, vcvt_f32_f16(vld1_f16(pdet0 + px)));

				float32x4_t body = vfmaq_f32(vb, a, bs);
				body = vfmaq_f32(body, m, dd);
				float32x4_t v = vmulq_f32(gg, w);
				v = vfmaq_f32(v, vsubq_f32(vOne, w), body);
				v = vminq_f32(vmaxq_f32(v, vLo), vHi);
				vst1_f16(po + px, vcvt_f16_f32(v));
			}
		}

		return 0;
	}

	/* one horizontal lerp between pixel i and i+1, four channels at once */
	static inline float32x4_t lerp4(const float *p, float32x4_t w)
	{
		const float32x4_t a = vld1q_f32(p);
		const float32x4_t b = vld1q_f32(p + 4);
		return vfmaq_f32(a, vsubq_f32(b, a), w);
	}

	/* same exp2-by-bit-trick approximation as sigmoidInPlace, one vector */
	static inline float32x4_t sigmoid4(float32x4_t xin)
	{
		const float32x4_t one = vdupq_n_f32(1.0f);
		const float32x4_t lim = vdupq_n_f32(60.0f);
		float32x4_t x = vnegq_f32(xin);
		x = vminq_f32(vmaxq_f32(x, vnegq_f32(lim)), lim);
		float32x4_t t = vmulq_f32(x, vdupq_n_f32(1.44269504f));
		float32x4_t fl = vrndmq_f32(t);
		float32x4_t fr = vsubq_f32(t, fl);
		float32x4_t poly = vfmaq_f32(vdupq_n_f32(0.0555041f),
					     vdupq_n_f32(0.0096181f), fr);
		poly = vfmaq_f32(vdupq_n_f32(0.2402265f), poly, fr);
		poly = vfmaq_f32(vdupq_n_f32(0.6931472f), poly, fr);
		poly = vfmaq_f32(vdupq_n_f32(0.9999999f), poly, fr);
		int32x4_t e = vcvtq_s32_f32(fl);
		int32x4_t bits = vshlq_n_s32(vaddq_s32(e, vdupq_n_s32(127)), 23);
		float32x4_t ex = vmulq_f32(vreinterpretq_f32_s32(bits), poly);
		return vdivq_f32(one, vaddq_f32(one, ex));
	}

	/*
	 * expf() per element would dominate: 3 sigmoids x 968 x 4 x 548 is 6.4M
	 * calls per frame, tens of ms scalar. This is the usual exp2-by-bit-trick
	 * approximation, max abs error ~1e-6 on sigmoid's range, four at a time.
	 */
	static inline void sigmoidInPlace(float *p, size_t n)
	{
		size_t i = 0;
#if __ARM_NEON
		const float32x4_t one = vdupq_n_f32(1.0f);
		const float32x4_t log2e = vdupq_n_f32(1.44269504f);
		const float32x4_t c0 = vdupq_n_f32(0.9999999f);
		const float32x4_t c1 = vdupq_n_f32(0.6931472f);
		const float32x4_t c2 = vdupq_n_f32(0.2402265f);
		const float32x4_t c3 = vdupq_n_f32(0.0555041f);
		const float32x4_t c4 = vdupq_n_f32(0.0096181f);
		const float32x4_t lim = vdupq_n_f32(60.0f);
		for (; i + 4 <= n; i += 4) {
			/* sigmoid(x) = 1 / (1 + exp(-x)) */
			float32x4_t x = vnegq_f32(vld1q_f32(p + i));
			x = vminq_f32(vmaxq_f32(x, vnegq_f32(lim)), lim);
			float32x4_t t = vmulq_f32(x, log2e);
			float32x4_t fl = vrndmq_f32(t);
			float32x4_t fr = vsubq_f32(t, fl);          /* frac in [0,1) */
			/*
			 * 2^fr, minimax polynomial in fr ITSELF. These coefficients
			 * are for 2^x, not e^x -- scaling fr by ln2 first (which is
			 * what a e^u series would want) evaluates the wrong function
			 * and cost 12/255 of output error before it was caught.
			 */
			float32x4_t poly = vfmaq_f32(c3, c4, fr);
			poly = vfmaq_f32(c2, poly, fr);
			poly = vfmaq_f32(c1, poly, fr);
			poly = vfmaq_f32(c0, poly, fr);
			/* 2^fl by exponent injection */
			int32x4_t e = vcvtq_s32_f32(fl);
			int32x4_t bits = vshlq_n_s32(vaddq_s32(e, vdupq_n_s32(127)), 23);
			float32x4_t p2 = vreinterpretq_f32_s32(bits);
			float32x4_t ex = vmulq_f32(p2, poly);
			vst1q_f32(p + i, vdivq_f32(one, vaddq_f32(one, ex)));
		}
#endif
		for (; i < n; i++)
			p[i] = 1.0f / (1.0f + std::exp(-p[i]));
	}

	/*
	 * The output loop, 4 lanes at a time.
	 *
	 * The previous BilateralAssemble did this in NEON (fp32 4-wide and fp16
	 * 8-wide); the LR rewrite fused the upsample in -- which was the right
	 * call, it removed ~59 MB/frame of DRAM traffic -- but left this loop
	 * scalar. That is 2.1M scalar FMAs per frame on a core that issues four
	 * at once, and profiling put the model at ~40% memory-bound and the rest
	 * compute, so this is where the rest went.
	 *
	 * NOTE the mixed precision: a/b/m/w are float (they come from the
	 * interpolation, which runs in float), while base/detail/det/g/o are T.
	 * So this cannot be the old fp16 8-wide loop copied across -- when T is
	 * __fp16 the image data is widened to float, the maths is done in float,
	 * and the result is narrowed on store. Same arithmetic as the scalar
	 * version, same rounding, 4 lanes instead of 1.
	 */
	inline void assembleRow(const float *base, const float *detail, const float *det,
				const float *g, const float *a, const float *b,
				const float *m, const float *w, float *o, size_t n) const
	{
		size_t i = 0;
#if __ARM_NEON
		const float32x4_t vAMin = vdupq_n_f32(aMin_);
		const float32x4_t vOne = vdupq_n_f32(1.0f);
		const float32x4_t vLo = vdupq_n_f32(loQ_);
		const float32x4_t vHi = vdupq_n_f32(hiQ_);
		for (; i + 4 <= n; i += 4) {
			float32x4_t va = vaddq_f32(vld1q_f32(a + i), vAMin);
			float32x4_t vw = vld1q_f32(w + i);
			float32x4_t vdd = vaddq_f32(vld1q_f32(detail + i), vld1q_f32(det + i));
			float32x4_t body = vld1q_f32(b + i);
			body = vfmaq_f32(body, va, vld1q_f32(base + i));
			body = vfmaq_f32(body, vld1q_f32(m + i), vdd);
			float32x4_t v = vmulq_f32(vld1q_f32(g + i), vw);
			v = vfmaq_f32(v, vsubq_f32(vOne, vw), body);
			vst1q_f32(o + i, vminq_f32(vmaxq_f32(v, vLo), vHi));
		}
#endif
		assembleTail<float>(base, detail, det, g, a, b, m, w, o, i, n);
	}

#if defined(__ARM_FP16_FORMAT_IEEE) && __ARM_NEON
	inline void assembleRow(const __fp16 *base, const __fp16 *detail,
				const __fp16 *det, const __fp16 *g, const float *a,
				const float *b, const float *m, const float *w,
				__fp16 *o, size_t n) const
	{
		size_t i = 0;
		const float32x4_t vAMin = vdupq_n_f32(aMin_);
		const float32x4_t vOne = vdupq_n_f32(1.0f);
		const float32x4_t vLo = vdupq_n_f32(loQ_);
		const float32x4_t vHi = vdupq_n_f32(hiQ_);
		for (; i + 4 <= n; i += 4) {
			float32x4_t vbase = vcvt_f32_f16(vld1_f16(base + i));
			float32x4_t vdt = vcvt_f32_f16(vld1_f16(detail + i));
			float32x4_t vde = vcvt_f32_f16(vld1_f16(det + i));
			float32x4_t vg = vcvt_f32_f16(vld1_f16(g + i));
			float32x4_t va = vaddq_f32(vld1q_f32(a + i), vAMin);
			float32x4_t vw = vld1q_f32(w + i);
			float32x4_t body = vld1q_f32(b + i);
			body = vfmaq_f32(body, va, vbase);
			body = vfmaq_f32(body, vld1q_f32(m + i), vaddq_f32(vdt, vde));
			float32x4_t v = vmulq_f32(vg, vw);
			v = vfmaq_f32(v, vsubq_f32(vOne, vw), body);
			v = vminq_f32(vmaxq_f32(v, vLo), vHi);
			vst1_f16(o + i, vcvt_f16_f32(v));
		}
		assembleTail<__fp16>(base, detail, det, g, a, b, m, w, o, i, n);
	}
#endif

	template<typename T>
	inline void assembleTail(const T *base, const T *detail, const T *det, const T *g,
				 const float *a, const float *b, const float *m,
				 const float *w, T *o, size_t i, size_t n) const
	{
		for (; i < n; i++) {
			const float body = (a[i] + aMin_) * float(base[i]) + b[i] +
					   m[i] * (float(detail[i]) + float(det[i]));
			float v = (1.0f - w[i]) * body + w[i] * float(g[i]);
			v = std::min(std::max(v, loQ_), hiQ_);
			o[i] = T(v);
		}
	}

	float aMin_ = 0.1f;
	float loQ_ = 0.0f;
	float hiQ_ = 1.0f;
};

::ncnn::Layer *BilateralAssembleLR_layer_creator(void *userdata);
DEFINE_LAYER_CREATOR(BilateralAssembleLR)

} /* namespace RPiController */
