/* SPDX-License-Identifier: BSD-2-Clause */
/*
 * bilateral_assemble.h - fused guided-filter assemble for the bilateral grid net
 *
 * Replaces the eleven trailing elementwise layers of bilateral_grid_*.param
 * with one pass:
 *
 *   out = clip( (1-w) * ( (a + A_MIN)*base + b + m*(detail + det) ) + w*g, 0, 1 )
 *
 * WHY. Those eleven layers do one arithmetic operation per pixel each, and each
 * one reads and writes the whole full-resolution tensor to do it. At 968x548x4
 * that is ~4.2 MB per tensor in fp16, so the chain costs ~30 full-resolution
 * tensor traversals to perform about a dozen flops per pixel. The net is bound
 * by that traffic, not by arithmetic: 161 MMAC/frame against a measured 125 ms
 * is under 3% of what the A76s can issue. Fusing turns ~30 traversals into 9.
 *
 * LAYOUT INDEPENDENCE. Every operation here is elementwise and every input has
 * the same shape (w, h, 4), so the kernel never needs to know how ncnn has laid
 * the data out. It walks all tensors flat, in lockstep. That is what lets the
 * layer accept fp16 and packed layouts unchanged -- declaring otherwise would
 * make ncnn insert Cast/Packing layers on all eight inputs and reintroduce
 * exactly the traffic this removes.
 */
#pragma once

#include <algorithm>

#if defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>
#endif

#include <ncnn/layer.h>
#include <ncnn/mat.h>

namespace RPiController {

class BilateralAssemble : public ncnn::Layer
{
public:
	BilateralAssemble()
	{
		one_blob_only = false;
		support_inplace = false;
		/*
		 * Handled natively -- see LAYOUT INDEPENDENCE above. If these were
		 * left false ncnn would convert eight full-resolution inputs before
		 * calling us.
		 */
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

	/*
	 * bottoms, in order:
	 *   0 a      sigmoid(a_raw), upsampled      4 b     upsampled affine offset
	 *   1 m      sigmoid(m_raw), the gate       5 base  blur5(guide)
	 *   2 w      sigmoid(skip weight)           6 detail guide - base
	 *   3 det    detail-branch output           7 g     guide (temporal mean)
	 */
	int forward(const std::vector<ncnn::Mat> &bots, std::vector<ncnn::Mat> &outs,
		    const ncnn::Option &opt) const override
	{
		if (bots.size() != 8)
			return -1;

		const ncnn::Mat &a = bots[0];
		const ncnn::Mat &m = bots[1];
		const ncnn::Mat &wgt = bots[2];
		const ncnn::Mat &det = bots[3];
		const ncnn::Mat &b = bots[4];
		const ncnn::Mat &base = bots[5];
		const ncnn::Mat &detail = bots[6];
		const ncnn::Mat &g = bots[7];

		/*
		 * Lockstep walking is only valid if every input really does have the
		 * same element count and element size. Bail rather than read past an
		 * end if a future export changes a shape.
		 */
		const size_t n = size_t(a.w) * a.h * a.c * a.elempack;
		const size_t es = a.elemsize / a.elempack;
		for (const ncnn::Mat &t : bots) {
			if (size_t(t.w) * t.h * t.c * t.elempack != n ||
			    t.elemsize / t.elempack != es)
				return -1;
		}

		ncnn::Mat &out = outs[0];
		out.create_like(a, opt.blob_allocator);
		if (out.empty())
			return -100;

		if (es == 4)
			runF32(a, m, wgt, det, b, base, detail, g, out, n);
#if defined(__ARM_FP16_FORMAT_IEEE)
		else if (es == 2)
			runF16(a, m, wgt, det, b, base, detail, g, out, n);
#endif
		else
			return -1;

		return 0;
	}

private:
	/*
	 * The kernels below are vectorised, and that is not an optimisation
	 * detail -- it is the whole result. A scalar version of this layer
	 * measured 0.99x against the eleven layers it replaces in fp16: ncnn's
	 * BinaryOp kernels are NEON fp16, so replacing ten vectorised passes
	 * with one scalar pass trades memory traffic for arithmetic throughput
	 * and breaks even. The single pass only pays once it issues as wide as
	 * what it replaced.
	 */
	void runF32(const ncnn::Mat &a, const ncnn::Mat &m, const ncnn::Mat &wgt,
		    const ncnn::Mat &det, const ncnn::Mat &b, const ncnn::Mat &base,
		    const ncnn::Mat &detail, const ncnn::Mat &g, ncnn::Mat &out,
		    size_t n) const;
#if defined(__ARM_FP16_FORMAT_IEEE)
	void runF16(const ncnn::Mat &a, const ncnn::Mat &m, const ncnn::Mat &wgt,
		    const ncnn::Mat &det, const ncnn::Mat &b, const ncnn::Mat &base,
		    const ncnn::Mat &detail, const ncnn::Mat &g, ncnn::Mat &out,
		    size_t n) const;
#endif

	template<typename T>
	inline void scalarTail(const T *pa, const T *pm, const T *pw, const T *pd,
			       const T *pb, const T *pbase, const T *pdet,
			       const T *pg, T *po, size_t i, size_t n) const
	{
		for (; i < n; i++) {
			const float av = float(pa[i]) + aMin_;
			const float wv = float(pw[i]);
			const float body = av * float(pbase[i]) + float(pb[i]) +
					   float(pm[i]) * (float(pdet[i]) + float(pd[i]));
			float v = (1.0f - wv) * body + wv * float(pg[i]);
			v = std::min(std::max(v, loQ_), hiQ_);
			po[i] = T(v);
		}
	}

	float aMin_ = 0.1f;
	float loQ_ = 0.0f;
	float hiQ_ = 1.0f;
};

/* ---- fp32: 4-wide ------------------------------------------------------- */
inline void BilateralAssemble::runF32(const ncnn::Mat &a, const ncnn::Mat &m,
				      const ncnn::Mat &wgt, const ncnn::Mat &det,
				      const ncnn::Mat &b, const ncnn::Mat &base,
				      const ncnn::Mat &detail, const ncnn::Mat &g,
				      ncnn::Mat &out, size_t n) const
{
	const float *pa = a, *pm = m, *pw = wgt, *pd = det;
	const float *pb = b, *pbase = base, *pdet = detail, *pg = g;
	float *po = out;
	size_t i = 0;
#if __ARM_NEON
	const float32x4_t vAMin = vdupq_n_f32(aMin_);
	const float32x4_t vOne = vdupq_n_f32(1.0f);
	const float32x4_t vLo = vdupq_n_f32(loQ_);
	const float32x4_t vHi = vdupq_n_f32(hiQ_);
	for (; i + 4 <= n; i += 4) {
		float32x4_t va = vaddq_f32(vld1q_f32(pa + i), vAMin);
		float32x4_t vw = vld1q_f32(pw + i);
		float32x4_t vdd = vaddq_f32(vld1q_f32(pdet + i), vld1q_f32(pd + i));
		float32x4_t body = vld1q_f32(pb + i);
		body = vfmaq_f32(body, va, vld1q_f32(pbase + i));
		body = vfmaq_f32(body, vld1q_f32(pm + i), vdd);
		float32x4_t v = vmulq_f32(vld1q_f32(pg + i), vw);
		v = vfmaq_f32(v, vsubq_f32(vOne, vw), body);
		vst1q_f32(po + i, vminq_f32(vmaxq_f32(v, vLo), vHi));
	}
#endif
	scalarTail<float>(pa, pm, pw, pd, pb, pbase, pdet, pg, po, i, n);
}

/* ---- fp16: 8-wide, native half arithmetic ------------------------------- */
#if defined(__ARM_FP16_FORMAT_IEEE)
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
#define BA_FP16_TARGET
#else
/*
 * libcamera does not build with +fp16 globally, so request it per function
 * rather than losing the vector path (and with it the entire speedup).
 */
#define BA_FP16_TARGET __attribute__((target("arch=armv8.2-a+fp16")))
#endif

BA_FP16_TARGET
inline void BilateralAssemble::runF16(const ncnn::Mat &a, const ncnn::Mat &m,
				      const ncnn::Mat &wgt, const ncnn::Mat &det,
				      const ncnn::Mat &b, const ncnn::Mat &base,
				      const ncnn::Mat &detail, const ncnn::Mat &g,
				      ncnn::Mat &out, size_t n) const
{
	const __fp16 *pa = a, *pm = m, *pw = wgt, *pd = det;
	const __fp16 *pb = b, *pbase = base, *pdet = detail, *pg = g;
	__fp16 *po = out;
	size_t i = 0;
	const float16x8_t vAMin = vdupq_n_f16((__fp16)aMin_);
	const float16x8_t vOne = vdupq_n_f16((__fp16)1.0f);
	const float16x8_t vLo = vdupq_n_f16((__fp16)loQ_);
	const float16x8_t vHi = vdupq_n_f16((__fp16)hiQ_);
	for (; i + 8 <= n; i += 8) {
		float16x8_t va = vaddq_f16(vld1q_f16(pa + i), vAMin);
		float16x8_t vw = vld1q_f16(pw + i);
		float16x8_t vdd = vaddq_f16(vld1q_f16(pdet + i), vld1q_f16(pd + i));
		float16x8_t body = vld1q_f16(pb + i);
		body = vfmaq_f16(body, va, vld1q_f16(pbase + i));
		body = vfmaq_f16(body, vld1q_f16(pm + i), vdd);
		float16x8_t v = vmulq_f16(vld1q_f16(pg + i), vw);
		v = vfmaq_f16(v, vsubq_f16(vOne, vw), body);
		vst1q_f16(po + i, vminq_f16(vmaxq_f16(v, vLo), vHi));
	}
	scalarTail<__fp16>(pa, pm, pw, pd, pb, pbase, pdet, pg, po, i, n);
}
#endif /* __ARM_FP16_FORMAT_IEEE */

::ncnn::Layer *BilateralAssemble_layer_creator(void *userdata);
DEFINE_LAYER_CREATOR(BilateralAssemble)

} /* namespace RPiController */
