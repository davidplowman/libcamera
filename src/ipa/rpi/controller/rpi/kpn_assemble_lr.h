/* SPDX-License-Identifier: BSD-2-Clause */
/*
 * kpn_assemble_lr.h - fused KPN + bilateral-grid output tail.
 *
 * Replaces the ~50 layer exported tail (Interp x2, Softmax, Pad, Gather x2,
 * Permute, Reshape x3, GatherElements x2, ~12 BinaryOps, Reduction x2) with a
 * single row-parallel pass.  Nothing full-resolution is ever materialised: the
 * 9 kernel logits and the 160 grid coefficients are bilinearly interpolated
 * into a per-thread line buffer and consumed immediately, so the 38 MB/frame
 * of kernel planes and 170 MB/frame of grid planes the exported graph would
 * write and re-read never reach DRAM.
 *
 * Also sidesteps an export bug: pnnx emits the unfold's zero-pad as a bare
 * 'Pad' with no parameters, which stock ncnn cannot even register.
 */
#pragma once

#include <algorithm>
#include <cmath>
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

class KpnAssembleLR : public ncnn::Layer
{
public:
	KpnAssembleLR()
	{
		one_blob_only = false;
		support_inplace = false;
		support_packing = true;
		support_fp16_storage = true;
		support_bf16_storage = false;
	}

	int load_param(const ncnn::ParamDict &pd) override
	{
		taps_ = pd.get(0, 9);
		bins_ = pd.get(1, 8);
		outCh_ = pd.get(2, 4);
		/* 3: clamp the output to [0,1]. Correct when this layer IS the
		 * graph output (m9). WRONG mid-graph (h2), where the box, the
		 * full-res stack and the LF locks still follow. */
		clip_ = pd.get(3, 1);
		kside_ = int(std::lround(std::sqrt(double(taps_))));
		coeffs_ = outCh_ * (outCh_ + 1);
		return 0;
	}

	/*
	 * bots: 0 = coeff LR (taps + coeffs*bins ch), 1 = live (outCh, full res)
	 *       2 = guide (1 ch, full res) -- OPTIONAL
	 *
	 * With two bottoms the guide is derived here instead.  It is only
	 * clip(box3(mean of the four planes), 0, 1) with fixed 1/9 weights and no
	 * learnable parameters, but as exported it is a Slice into four 1-channel
	 * full-res tensors, three BinaryOps, a scale and a 3x3 Convolution -- six
	 * layers on 1-channel full-res blobs.  ncnn threads elementwise work over
	 * channels, so at one channel every one of them runs single-threaded, and
	 * together they cost more than this entire layer.  Folded in here the guide
	 * is free: it reuses the same nine taps the kernel apply already loads.
	 */
	int forward(const std::vector<ncnn::Mat> &bots, std::vector<ncnn::Mat> &outs,
		    const ncnn::Option &opt) const override
	{
		if (bots.size() != 2 && bots.size() != 3)
			return -1;

		const bool selfGuide = (bots.size() == 2);
		const ncnn::Mat &lr = bots[0];
		const ncnn::Mat &live = bots[1];
		const ncnn::Mat &g = selfGuide ? live : bots[2];

		const int ow = live.w, oh = live.h;
		const int lw = lr.w, lh = lr.h, lep = lr.elempack;
		const int nch = taps_ + coeffs_ * bins_;
		const size_t esL = lr.elemsize / lep;
		const size_t esV = live.elemsize / live.elempack;
		const size_t esG = g.elemsize / g.elempack;
		const int vep = live.elempack;

		if (lr.c * lep < nch)
			return -1;
		/* the folded guide is a 3x3 box; wider kernels must pass one in */
		if (selfGuide && kside_ != 3)
			return -1;

		ncnn::Mat &out = outs[0];
		out.create(ow, oh, outCh_ / vep, live.elemsize, vep, opt.blob_allocator);
		if (out.empty())
			return -100;

		/* align_corners = 1, matching the exported Interp 6=1 */
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

		/*
		 * Line buffer is PIXEL-major, not channel-major, and the grid
		 * channels are re-ordered on the way in:
		 *
		 *   vrow[i * stride + 0 .. taps-1]              kernel logits
		 *   vrow[i * stride + taps + z*coeffs + k]      grid coeff k, bin z
		 *
		 * The export stores coefficient k at LR channel taps + k*bins + z,
		 * so a pixel needing all 20 coefficients of one luma bin would walk
		 * a stride-8 gather across 82 KB.  Re-ordering to bin-major makes
		 * the 20 coefficients of bin z contiguous AND puts bin z+1 in the
		 * next 20 floats, so the whole slice is two 80-byte streams.
		 */
		const int stride = (nch + 7) & ~7;
		const int nthreads = std::max(1, opt.num_threads);
		std::vector<float> scratch(size_t(nthreads) * size_t(stride) * lw);

		/* src LR channel -> line-buffer slot */
		std::vector<int> slot(nch);
		for (int t = 0; t < taps_; t++)
			slot[t] = t;
		for (int k = 0; k < coeffs_; k++)
			for (int z = 0; z < bins_; z++)
				slot[taps_ + k * bins_ + z] = taps_ + z * coeffs_ + k;

		const int pad = kside_ / 2;
		const bool packed = (vep == outCh_);

#ifdef _OPENMP
		#pragma omp parallel for num_threads(nthreads)
#endif
		for (int y = 0; y < oh; y++) {
#ifdef _OPENMP
			const int tid = omp_get_thread_num();
#else
			const int tid = 0;
#endif
			float *vrow = scratch.data() + size_t(tid) * size_t(stride) * lw;

			float fy = y * sy;
			int y0 = int(fy);
			if (y0 > lh - 2) y0 = std::max(0, lh - 2);
			const float wy = (lh > 1) ? fy - float(y0) : 0.0f;
			const int y1 = std::min(y0 + 1, lh - 1);

			/*
			 * Blocked transpose.  The naive nesting (channel outer, i inner)
			 * writes with a 704-byte stride and so sweeps the whole 85 KB line
			 * buffer once per channel -- 169 sweeps per output row, ~8 GB/frame
			 * of L2 traffic.  Walking i in tiles of 16 keeps the live window at
			 * 16 * stride * 4 = 11 KB, which stays in L1 for the whole tile.
			 */
			const int IB = 16;
			for (int ib = 0; ib < lw; ib += IB) {
				const int ie = std::min(ib + IB, lw);
				for (int ch = 0; ch < nch; ch++) {
					const int pl = ch / lep, lane = ch % lep;
					float *dst = vrow + slot[ch];
					if (esL == 4) {
						const float *p = (const float *)lr.channel(pl);
						for (int i = ib; i < ie; i++) {
							const float v0 = p[(size_t(y0) * lw + i) * lep + lane];
							const float v1 = p[(size_t(y1) * lw + i) * lep + lane];
							dst[size_t(i) * stride] = v0 + (v1 - v0) * wy;
						}
					} else {
						const __fp16 *p = (const __fp16 *)lr.channel(pl);
						for (int i = ib; i < ie; i++) {
							const float v0 = float(p[(size_t(y0) * lw + i) * lep + lane]);
							const float v1 = float(p[(size_t(y1) * lw + i) * lep + lane]);
							dst[size_t(i) * stride] = v0 + (v1 - v0) * wy;
						}
					}
				}
			}

			const float *gp32 = (!selfGuide && esG == 4) ? (const float *)g.channel(0) : nullptr;
			const __fp16 *gp16 = (!selfGuide && esG == 2) ? (const __fp16 *)g.channel(0) : nullptr;
			const float *gr32 = gp32 ? gp32 + size_t(y) * ow : nullptr;
			const __fp16 *gr16 = gp16 ? gp16 + size_t(y) * ow : nullptr;

			/* the kside_ live rows this output row can reach */
			const float *lrow32[8] = { nullptr };
			const __fp16 *lrow16[8] = { nullptr };
			if (packed) {
				for (int dy = -pad, r = 0; dy <= pad; dy++, r++) {
					const int yy = y + dy;
					if (yy < 0 || yy >= oh) continue;
					if (esV == 4)
						lrow32[r] = (const float *)live.channel(0) + size_t(yy) * ow * vep;
					else
						lrow16[r] = (const __fp16 *)live.channel(0) + size_t(yy) * ow * vep;
				}
			}
			float *orow32 = (esV == 4) ? (float *)out.channel(0) + size_t(y) * ow * vep : nullptr;
			__fp16 *orow16 = (esV == 2) ? (__fp16 *)out.channel(0) + size_t(y) * ow * vep : nullptr;

			float wk[64], kpn[8], c20[64];

			for (int x = 0; x < ow; x++) {
				const int i = x0[x];
				const float fx = wx[x];
				const float *A = vrow + size_t(i) * stride;
				const float *B = A + stride;

				/* --- kernel logits -> softmax --- */
#if defined(__aarch64__)
				if (taps_ == 9) {
					/* 9 taps = two quads and a scalar; the exp is the
					 * single hottest thing in this layer (4.8 M calls a
					 * frame), so it runs 4-wide. */
					const float32x4_t fxv = vdupq_n_f32(fx);
					const float32x4_t a0 = vld1q_f32(A), b0 = vld1q_f32(B);
					const float32x4_t a1 = vld1q_f32(A + 4), b1 = vld1q_f32(B + 4);
					const float32x4_t L0 = vmlaq_f32(a0, vsubq_f32(b0, a0), fxv);
					const float32x4_t L1 = vmlaq_f32(a1, vsubq_f32(b1, a1), fxv);
					const float l8 = A[8] + (B[8] - A[8]) * fx;
					const float mx = std::max(std::max(vmaxvq_f32(L0), vmaxvq_f32(L1)), l8);
					const float32x4_t mxv = vdupq_n_f32(mx);
					const float32x4_t E0 = fastExpQ(vsubq_f32(L0, mxv));
					const float32x4_t E1 = fastExpQ(vsubq_f32(L1, mxv));
					const float e8 = fastExp(l8 - mx);
					const float inv = 1.0f / (vaddvq_f32(E0) + vaddvq_f32(E1) + e8);
					vst1q_f32(wk, vmulq_n_f32(E0, inv));
					vst1q_f32(wk + 4, vmulq_n_f32(E1, inv));
					wk[8] = e8 * inv;
				} else
#endif
				{
					float mx = -1e30f;
					for (int t = 0; t < taps_; t++) {
						const float l = A[t] + (B[t] - A[t]) * fx;
						wk[t] = l;
						if (l > mx) mx = l;
					}
					float sum = 0.0f;
					for (int t = 0; t < taps_; t++) {
						const float e = fastExp(wk[t] - mx);
						wk[t] = e;
						sum += e;
					}
					const float inv = 1.0f / sum;
					for (int t = 0; t < taps_; t++)
						wk[t] *= inv;
				}

				/* --- kernel apply; the fast path accumulates the guide too --- */
				float gv;
#if defined(__aarch64__)
				const bool fast = (packed && esV == 2 && outCh_ == 4);
				float32x4_t acc = vdupq_n_f32(0.0f);
				if (fast) {
					/*
					 * elempack 4 puts R/G1/G2/B of one pixel in 8 contiguous
					 * bytes, so a tap is one vld1_f16 plus one fmla-by-scalar.
					 * gacc sums the same nine taps unweighted, which is exactly
					 * the 3x3 box the guide needs -- the guide therefore costs
					 * nine adds rather than six full-res ncnn layers.
					 */
					float32x4_t gacc = vdupq_n_f32(0.0f);
					int t = 0;
					for (int r = 0; r < kside_; r++) {
						const __fp16 *rp = lrow16[r];
						if (!rp) { t += kside_; continue; }
						for (int dx = -pad; dx <= pad; dx++, t++) {
							const int xx = x + dx;
							if (xx < 0 || xx >= ow) continue;
							const float32x4_t v =
								vcvt_f32_f16(vld1_f16((const __fp16 *)(rp + size_t(xx) * 4)));
							acc = vfmaq_n_f32(acc, v, wk[t]);
							gacc = vaddq_f32(gacc, v);
						}
					}
					gv = selfGuide ? vaddvq_f32(gacc) * (0.25f / 9.0f)
						       : (gr32 ? gr32[x] : float(gr16[x]));
				} else
#endif
				{
					for (int c = 0; c < outCh_; c++)
						kpn[c] = 0.0f;
					float gsum = 0.0f;
					int t = 0;
					for (int dy = -pad; dy <= pad; dy++) {
						const int yy = y + dy;
						if (yy < 0 || yy >= oh) { t += kside_; continue; }
						for (int dx = -pad; dx <= pad; dx++, t++) {
							const int xx = x + dx;
							if (xx < 0 || xx >= ow) continue;
							const float w = wk[t];
							for (int c = 0; c < outCh_; c++) {
								const size_t off = (size_t(yy) * ow + xx) * vep + (c % vep);
								const float pv = (esV == 4)
									? ((const float *)live.channel(c / vep))[off]
									: float(((const __fp16 *)live.channel(c / vep))[off]);
								kpn[c] += w * pv;
								gsum += pv;
							}
						}
					}
					gv = selfGuide ? gsum * (0.25f / 9.0f)
						       : (gr32 ? gr32[x] : float(gr16[x]));
				}
				if (gv < 0.0f) gv = 0.0f;
				if (gv > 1.0f) gv = 1.0f;

				/* --- bilateral slice: two adjacent luma bins --- */
				float z = gv * float(bins_ - 1);
				int zi = int(z);
				if (zi < 0) zi = 0;
				if (zi > bins_ - 2) zi = bins_ - 2;
				const float fz = z - float(zi);

				const float *Az = A + taps_ + zi * coeffs_;
				const float *Bz = B + taps_ + zi * coeffs_;
				for (int k = 0; k < coeffs_; k++) {
					const float a = Az[k] + (Bz[k] - Az[k]) * fx;
					const float b = Az[coeffs_ + k] + (Bz[coeffs_ + k] - Az[coeffs_ + k]) * fx;
					c20[k] = a + (b - a) * fz;
				}

				/* --- apply the predicted kernel --- */
#if defined(__aarch64__)
				if (packed && esV == 2 && outCh_ == 4) {
					/*
					 * elempack 4 puts R/G1/G2/B of one pixel in 8
					 * contiguous bytes, so a tap is one vld1_f16 and one
					 * fmla-by-scalar -- 9 of each per pixel, versus the
					 * 36 scalar half-to-float converts of the generic path.
					 */
					float32x4_t acc2 = vdupq_n_f32(0.0f);
					int t = 0;
					for (int r = 0; r < kside_; r++) {
						const __fp16 *rp = lrow16[r];
						if (!rp) { t += kside_; continue; }
						for (int dx = -pad; dx <= pad; dx++, t++) {
							const int xx = x + dx;
							if (xx < 0 || xx >= ow) continue;
							const float16x4_t h = vld1_f16((const __fp16 *)(rp + size_t(xx) * 4));
							acc2 = vfmaq_n_f32(acc2, vcvt_f32_f16(h), wk[t]);
						}
					}
					float32x4_t o;
					{
						float r0 = c20[4]  + vaddvq_f32(vmulq_f32(vld1q_f32(c20 + 0),  acc2));
						float r1 = c20[9]  + vaddvq_f32(vmulq_f32(vld1q_f32(c20 + 5),  acc2));
						float r2 = c20[14] + vaddvq_f32(vmulq_f32(vld1q_f32(c20 + 10), acc2));
						float r3 = c20[19] + vaddvq_f32(vmulq_f32(vld1q_f32(c20 + 15), acc2));
						const float tmp[4] = { r0, r1, r2, r3 };
						o = vld1q_f32(tmp);
					}
					if (clip_)
						o = vminq_f32(vmaxq_f32(o, vdupq_n_f32(0.0f)),
							      vdupq_n_f32(1.0f));
					vst1_f16((__fp16 *)(orow16 + size_t(x) * 4), vcvt_f16_f32(o));
					continue;
				}
#endif
				for (int c = 0; c < outCh_; c++)
					kpn[c] = 0.0f;
				if (packed) {
					int t = 0;
					for (int r = 0; r < kside_; r++) {
						if (!lrow32[r] && !lrow16[r]) { t += kside_; continue; }
						for (int dx = -pad; dx <= pad; dx++, t++) {
							const int xx = x + dx;
							if (xx < 0 || xx >= ow) continue;
							const float w = wk[t];
							if (esV == 4) {
								const float *v = lrow32[r] + size_t(xx) * vep;
								for (int c = 0; c < outCh_; c++)
									kpn[c] += w * v[c];
							} else {
								const __fp16 *v = lrow16[r] + size_t(xx) * vep;
								for (int c = 0; c < outCh_; c++)
									kpn[c] += w * float(v[c]);
							}
						}
					}
				} else {
					int t = 0;
					for (int dy = -pad; dy <= pad; dy++) {
						const int yy = y + dy;
						if (yy < 0 || yy >= oh) { t += kside_; continue; }
						for (int dx = -pad; dx <= pad; dx++, t++) {
							const int xx = x + dx;
							if (xx < 0 || xx >= ow) continue;
							const float w = wk[t];
							for (int c = 0; c < outCh_; c++) {
								const size_t off = (size_t(yy) * ow + xx) * vep + (c % vep);
								const float pv = (esV == 4)
									? ((const float *)live.channel(c / vep))[off]
									: float(((const __fp16 *)live.channel(c / vep))[off]);
								kpn[c] += w * pv;
							}
						}
					}
				}

				/* --- 4x5 affine --- */
#if defined(__aarch64__)
				if (fast) {
					const float r0 = c20[4]  + vaddvq_f32(vmulq_f32(vld1q_f32(c20 + 0),  acc));
					const float r1 = c20[9]  + vaddvq_f32(vmulq_f32(vld1q_f32(c20 + 5),  acc));
					const float r2 = c20[14] + vaddvq_f32(vmulq_f32(vld1q_f32(c20 + 10), acc));
					const float r3 = c20[19] + vaddvq_f32(vmulq_f32(vld1q_f32(c20 + 15), acc));
					const float tmp[4] = { r0, r1, r2, r3 };
					float32x4_t o = vld1q_f32(tmp);
					if (clip_)
						o = vminq_f32(vmaxq_f32(o, vdupq_n_f32(0.0f)),
							      vdupq_n_f32(1.0f));
					vst1_f16((__fp16 *)(orow16 + size_t(x) * 4), vcvt_f16_f32(o));
					continue;
				}
#endif
				for (int c = 0; c < outCh_; c++) {
					const float *row = c20 + size_t(c) * (outCh_ + 1);
					float accs = row[outCh_];
					for (int k = 0; k < outCh_; k++)
						accs += row[k] * kpn[k];
					if (clip_) {
						if (accs < 0.0f) accs = 0.0f;
						if (accs > 1.0f) accs = 1.0f;
					}
					if (packed) {
						if (esV == 4) orow32[size_t(x) * vep + c] = accs;
						else orow16[size_t(x) * vep + c] = (__fp16)accs;
					} else {
						const size_t off = (size_t(y) * ow + x) * vep + (c % vep);
						if (esV == 4) ((float *)out.channel(c / vep))[off] = accs;
						else ((__fp16 *)out.channel(c / vep))[off] = (__fp16)accs;
					}
				}
			}
		}

		return 0;
	}


private:
	/* 2^x with a degree-4 minimax poly on the fraction; ~1e-6 relative,
	 * far inside fp16 storage precision and ~10x faster than expf(). */
	static inline float fastExp(float x)
	{
		x *= 1.44269504088896f;
		if (x < -60.0f) return 0.0f;
		float f = std::floor(x);
		float t = x - f;
		float p = 1.0f + t * (0.69314718f + t * (0.24022651f +
			  t * (0.05550411f + t * 0.00961812f)));
		union { float f; int i; } u;
		u.i = (int(f) + 127) << 23;
		return p * u.f;
	}

#if defined(__aarch64__)
	static inline float32x4_t fastExpQ(float32x4_t x)
	{
		x = vmaxq_f32(x, vdupq_n_f32(-80.0f));
		x = vmulq_n_f32(x, 1.44269504088896f);
		const float32x4_t f = vrndmq_f32(x);
		const float32x4_t t = vsubq_f32(x, f);
		float32x4_t p = vmlaq_n_f32(vdupq_n_f32(0.05550411f), t, 0.00961812f);
		p = vmlaq_f32(vdupq_n_f32(0.24022651f), p, t);
		p = vmlaq_f32(vdupq_n_f32(0.69314718f), p, t);
		p = vmlaq_f32(vdupq_n_f32(1.0f), p, t);
		int32x4_t e = vcvtq_s32_f32(f);
		e = vshlq_n_s32(vaddq_s32(e, vdupq_n_s32(127)), 23);
		return vmulq_f32(p, vreinterpretq_f32_s32(e));
	}
#endif

	int taps_ = 9;
	int clip_;
	int bins_ = 8;
	int outCh_ = 4;
	int kside_ = 3;
	int coeffs_ = 20;
};

::ncnn::Layer *KpnAssembleLR_layer_creator(void *userdata);
DEFINE_LAYER_CREATOR(KpnAssembleLR)

} /* namespace RPiController */
