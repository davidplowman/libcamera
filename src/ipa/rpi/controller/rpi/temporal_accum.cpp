/* SPDX-License-Identifier: BSD-2-Clause */
/*
 * Copyright (C) 2026, Raspberry Pi Ltd
 *
 * Recursive temporal accumulation with a noise-model-gated per-pixel weight.
 */
#include "temporal_accum.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <thread>

#if defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>
#endif

using namespace RPiController;


TemporalAccum::~TemporalAccum()
{
	{
		std::lock_guard<std::mutex> lk(mu_);
		stop_ = true;
	}
	cvWork_.notify_all();
	for (auto &t : pool_)
		if (t.joinable())
			t.join();
}

void TemporalAccum::startPool(unsigned threads)
{
	if (nThreads_ == threads)
		return;
	{
		std::lock_guard<std::mutex> lk(mu_);
		stop_ = true;
	}
	cvWork_.notify_all();
	for (auto &t : pool_)
		if (t.joinable())
			t.join();
	pool_.clear();
	stop_ = false;
	nThreads_ = threads;
	if (threads <= 1)
		return;

	for (unsigned i = 1; i < threads; ++i)
		pool_.emplace_back([this, i] {
			unsigned seen = 0;
			for (;;) {
				std::unique_lock<std::mutex> lk(mu_);
				cvWork_.wait(lk, [&] { return stop_ || jobGen_ != seen; });
				if (stop_)
					return;
				seen = jobGen_;
				auto fn = job_;
				const unsigned n = jobN_, nt = nThreads_;
				lk.unlock();
				fn(n * i / nt, n * (i + 1) / nt);
				lk.lock();
				if (++jobDone_ == nt - 1)
					cvDone_.notify_one();
			}
		});
}

/*
 * Split [0,n) across the pool. Thread 0's share runs on the caller, so a
 * single-threaded config costs nothing and the caller is never idle.
 */
void TemporalAccum::parallelFor(unsigned n, std::function<void(unsigned, unsigned)> const &fn,
				unsigned threads)
{
	const unsigned nt = std::clamp(threads, 1u, 8u);
	startPool(nt);
	if (nt <= 1 || !n) {
		fn(0, n);
		return;
	}
	{
		std::lock_guard<std::mutex> lk(mu_);
		job_ = fn;
		jobN_ = n;
		jobDone_ = 0;
		++jobGen_;
	}
	cvWork_.notify_all();
	fn(0, n / nt);
	std::unique_lock<std::mutex> lk(mu_);
	cvDone_.wait(lk, [&] { return jobDone_ == nt - 1; });
}

void TemporalAccum::setNoise(float const a[4], float const b[4])
{
	for (unsigned c = 0; c < 4; ++c) {
		a_[c] = a[c];
		b_[c] = b[c];
	}
	lutBlack_ = -1.0f; /* force a rebuild */
}

void TemporalAccum::setParams(float trust, float kZ)
{
	/* Capped below 1: at exactly 1 the accumulator never updates again and
	 * the picture freezes. */
	trust_ = std::clamp(trust, 0.0f, 0.98f);
	kZ_ = std::max(kZ, 1.0f);
}

void TemporalAccum::buildLut(float black, float white)
{
	const float span = std::max(white - black, 1.0f);
	lut_.resize(4 * kLut);
	for (unsigned c = 0; c < 4; ++c)
		for (unsigned i = 0; i < kLut; ++i) {
			float s = float(i) / float(kLut - 1);   /* normalised signal */
			float var = std::max(a_[c] * s + b_[c], 1e-12f);
			lut_[c * kLut + i] = std::sqrt(var) * span * sigma_scale_;
		}
	lutBlack_ = black;
	lutWhite_ = white;
	lutScale_ = float(kLut - 1) / span;
}

/*
 * Shift the accumulator by a whole number of PACKED pixels, i.e. an even
 * number of Bayer pixels, so R stays on R and G on G. That makes this a pure
 * gather with no interpolation and no CFA repair.
 *
 * Pixels that shift in from outside have no history at all. They are seeded
 * from the current frame, which is what "no history" means: the blend then
 * returns the current frame there rather than a stale or zero value.
 */
void TemporalAccum::warpAccum(uint16_t const *bayer, unsigned stridePx, int dx, int dy)
{
	warp_.resize(accum_.size());
	nwarp_.assign(accum_.size(), __fp16(1.0f));
	const int sx = 2 * dx, sy = 2 * dy;   /* in Bayer pixels */
	for (unsigned y = 0; y < h_; ++y) {
		AccT *dst = warp_.data() + size_t(y) * w_;
		const int srcY = int(y) + sy;
		const uint16_t *cur = bayer + size_t(y) * stridePx;
		if (srcY < 0 || srcY >= int(h_)) {
			__fp16 *dn0 = nwarp_.data() + size_t(y) * w_;
			for (unsigned x = 0; x < w_; ++x) {
				dst[x] = cur[x];   /* both uint16 DN */
				dn0[x] = 1.0f;
			}
			continue;
		}
		const AccT *src = accum_.data() + size_t(srcY) * w_;
		const __fp16 *sn = n_.data() + size_t(srcY) * w_;
		__fp16 *dn = nwarp_.data() + size_t(y) * w_;
		for (unsigned x = 0; x < w_; ++x) {
			const int srcX = int(x) + sx;
			const bool oob = (srcX < 0 || srcX >= int(w_));
			dst[x] = oob ? cur[x] : src[srcX];
			/* history scrolled in from outside the frame has none */
			dn[x] = oob ? 1.0f : sn[srcX];
		}
	}
	accum_.swap(warp_);
	n_.swap(nwarp_);
}

float TemporalAccum::apply(uint16_t *bayer, unsigned stridePx, unsigned bayerW,
			   unsigned bayerH, float black, float white, unsigned threads,
			   bool writeBack, unsigned writeCols)
{
	if (!bayer || !bayerW || !bayerH)
		return 0.0f;

	if (w_ != bayerW || h_ != bayerH) {
		w_ = bayerW;
		h_ = bayerH;
		accum_.assign(size_t(w_) * h_, AccT(0));
		n_.assign(size_t(w_) * h_, __fp16(1.0f));
		primed_ = false;
	}
	if (black != lutBlack_ || white != lutWhite_)
		buildLut(black, white);

	/* First frame after a reset: history IS this frame, nothing to blend. */
	if (!primed_) {
		for (unsigned y = 0; y < h_; ++y) {
			const uint16_t *src = bayer + size_t(y) * stridePx;
			AccT *dst = accum_.data() + size_t(y) * w_;
			/* same type and same units -- a straight copy now */
			std::memcpy(dst, src, size_t(w_) * sizeof(AccT));
		}
		if (n_.size() != accum_.size())
			n_.assign(accum_.size(), __fp16(1.0f));
		else
			std::fill(n_.begin(), n_.end(), __fp16(1.0f));
		primed_ = true;
		/*
		 * Zero, not one. This is "fraction of blocks gated", and on a
		 * priming frame no block was evaluated -- there was no history
		 * to disagree with. Returning 1.0 reads to the caller as "the
		 * whole frame moved", which trips its scene-cut reset, which
		 * clears primed_ again: the accumulator primes, is reset, and
		 * primes again forever, never accumulating anything. Observed
		 * on the camera as "temporal reset: 100% of blocks gated" on
		 * 87 of 87 frames.
		 */
		return 0.0f;
	}

	/*
	 * Align history to the incoming frame BEFORE anything compares them.
	 * Everything downstream -- the gate, the blend -- assumes the two are
	 * registered; a global shift left in place makes every block look like
	 * motion at once.
	 */
	lastDx_ = lastDy_ = 0;
	if (motion_enable_) {
		GlobalMotion::Result m = motion_.estimate(bayer, stridePx, accum_.data(),
							  w_, h_, threads);
		/*
		 * Only accept a shift that actually explains the frame better.
		 * On a featureless or very noisy frame the minimum wanders, and
		 * warping by a vector that did not help costs alignment for
		 * nothing.
		 */
		/*
		 * Two independent conditions, because the cost ratio alone is
		 * buyable. A whole-frame MAD is a sum, so one large or
		 * high-contrast moving object contributes enough of the total
		 * that shifting the frame to TRACK IT lowers the sum -- while
		 * mis-registering the static background that makes up the rest.
		 * The visible result is the whole picture lurching whenever a
		 * single object moves. The tile vote closes that: real camera
		 * motion moves every tile and scores near 1.0, an object moves
		 * few and scores low, and no amount of contrast changes the
		 * count. A fast lighting change is rejected for the same reason
		 * -- it inflates costZero everywhere but no shift explains it,
		 * so the tiles do not agree.
		 */
		if ((m.dx || m.dy) && m.cost < motion_ratio_ * m.costZero &&
		    m.agree >= motion_agree_) {
			lastDx_ = m.dx;
			lastDy_ = m.dy;
			warpAccum(bayer, stridePx, m.dx, m.dy);
		}
	}

	/*
	 * WHY THE GATE IS PER BLOCK AND NOT PER PIXEL
	 *
	 * A per-pixel test |x-y| vs k*sigma looks right and is actively
	 * harmful: it lowers the weight on history exactly when the new sample
	 * is a noise outlier, so the outliers are the samples it admits most.
	 * Measured, that capped a trust=0.9 accumulator at 1.36x noise
	 * reduction against a theoretical 4.36x -- most of the averaging thrown
	 * away, and it still missed a whole-frame 5 px shift.
	 *
	 * Motion is spatially coherent and noise is not, so the decision is
	 * made per block and the per-pixel noise draw never touches it.
	 * E|N(0,sigma)| = sigma*sqrt(2/pi), so a block of pure noise has a
	 * known expected mean absolute difference; subtracting it leaves zero
	 * in static regions (full trust, full averaging) and grows only where
	 * something actually moved.
	 *
	 * The excess is measured in STANDARD ERRORS of that block mean, not in
	 * sigma. sd|N(0,sigma)| = sigma*sqrt(1-2/pi), so over n pixels the mean
	 * has standard error 0.6028*sigma/sqrt(n) -- 0.038*sigma at n=256. A
	 * threshold quoted in sigma is therefore ~80 standard errors, which is
	 * why an earlier k=3 version sat blind through a 3x brightness jump.
	 * In standard errors the same block is both quiet under noise and
	 * violently loud under real change.
	 */
	const unsigned gw = (w_ + kBlock - 1) / kBlock;
	const unsigned gh = (h_ + kBlock - 1) / kBlock;
	std::vector<float> bw(size_t(gw) * gh, trust_);
	/*
	 * The field the BLEND reads, which is bw min-dilated by one block. The
	 * gate decides per 16 px block but the weight is applied bilinearly
	 * (see blendBand), so a gated block's zero only lands exactly at the
	 * block CENTRE: next to an ungated block at trust_ the shared edge
	 * interpolates to trust_/2. That leaves a +-8 px ring around every
	 * moving object blending at ~0.4 instead of 0, which writes object
	 * content into memory; once the object moves on the pixel looks static,
	 * nothing gates it, and it decays at the ungated rate -- ~11 frames to
	 * 1%, about 580 ms of visible trail at 53 ms/frame. The tell is that
	 * the measured ghost peaks 4-8 px behind a trailing edge and collapses
	 * at 9 px, i.e. exactly half a block.
	 *
	 * Dilating by one block makes a gated block zero the whole region its
	 * weight can reach. Kept SEPARATE from bw so the gated-block count
	 * returned below still reflects what the gate actually decided: that
	 * count drives the wholesale reset at accum_reset_frac, and inflating
	 * it with the dilated area would fire spurious full resets.
	 */
	std::vector<float> bwBlend;
	const bool nAvail = ramp_ && n_.size() == accum_.size();

	/* relaxed: these are diagnostics, exactness across threads is not needed */
	std::atomic<double> sumDiff{0.0}, sumSigma{0.0};
	std::atomic<uint64_t> nBlk{0};
	auto statBand = [&](unsigned by0, unsigned by1) {
		for (unsigned by = by0; by < by1; ++by) {
			for (unsigned bx = 0; bx < gw; ++bx) {
				double sd = 0.0, ss = 0.0;
				unsigned n = 0;
				const unsigned y1 = std::min((by + 1) * kBlock, h_);
				const unsigned x1 = std::min((bx + 1) * kBlock, w_);
				/*
				 * Sample a quarter of the block: rows 0,1,4,5,...
				 * and the same in x, so all four CFA channels stay
				 * represented while a quarter of the memory is
				 * touched. The z-score divides by the standard
				 * error computed from the ACTUAL n, so the
				 * statistic stays correctly normalised -- only its
				 * precision drops, which kZ absorbs (kZ 20 at
				 * n=256 and kZ 10 at n=64 are the same threshold
				 * in sigma). Worth ~4 ms a frame on the Pi.
				 */
				double sn = 0.0;
				for (unsigned y = by * kBlock; y < y1; y += (y & 1u) ? 3u : 1u) {
					const uint16_t *src = bayer + size_t(y) * stridePx;
					const AccT *acc = accum_.data() + size_t(y) * w_;
					const __fp16 *nrow = nAvail ? n_.data() + size_t(y) * w_ : nullptr;
					const unsigned cRow = (y & 1u) << 1;
					for (unsigned x = bx * kBlock; x < x1; x += (x & 1u) ? 3u : 1u) {
						int li = int((float(src[x]) - black) * lutScale_);
						li = std::clamp(li, 0, int(kLut) - 1);
						ss += lut_[(cRow | (x & 1u)) * kLut + unsigned(li)];
						sd += std::fabs(float(src[x]) - float(acc[x]));
						if (nrow)
							sn += nrow[x];
						++n;
					}
				}
				if (!n)
					continue;
				const float mean = float(sd / n);
				/*
				 * |new - acc| is the difference of TWO noisy values, and
				 * acc has been averaged only n_eff times, so its sd is
				 * sigma*sqrt(1 + 1/n_eff) -- not sigma.
				 *
				 * Ignoring that term is self-reinforcing and was costing
				 * almost all of the temporal denoising: right after a
				 * prime n_eff is 1, the true mean |difference| is
				 * sqrt(2)*0.7979*sigma, the gate reads the excess as
				 * motion and crushes the weight to ~0.23*trust, so the
				 * accumulator stays noisy, so n_eff stays at 1, so the
				 * gate crushes it again. Measured: the accumulator gained
				 * 0.75 dB over a single frame after 40 static frames,
				 * where depth 19 should be worth ~12 dB.
				 */
				const float nEff = nAvail ? std::max(float(sn / n), 1.0f) : 1e6f;
				const float infl = std::sqrt(1.0f + 1.0f / nEff);
				const float sigma = std::max(float(ss / n), 1e-6f) * infl;
				/* sqrt(2/pi) -- the mean |deviation| of pure noise */
				const float expect = 0.7978845608f * sigma;
				/* sqrt(1-2/pi) -- its standard deviation */
				const float se = std::max(0.6028102749f * sigma /
							  std::sqrt(float(n)), 1e-6f);
				const float r = std::max(mean - expect, 0.0f) / (kZ_ * se);
				sumDiff.fetch_add(double(mean), std::memory_order_relaxed);
				sumSigma.fetch_add(double(sigma), std::memory_order_relaxed);
				nBlk.fetch_add(1, std::memory_order_relaxed);
				bw[size_t(by) * gw + bx] =
					r >= 1.0f ? 0.0f : trust_ * (1.0f - r * r);
			}
		}
	};

	/*
	 * Blend the block weights rather than stepping between them: a hard
	 * block edge in the weight is a hard edge in how much noise survives,
	 * which shows up as a visible 16 px grid.
	 */
	/*
	 * n saturates where the running mean (n-1)/n first reaches trust_, i.e.
	 * n = 1/(1-trust). Steady state is then exactly the old fixed-w filter, so
	 * `accum_trust` still means the depth it always meant -- only the approach
	 * to it changes.
	 */
	const float span = std::max(white - black, 1.0f);
	const float nMax = (trust_ < 1.0f) ? 1.0f / (1.0f - trust_) : 1e6f;
	const float invTrust = (trust_ > 1e-6f) ? 1.0f / trust_ : 0.0f;
	const bool ramp = ramp_ && !n_.empty();

	auto blendBand = [&](unsigned y0, unsigned y1) {
		/*
		 * The weight field is bilinear over a 16 px block grid, so within
		 * one block x0i is CONSTANT and the weight along the row is just a
		 * straight line. The previous version rediscovered that per pixel:
		 * a divide, a floor, two clamps and four gathers from bw[] for
		 * every one of 2.1M pixels, to evaluate a function with ~121
		 * distinct breakpoints. Here the vertical lerp collapses to one
		 * gw-length row, and each block walks a linear ramp -- no floor,
		 * no clamp, no gather in the inner loop.
		 */
		std::vector<float> brow(gw);
		/* writeBack region, so a run never straddles the boundary and the
		 * vector store is uniformly on or off inside a run. */
		const unsigned wlim = writeBack
			? (writeCols ? std::min<unsigned>(writeCols, w_) : w_)
			: 0u;

		for (unsigned y = y0; y < y1; ++y) {
			uint16_t *src = bayer + size_t(y) * stridePx;
			AccT *acc = accum_.data() + size_t(y) * w_;
			__fp16 *np = ramp ? n_.data() + size_t(y) * w_ : nullptr;
			const unsigned cRowB = (y & 1u) << 1;
			const float fy = (float(y) + 0.5f) / kBlock - 0.5f;
			const int y0i = int(std::floor(fy));
			const float ty = fy - float(y0i);
			const unsigned ya = unsigned(std::clamp(y0i, 0, int(gh) - 1));
			const unsigned yb = unsigned(std::clamp(y0i + 1, 0, int(gh) - 1));
			const float *ra = bwBlend.data() + size_t(ya) * gw;
			const float *rb = bwBlend.data() + size_t(yb) * gw;
			for (unsigned b = 0; b < gw; ++b)
				brow[b] = ra[b] + (rb[b] - ra[b]) * ty;

			unsigned x = 0;
			while (x < w_) {
				const float fx = (float(x) + 0.5f) / kBlock - 0.5f;
				const int x0i = int(std::floor(fx));
				const unsigned xa = unsigned(std::clamp(x0i, 0, int(gw) - 1));
				const unsigned xb = unsigned(std::clamp(x0i + 1, 0, int(gw) - 1));
				unsigned xe = unsigned(std::max(0,
					(x0i + 1) * int(kBlock) + int(kBlock) / 2));
				xe = std::min(xe <= x ? x + 1 : xe, w_);
				if (x < wlim && xe > wlim)
					xe = wlim;
				const bool wb = (x < wlim);

				const float w0 = brow[xa], w1 = brow[xb];
				const float step = (w1 - w0) / float(kBlock);
				float wgt = w0 + (fx - float(x0i)) * (w1 - w0);

				/*
				 * Per-pixel gate. Runs scalar: it needs this
				 * pixel's own sigma, and the NEON path below
				 * assumes a weight that varies only along the
				 * block ramp. Enabled only when kPix_ > 0.
				 */
				const bool pixGate = (kPix_ > 0.0f);

#if defined(__aarch64__)
				/*
				 * Lane-constant noise model for this row: lanes
				 * x..x+3 alternate between two CFA channels, and
				 * the parity is preserved as x advances by 4.
				 */
				const unsigned cA = cRowB | (x & 1u);
				const unsigned cB = cRowB | ((x + 1) & 1u);
				const float aL[4] = { a_[cA], a_[cB], a_[cA], a_[cB] };
				const float bL[4] = { b_[cA], b_[cB], b_[cA], b_[cB] };
				const float32x4_t vA = vld1q_f32(aL), vB = vld1q_f32(bL);
				const float32x4_t vSpan = vdupq_n_f32(span);
				const float32x4_t vInvSpan = vdupq_n_f32(1.0f / span);
				const float32x4_t vBlack = vdupq_n_f32(black);
				/* with the ramp off there is no per-pixel n, so the
				 * sqrt(1 + 1/n) widening folds into the constant --
				 * matching what the scalar tail does. */
				const float32x4_t vKPix = vdupq_n_f32(
					ramp ? kPix_ : kPix_ * std::sqrt(1.0f + 1.0f / nMax));
				const float32x4_t vTwo = vdupq_n_f32(2.0f);
				const float32x4_t vZero = vdupq_n_f32(0.0f);
				if (true) {
				const float32x4_t vOne = vdupq_n_f32(1.0f);
				const float32x4_t vStep4 = vdupq_n_f32(step * 4.0f);
				float32x4_t vw = { wgt, wgt + step,
						   wgt + 2.0f * step, wgt + 3.0f * step };
				const float32x4_t vNMax = vdupq_n_f32(nMax);
				const float32x4_t vInvT = vdupq_n_f32(invTrust);
				const float32x4_t vTrust = vdupq_n_f32(trust_);
				for (; x + 4 <= xe; x += 4) {
					/* uint16 DN -> fp32, the same widening this
					 * loop already uses for the Bayer source. */
					const float32x4_t va =
						vcvtq_f32_u32(vmovl_u16(vld1_u16(acc + x)));
					const float32x4_t vs =
						vcvtq_f32_u32(vmovl_u16(vld1_u16(src + x)));
					float32x4_t vwp = vw;
					if (pixGate) {
						/* sigma(signal) analytically -- same model
						 * the LUT tabulates, no gather. */
						/* the LUT clamps its index to [0, kLut-1];
						 * match that. 46%% of pixels sit BELOW black
						 * at this gain, and letting s go negative
						 * drives var -> 0, sigma -> 0 and the gate
						 * to reject every one of them. */
						float32x4_t s =
							vmulq_f32(vsubq_f32(vs, vBlack), vInvSpan);
						s = vminq_f32(vmaxq_f32(s, vZero), vOne);
						float32x4_t var = vmlaq_f32(vB, vA, s);
						var = vmaxq_f32(var, vdupq_n_f32(1e-12f));
						float32x4_t sig = vmulq_f32(vsqrtq_f32(var), vSpan);
						/* widen by sqrt(1 + 1/n): acc has its own noise */
						if (ramp) {
							const float32x4_t vn0 = vcvt_f32_f16(vld1_f16(np + x));
							float32x4_t rr = vrecpeq_f32(vn0);
							rr = vmulq_f32(vrecpsq_f32(vn0, rr), rr);
							sig = vmulq_f32(sig, vsqrtq_f32(
								vaddq_f32(vdupq_n_f32(1.0f), rr)));
						}
						const float32x4_t thr =
							vmaxq_f32(vmulq_f32(vKPix, sig),
								  vdupq_n_f32(1e-6f));
						float32x4_t t = vabsq_f32(vsubq_f32(vs, va));
						/* t/thr via Newton reciprocal */
						float32x4_t ri = vrecpeq_f32(thr);
						ri = vmulq_f32(vrecpsq_f32(thr, ri), ri);
						ri = vmulq_f32(vrecpsq_f32(thr, ri), ri);
						t = vmulq_f32(t, ri);
						const float32x4_t g =
							vminq_f32(vmaxq_f32(vsubq_f32(vTwo, t),
									    vZero), vOne);
						vwp = vmulq_f32(vwp, g);
					}
					if (ramp) {
						/* n <- min(n * (w/trust) + 1, nMax) */
						float32x4_t vn = vcvt_f32_f16(vld1_f16(np + x));
						vn = vmlaq_f32(vOne, vn,
							       vmulq_f32(vwp, vInvT));
						vn = vminq_f32(vn, vNMax);
						vst1_f16(np + x, vcvt_f16_f32(vn));
						/* 1 - 1/n by Newton reciprocal: vdivq_f32
						 * is aarch64-only and n is 1..nMax, so
						 * two steps are exact to fp32 here. */
						float32x4_t r = vrecpeq_f32(vn);
						r = vmulq_f32(vrecpsq_f32(vn, r), r);
						r = vmulq_f32(vrecpsq_f32(vn, r), r);
						vwp = vminq_f32(vsubq_f32(vOne, r), vTrust);
					}
					const float32x4_t vo =
						vfmaq_f32(vmulq_f32(vwp, va),
							  vsubq_f32(vOne, vwp), vs);
					/* FCVTAU + saturating narrow: nearest, ties
					 * away from zero, clamped to [0,65535] --
					 * identical to the writeBack store below. */
					vst1_u16(acc + x, vqmovn_u32(vcvtaq_u32_f32(vo)));
					if (wb) {
						/*
						 * FCVTAU: nearest, ties AWAY from
						 * zero -- the same rule as
						 * std::lround -- and it saturates,
						 * which is the clamp to [0,65535].
						 */
						vst1_u16(src + x,
							 vqmovn_u32(vcvtaq_u32_f32(vo)));
					}
					vw = vaddq_f32(vw, vStep4);
				}
				}
				wgt = w0 + ((float(x) + 0.5f) / kBlock - 0.5f
					    - float(x0i)) * (w1 - w0);
#endif
				for (; x < xe; ++x) {
					float wp = wgt;
					if (pixGate) {
						/* same LUT statBand uses: sigma in DN
						 * for this pixel's own level and CFA
						 * channel, so the threshold means the
						 * same thing at every gain. */
						int li = int((float(src[x]) - black) * lutScale_);
						li = std::clamp(li, 0, int(kLut) - 1);
						const float sig = lut_[(cRowB | (x & 1u)) * kLut
								       + unsigned(li)];
						/*
						 * acc carries sigma/sqrt(n) of its own, so
						 * |new - acc| has sd sigma*sqrt(1 + 1/n),
						 * not sigma. Without this the gate is
						 * 1.41x too tight at n=1 and trips on its
						 * own noise right after a reset -- which
						 * resets n, which tightens it again.
						 */
						const float nn = ramp ? float(np[x]) : nMax;
						const float thr = kPix_ * sig *
								  std::sqrt(1.0f + 1.0f / nn);
						const float t = std::fabs(float(src[x]) - float(acc[x])) /
								std::max(thr, 1e-6f);
						wp *= std::clamp(2.0f - t, 0.0f, 1.0f);
					}
					if (ramp) {
						float nn = float(np[x]) * (wp * invTrust);
						nn = std::min(nn + 1.0f, nMax);
						np[x] = __fp16(nn);
						wp = std::min(1.0f - 1.0f / nn, trust_);
					}
					const float out = wp * float(acc[x]) +
							  (1.0f - wp) * float(src[x]);
					/* nearest, ties away, clamped -- the same rule
					 * FCVTAU applies in the NEON body above, so the
					 * vector and scalar tails agree bit for bit. */
					const uint16_t q = uint16_t(std::clamp(
						int(std::lround(out)), 0, 65535));
					acc[x] = q;
					if (wb)
						src[x] = q;
					wgt += step;
				}
			}
			/*
			 * FUSED PACK, one row. See PackSink. acc[] and src[]
			 * are hot here; packMemory/packTile would re-read them
			 * from DRAM. Bit-for-bit the same expression they use:
			 * clamp((v - black) * inv, lo, 1). `inv` already
			 * carries the brightness scale (bnorm_pre_).
			 */
			if (sink_.mem && (y >> 1) >= sink_.cy0 &&
			    (y >> 1) - sink_.cy0 < sink_.h) {
				const unsigned yy = (y >> 1) - sink_.cy0;
				const size_t pl = size_t(sink_.h) * sink_.w;
				const unsigned pb = (y & 1u) ? 2u : 0u;
				__fp16 *m0 = sink_.mem + (pb + 0) * pl + size_t(yy) * sink_.w;
				__fp16 *m1 = sink_.mem + (pb + 1) * pl + size_t(yy) * sink_.w;
				__fp16 *c0 = sink_.cur ? sink_.cur + (pb + 0) * pl
							 + size_t(yy) * sink_.w : nullptr;
				__fp16 *c1 = sink_.cur ? sink_.cur + (pb + 1) * pl
							 + size_t(yy) * sink_.w : nullptr;
				const float bl = sink_.black, iv = sink_.inv, lo = sink_.lo;
				for (unsigned xx = 0; xx < sink_.w; ++xx) {
					const unsigned sx = (sink_.cx0 + xx) * 2u;
					if (sx + 1 >= w_) {
						m0[xx] = m1[xx] = __fp16(0.0f);
						if (c0) c0[xx] = c1[xx] = __fp16(0.0f);
						continue;
					}
					float a0 = (float(acc[sx]) - bl) * iv;
					float a1 = (float(acc[sx + 1]) - bl) * iv;
					m0[xx] = __fp16(a0 < lo ? lo : (a0 > 1.0f ? 1.0f : a0));
					m1[xx] = __fp16(a1 < lo ? lo : (a1 > 1.0f ? 1.0f : a1));
					if (c0) {
						float r0 = (float(src[sx]) - bl) * iv;
						float r1 = (float(src[sx + 1]) - bl) * iv;
						c0[xx] = __fp16(r0 < lo ? lo : (r0 > 1.0f ? 1.0f : r0));
						c1[xx] = __fp16(r1 < lo ? lo : (r1 > 1.0f ? 1.0f : r1));
					}
				}
			}
		}
	};

	/* stats must be complete before any pixel is overwritten, so these are
	 * two separate parallelFor calls rather than one. */
	parallelFor(gh, statBand, threads);

	/*
	 * Counted BEFORE the dilation, so `gated` keeps meaning "blocks the gate
	 * rejected" and the accum_reset_frac threshold behaves as it always did.
	 */
	uint64_t gatedBlocks = 0;
	/* r >= x  <=>  weight <= trust*(1-x^2); x=1/2 and x=1/4 are kZ/2, kZ/4. */
	const float wHalf = trust_ * (1.0f - 0.25f);
	const float wQuarter = trust_ * (1.0f - 0.0625f);
	uint64_t nHalf = 0, nQuarter = 0;
	for (float v : bw) {
		if (v <= 0.0f)
			++gatedBlocks;
		if (v <= wHalf)
			++nHalf;
		if (v <= wQuarter)
			++nQuarter;
	}
	/* sampled every 4th pixel in each direction -- a diagnostic, not a metric */
	if (!n_.empty()) {
		double sn = 0.0; uint64_t cn = 0;
		for (unsigned y = 0; y < h_; y += 4) {
			const __fp16 *r = n_.data() + size_t(y) * w_;
			for (unsigned x = 0; x < w_; x += 4) { sn += float(r[x]); ++cn; }
		}
		last_mean_n_ = cn ? float(sn / double(cn)) : 0.0f;
	}
	const uint64_t nb = nBlk.load();
	last_mean_diff_ = nb ? float(sumDiff.load() / double(nb)) : 0.0f;
	last_mean_sigma_ = nb ? float(sumSigma.load() / double(nb)) : 0.0f;
	const double inv = bw.empty() ? 0.0 : 1.0 / double(bw.size());
	last_g_half_ = float(double(nHalf) * inv);
	last_g_quarter_ = float(double(nQuarter) * inv);

	/* 3x3 minimum over the block grid: ~gw*gh cells, far under 0.1 ms. */
	bwBlend.resize(bw.size());
	for (unsigned by = 0; by < gh; ++by) {
		const unsigned ya = by ? by - 1 : 0;
		const unsigned yb = (by + 1 < gh) ? by + 1 : gh - 1;
		for (unsigned bx = 0; bx < gw; ++bx) {
			const unsigned xa = bx ? bx - 1 : 0;
			const unsigned xb = (bx + 1 < gw) ? bx + 1 : gw - 1;
			float m = bw[size_t(by) * gw + bx];
			for (unsigned yy = ya; yy <= yb; ++yy)
				for (unsigned xx = xa; xx <= xb; ++xx)
					m = std::min(m, bw[size_t(yy) * gw + xx]);
			bwBlend[size_t(by) * gw + bx] = m;
		}
	}

	parallelFor(h_, blendBand, threads);

	return float(double(gatedBlocks) / double(bw.size()));
}
