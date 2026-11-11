/* SPDX-License-Identifier: BSD-2-Clause */
/*
 * Copyright (C) 2026, Raspberry Pi Ltd
 *
 * Global translation and rotation from left/right half projections
 */

#include "global_motion.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

using namespace RPiController;

namespace {

using Profile = std::vector<double>;

/*
 * Width of the box blur subtracted from each profile before correlating, to
 * remove shading and large-scale brightness structure.
 */
constexpr int kHighpassWidth = 31;

/*
 * A half's shifts are only used when both of its normalised correlation
 * peaks reach this; below it, the half (featureless, or changed completely)
 * proposes nothing.
 */
constexpr double kMinPeak = 0.3;

/*
 * When the sum of an effective centre's weights falls below this fraction of
 * its ideal value, the weights mostly cancel, and the geometric centre of
 * the half is used instead.
 */
constexpr double kMinWeightRatio = 0.2;

/*
 * The mean of the halves' shifts is a translation candidate when they agree
 * to within this, in working pixels on each axis: they then measure the same
 * motion, and their mean is the better estimate of it.
 */
constexpr double kMeanTolerance = 1.0;

/*
 * The profile minus its box blur. Near the ends the blur averages only the
 * part of the box that lies within the profile, so the ends aren't biased.
 * The effective centres treat this operator as symmetric, which is exact
 * away from the ends and close enough at them.
 */
Profile highpass(const Profile &p)
{
	const int n = p.size(), r = kHighpassWidth / 2;
	Profile sum(n + 1, 0.0);
	for (int i = 0; i < n; i++)
		sum[i + 1] = sum[i] + p[i];

	Profile out(n);
	for (int i = 0; i < n; i++) {
		const int lo = std::max(0, i - r), hi = std::min(n, i + r + 1);
		out[i] = p[i] - (sum[hi] - sum[lo]) / (hi - lo);
	}
	return out;
}

/* Central difference, zero at both ends (the operator D). */
Profile deriv(const Profile &p)
{
	const int n = p.size();
	Profile d(n, 0.0);
	for (int i = 1; i + 1 < n; i++)
		d[i] = 0.5 * (p[i + 1] - p[i - 1]);
	return d;
}

/* D^T applied to g, so that g . (D m) == (D^T g) . m for any m. */
Profile derivAdjoint(const Profile &g)
{
	const int n = g.size();
	Profile h(n, 0.0);
	for (int i = 1; i + 1 < n; i++) {
		h[i + 1] += 0.5 * g[i];
		h[i - 1] -= 0.5 * g[i];
	}
	return h;
}

double dot(const Profile &a, const Profile &b)
{
	double s = 0.0;
	for (size_t i = 0; i < a.size(); i++)
		s += a[i] * b[i];
	return s;
}

struct Shift {
	double shift;
	double peak; /* normalised correlation at the peak, 0 if it isn't one */
};

/*
 * The shift k, to sub-pixel accuracy, for which cur[i] ~= ref[i - k], with k
 * in [-range, range]. Both profiles are high-passed, and the central part of
 * cur is correlated (normalised) against ref at every lag; the peak is
 * refined with a parabolic fit. A best match at either end of the range
 * isn't a real peak, so it has no confidence.
 */
Shift findShift(const Profile &ref, const Profile &cur, int range)
{
	const Profile a = highpass(ref), b = highpass(cur);
	const int m = int(a.size()) - 2 * range;
	if (m <= 1)
		return { 0.0, 0.0 };

	Profile seg(b.begin() + range, b.begin() + range + m);
	const double segMean = std::accumulate(seg.begin(), seg.end(), 0.0) / m;
	for (double &v : seg)
		v -= segMean;
	const double segSq = dot(seg, seg);

	Profile ncc(2 * range + 1);
	for (int k = -range; k <= range; k++) {
		const double *win = a.data() + range - k;
		double winSum = 0.0;
		for (int j = 0; j < m; j++)
			winSum += win[j];
		const double winMean = winSum / m;
		double cross = 0.0, winSq = 0.0;
		for (int j = 0; j < m; j++) {
			const double v = win[j] - winMean;
			cross += v * seg[j];
			winSq += v * v;
		}
		ncc[k + range] = cross / (std::sqrt(winSq * segSq) + 1e-12);
	}

	const int i = std::max_element(ncc.begin(), ncc.end()) - ncc.begin();
	if (i == 0 || i == 2 * range)
		return { double(i - range), 0.0 };

	double frac = 0.0;
	const double y0 = ncc[i - 1], y1 = ncc[i], y2 = ncc[i + 1];
	const double curv = y0 - 2.0 * y1 + y2;
	if (curv < 0.0)
		frac = 0.5 * (y0 - y2) / curv;
	return { i - range + frac, y1 };
}

struct Centre {
	double along, across;
	bool ok;
};

/*
 * The effective centre of one projection measurement, from the image M (the
 * mean of the current frame and the aligned previous one) summed across the
 * profile, prof, and its first moment across it, moment (the sum of M times
 * the across coordinate). along holds the coordinates along the profile,
 * and geometric the half's geometric centre, (along, across).
 *
 * Linearising the correlation of high-passed (L) profiles gives per-pixel
 * weights w(a, c) = g(a) dM/da(a, c), with g = L^T s, s = D(L prof), and the
 * measured shift is the motion field at the weighted centroid of w. The sums
 * reduce to 1D work:
 *   sum w          = g . D prof
 *   sum w along    = (g along) . D prof
 *   sum w across   = (D^T g) . moment
 */
Centre effectiveCentre(const Profile &prof, const Profile &moment, const Profile &along,
		       const Centre &geometric)
{
	const Profile s = deriv(highpass(prof));
	const Profile g = highpass(s);
	const Profile dprof = deriv(prof);

	const double sumW = dot(g, dprof);
	/* Ideally sumW ~= s . s; much less, and the weights mostly cancel. */
	if (sumW < kMinWeightRatio * dot(s, s))
		return geometric;

	double alongW = 0.0;
	for (size_t i = 0; i < g.size(); i++)
		alongW += g[i] * along[i] * dprof[i];
	return { alongW / sumW, dot(derivAdjoint(g), moment) / sumW, true };
}

/*
 * The row sums over columns [x0, x0 + n) of an image, and its column sums
 * there. Each sum has at most a few hundred terms, so float is plenty.
 */
void sumRowsCols(const float *img, unsigned int h, unsigned int stride, unsigned int x0,
		 unsigned int n, Profile &row, Profile &col)
{
	std::vector<float> colAcc(n, 0.0f);
	float *ca = colAcc.data();
	row.resize(h);
	for (unsigned int y = 0; y < h; y++) {
		const float *r = img + size_t(y) * stride + x0;
		unsigned int x = 0;
		float sum = 0.0f;
#if defined(__ARM_NEON)
		float32x4_t acc = vdupq_n_f32(0.0f);
		for (; x + 4 <= n; x += 4) {
			const float32x4_t v = vld1q_f32(r + x);
			acc = vaddq_f32(acc, v);
			vst1q_f32(ca + x, vaddq_f32(vld1q_f32(ca + x), v));
		}
		sum = vaddvq_f32(acc);
#endif
		for (; x < n; x++) {
			sum += r[x];
			ca[x] += r[x];
		}
		row[y] = sum;
	}
	col.assign(colAcc.begin(), colAcc.end());
}

/*
 * The sums and first moments of M = (cur + prev moved by (ix, iy)) / 2 over
 * columns [x0, x0 + n): rowSum[y] = sum_x M, rowMoment[y] = sum_x M xs[x],
 * colSum[x] = sum_y M and colMoment[x] = sum_y M ys[y], with x counted from
 * x0. Row y reads prev row y - iy and column x reads prev column
 * x0 + x - ix, each clamped to the frame, so that its edges repeat. Columns
 * [lo, hi) need no clamping, so prev is a contiguous run there.
 */
void sumMean(const float *cur, const float *prev, unsigned int w, unsigned int h,
	     unsigned int stride, unsigned int x0, unsigned int n, int ix, int iy, const float *xs,
	     const float *ys, Profile &rowSum, Profile &rowMoment, Profile &colSum,
	     Profile &colMoment)
{
	std::vector<float> colS(n, 0.0f), colM(n, 0.0f);
	float *cs = colS.data(), *cm = colM.data();
	const int off = int(x0) - ix; /* column x of the half reads prev column off + x */
	const unsigned int lo = std::clamp(-off, 0, int(n));
	const unsigned int hi = std::clamp(int(w) - off, int(lo), int(n));
	rowSum.resize(h);
	rowMoment.resize(h);

	for (unsigned int y = 0; y < h; y++) {
		const float *c = cur + size_t(y) * stride + x0;
		const float *p = prev + size_t(std::clamp(int(y) - iy, 0, int(h) - 1)) * stride;
		const float yv = ys[y];
		float rs = 0.0f, rm = 0.0f;
		auto add = [&](unsigned int x, float pv) {
			const float m = 0.5f * (c[x] + pv);
			rs += m;
			rm += m * xs[x];
			cs[x] += m;
			cm[x] += m * yv;
		};

		for (unsigned int x = 0; x < lo; x++)
			add(x, p[0]);
		unsigned int x = lo;
#if defined(__ARM_NEON)
		const float *pa = p + off + int(lo);
		const float32x4_t vHalf = vdupq_n_f32(0.5f), vY = vdupq_n_f32(yv);
		float32x4_t vrs = vdupq_n_f32(0.0f), vrm = vdupq_n_f32(0.0f);
		for (; x + 4 <= hi; x += 4, pa += 4) {
			const float32x4_t m = vmulq_f32(vaddq_f32(vld1q_f32(c + x), vld1q_f32(pa)), vHalf);
			vrs = vaddq_f32(vrs, m);
			vrm = vfmaq_f32(vrm, m, vld1q_f32(xs + x));
			vst1q_f32(cs + x, vaddq_f32(vld1q_f32(cs + x), m));
			vst1q_f32(cm + x, vfmaq_f32(vld1q_f32(cm + x), m, vY));
		}
		rs += vaddvq_f32(vrs);
		rm += vaddvq_f32(vrm);
#endif
		for (; x < hi; x++)
			add(x, p[off + int(x)]);
		for (; x < n; x++)
			add(x, p[w - 1]);

		rowSum[y] = rs;
		rowMoment[y] = rm;
	}
	colSum.assign(colS.begin(), colS.end());
	colMoment.assign(colM.begin(), colM.end());
}

/*
 * Measure one half, columns [x0, x1): its shift from the row and column sum
 * profiles, then (if estimating a rotation) the effective centres of both
 * measurements.
 */
GlobalMotionHalf measureHalf(const float *cur, const float *prev, unsigned int w, unsigned int h,
			     unsigned int stride, unsigned int x0, unsigned int x1,
			     const GlobalMotionConfig &config)
{
	const unsigned int hw = x1 - x0;
	const double cx = (w - 1) / 2.0, cy = (h - 1) / 2.0;
	GlobalMotionHalf half;

	/* Step 1: the half's translation, from row and column sum profiles. */
	Profile rowCur, rowPrev, colCur, colPrev;
	sumRowsCols(cur, h, stride, x0, hw, rowCur, colCur);
	sumRowsCols(prev, h, stride, x0, hw, rowPrev, colPrev);
	/* An axis with no search range isn't moving: a perfect match at 0. */
	const Shift noShift = { 0.0, 1.0 };
	const Shift sy = config.maxShiftY ? findShift(rowPrev, rowCur, config.maxShiftY) : noShift;
	const Shift sx = config.maxShiftX ? findShift(colPrev, colCur, config.maxShiftX) : noShift;
	half.dx = sx.shift;
	half.dy = sy.shift;
	half.peakX = sx.peak;
	half.peakY = sy.peak;
	if (!config.rotation)
		return half;

	/*
	 * Step 2: the effective centres, on the mean M of the current frame and
	 * the previous one moved by the (rounded) step 1 shift, the previous
	 * frame's edges repeated where that reaches beyond it. Only M's row and
	 * column sums and their first moments across are needed.
	 */
	const int ix = std::clamp(int(std::nearbyint(half.dx)), -int(config.maxShiftX),
				  int(config.maxShiftX));
	const int iy = std::clamp(int(std::nearbyint(half.dy)), -int(config.maxShiftY),
				  int(config.maxShiftY));

	Profile xs(hw), ys(h);
	std::vector<float> xsF(hw), ysF(h);
	for (unsigned int x = 0; x < hw; x++)
		xsF[x] = xs[x] = x0 + x - cx;
	for (unsigned int y = 0; y < h; y++)
		ysF[y] = ys[y] = y - cy;

	Profile rowSum, rowMoment, colSum, colMoment;
	sumMean(cur, prev, w, h, stride, x0, hw, ix, iy, xsF.data(), ysF.data(), rowSum, rowMoment,
		colSum, colMoment);

	const double gx = (x0 + x1 - 1) / 2.0 - cx;
	/* Vertical measurement: along y, across x. */
	const Centre vc = effectiveCentre(rowSum, rowMoment, ys, { 0.0, gx, false });
	/* Horizontal measurement: along x, across y. */
	const Centre hc = effectiveCentre(colSum, colMoment, xs, { gx, 0.0, false });
	half.cvY = vc.along;
	half.cvX = vc.across;
	half.chX = hc.along;
	half.chY = hc.across;
	half.centresOk = vc.ok && hc.ok;
	return half;
}

/* Sampling step of the final check, in working pixels, on both axes. */
constexpr unsigned int kCheckStep = 4;

/*
 * How well the motion (tx, ty, theta) aligns prev with cur: the mean over
 * every kCheckStep-th row and column of | cur(p) - prev(R^T (p - t)) |, each
 * capped at cap, so that a region that no candidate aligns (a moving object)
 * can't dominate. prev is sampled bilinearly, so that motions of a fraction
 * of a working pixel still count, with its edges repeated beyond the frame,
 * as the warp itself does.
 */
double alignmentCost(const float *cur, const float *prev, unsigned int w, unsigned int h,
		     unsigned int stride, double tx, double ty, double theta, float cap)
{
	const double cx = (w - 1) / 2.0, cy = (h - 1) / 2.0;
	const float c = std::cos(theta), s = std::sin(theta);
	const float maxX = w - 1, maxY = h - 1;
	double sum = 0.0;
	unsigned int n = 0;

	for (unsigned int y = 0; y < h; y += kCheckStep) {
		const float *cr = cur + size_t(y) * stride;
		/* Where column 0 of this row reads from; each column adds (c, -s). */
		const float u0 = -cx - tx, v = y - cy - ty;
		const float sx0 = c * u0 + s * v + cx, sy0 = -s * u0 + c * v + cy;
		float rowSum = 0.0f;
		for (unsigned int x = 0; x < w; x += kCheckStep) {
			const float sx = std::clamp(sx0 + c * x, 0.0f, maxX);
			const float sy = std::clamp(sy0 - s * x, 0.0f, maxY);
			const unsigned int x0 = std::min(unsigned(sx), w - 2), y0 = std::min(unsigned(sy), h - 2);
			const float fx = sx - x0, fy = sy - y0;
			const float *p0 = prev + size_t(y0) * stride + x0, *p1 = p0 + stride;
			const float top = p0[0] + fx * (p0[1] - p0[0]);
			const float bottom = p1[0] + fx * (p1[1] - p1[0]);
			const float d = std::abs(cr[x] - (top + fy * (bottom - top)));
			rowSum += std::min(d, cap);
			n++;
		}
		sum += rowSum;
	}
	return sum / n;
}

struct NoMotion {
	float cap; /* the cap on the differences for alignmentCost() */
	double cost; /* alignmentCost() with no motion */
};

/*
 * The differences | cur - prev | over the same samples as alignmentCost(),
 * with no motion. The cap is twice their median: unlike the mean, a moving
 * object covering less than half the frame barely moves that. The cost is
 * then the same as alignmentCost(..., 0, 0, 0, cap) (up to float rounding),
 * without the interpolation.
 */
NoMotion noMotionCost(const float *cur, const float *prev, unsigned int w, unsigned int h,
		      unsigned int stride)
{
	std::vector<float> d;
	d.reserve(((w + kCheckStep - 1) / kCheckStep) * ((h + kCheckStep - 1) / kCheckStep));
	for (unsigned int y = 0; y < h; y += kCheckStep) {
		const float *cr = cur + size_t(y) * stride, *pr = prev + size_t(y) * stride;
		for (unsigned int x = 0; x < w; x += kCheckStep)
			d.push_back(std::abs(cr[x] - pr[x]));
	}
	std::nth_element(d.begin(), d.begin() + d.size() / 2, d.end());
	const float cap = 2.0f * d[d.size() / 2];

	/* nth_element() only reorders d, so this is the same sum. */
	double sum = 0.0;
	for (float v : d)
		sum += std::min(v, cap);
	return { cap, sum / d.size() };
}

} /* namespace */

namespace RPiController {

GlobalMotionResult estimateGlobalMotion(const float *cur, const float *prev, unsigned int w,
					unsigned int h, unsigned int stride,
					const GlobalMotionConfig &config)
{
	GlobalMotionResult result;
	const GlobalMotionHalf &L = result.half[0] =
		measureHalf(cur, prev, w, h, stride, 0, w / 2, config);
	const GlobalMotionHalf &R = result.half[1] =
		measureHalf(cur, prev, w, h, stride, w / 2, w, config);

	const bool goodL = std::min(L.peakX, L.peakY) >= kMinPeak;
	const bool goodR = std::min(R.peakX, R.peakY) >= kMinPeak;

	/*
	 * Step 3: subtracting L from R eliminates t, leaving a 2x2 linear
	 * system in u = cos(theta), v = sin(theta):
	 *   [ a  -b ] [u]   [dX + a]
	 *   [ c   e ] [v] = [dY + c]
	 * A poor fit shows as sqrt(u^2 + v^2) far from 1, but that's left to the
	 * final check, which rejects a rotation that doesn't align the images.
	 */
	bool rotationCandidate = false;
	if (!config.rotation) {
		/* Not estimating a rotation: nothing to say why it isn't a candidate. */
	} else if (!goodL || !goodR) {
		result.reason = "correlation peak too low";
	} else {
		const double dX = R.dx - L.dx, dY = R.dy - L.dy;
		const double a = R.chX - L.chX, b = R.chY - L.chY;
		const double e = R.cvX - L.cvX, c = R.cvY - L.cvY;
		const double det = a * e + b * c;
		if (std::abs(det) <= 1e-6 * double(w) * w) {
			result.reason = "degenerate centres";
		} else {
			const double u = ((dX + a) * e + b * (dY + c)) / det;
			const double v = (a * (dY + c) - c * (dX + a)) / det;
			result.scale = std::hypot(u, v);
			result.theta = std::atan2(v, u);
			rotationCandidate = true;
		}
	}

	/*
	 * The candidates: the rotation, if the solve held up, and translations:
	 * the mean of the halves' shifts when they agree, and each usable half's
	 * own shift when they don't (a moving object may have pulled one half's
	 * measurement, leaving the other's as the background's).
	 */
	double rotTx = 0.0, rotTy = 0.0;
	if (rotationCandidate) {
		/* Back-substitute with the exact cos and sin, averaging the halves. */
		const double cosT = std::cos(result.theta), sinT = std::sin(result.theta);
		for (const GlobalMotionHalf &m : result.half) {
			rotTx += 0.5 * (m.dx - (cosT - 1.0) * m.chX + sinT * m.chY);
			rotTy += 0.5 * (m.dy - sinT * m.cvX - (cosT - 1.0) * m.cvY);
		}
	}
	struct Translation {
		double tx, ty;
	};
	Translation translations[3];
	unsigned int nTranslations = 0;
	/*
	 * A candidate that moves no point of the frame by kNegligible or more
	 * would be applied as no motion at all (it's half a Bayer quad), so it
	 * isn't worth checking. On a static scene, that's all of them.
	 */
	constexpr double kNegligible = 0.25;
	const double radius = std::hypot((w - 1) / 2.0, (h - 1) / 2.0);
	auto add = [&](double tx, double ty) {
		if (std::hypot(tx, ty) >= kNegligible)
			translations[nTranslations++] = { tx, ty };
	};
	/* Halves this close give practically the same warp: no need to try each. */
	const bool same = std::abs(L.dx - R.dx) < kNegligible && std::abs(L.dy - R.dy) < kNegligible;
	if (goodL && goodR && std::abs(L.dx - R.dx) <= kMeanTolerance &&
	    std::abs(L.dy - R.dy) <= kMeanTolerance)
		add(0.5 * (L.dx + R.dx), 0.5 * (L.dy + R.dy));
	if (!same && goodL)
		add(L.dx, L.dy);
	if (!same && goodR)
		add(R.dx, R.dy);
	const double theta = result.theta;
	result.theta = 0.0;
	if (rotationCandidate && std::hypot(rotTx, rotTy) + std::abs(theta) * radius < kNegligible)
		rotationCandidate = false;
	if (!rotationCandidate && !nTranslations)
		return result;

	/*
	 * The final check: each candidate must align the images better than the
	 * simpler ones, by the margin. Differences are capped at twice the median
	 * difference with no motion: well above what a good alignment leaves,
	 * so that a moving object is counted by its area, not its contrast.
	 */
	const NoMotion noMotion = noMotionCost(cur, prev, w, h, stride);
	const float cap = noMotion.cap;
	const double keep = 1.0 - config.checkMargin;
	double best = result.costNone = noMotion.cost;

	/* The best translation, which must beat no motion by the margin. */
	for (unsigned int i = 0; i < nTranslations; i++) {
		const Translation &t = translations[i];
		const double cost = alignmentCost(cur, prev, w, h, stride, t.tx, t.ty, 0.0, cap);
		if (result.costTranslation < 0.0 || cost < result.costTranslation) {
			result.costTranslation = cost;
			if (cost < keep * result.costNone) {
				best = cost;
				result.tx = t.tx;
				result.ty = t.ty;
				result.kind = GlobalMotionResult::Kind::Translation;
			}
		}
	}

	if (rotationCandidate) {
		result.costRotation = alignmentCost(cur, prev, w, h, stride, rotTx, rotTy, theta, cap);
		if (result.costRotation < keep * best) {
			result.tx = rotTx;
			result.ty = rotTy;
			result.theta = theta;
			result.kind = GlobalMotionResult::Kind::Rotation;
		}
	}

	return result;
}

} /* namespace RPiController */
