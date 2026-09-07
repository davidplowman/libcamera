/* SPDX-License-Identifier: BSD-2-Clause */
/*
 * Copyright (C) 2026, Raspberry Pi Ltd
 *
 * Global motion estimation between a Bayer frame and the accumulator.
 */
#include "global_motion.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <thread>

using namespace RPiController;

namespace {

/*
 * Mean absolute difference of `b` shifted by (dx, dy) against `a`, over the
 * overlapping region only. Normalised by the count so offsets that overlap
 * less are not rewarded for it -- an un-normalised SAD is minimised by
 * sliding the images apart until almost nothing overlaps.
 */
float mad(float const *a, float const *b, int w, int h, int dx, int dy, int step)
{
	const int x0 = std::max(0, -dx), x1 = std::min(w, w - dx);
	const int y0 = std::max(0, -dy), y1 = std::min(h, h - dy);
	if (x1 - x0 < 8 || y1 - y0 < 8)
		return std::numeric_limits<float>::max();

	double acc = 0.0;
	long n = 0;
	for (int y = y0; y < y1; y += step) {
		float const *pa = a + size_t(y) * w;
		float const *pb = b + size_t(y + dy) * w + dx;
		for (int x = x0; x < x1; x += step) {
			acc += std::fabs(pa[x] - pb[x]);
			++n;
		}
	}
	return n ? float(acc / n) : std::numeric_limits<float>::max();
}

/*
 * What fraction of 64x64 tiles is better explained by (dx,dy) than by (0,0)?
 * Tiles are scored independently, so the answer does not depend on how much
 * total difference any one region contributes -- which is the whole point.
 */
float agreeFraction(float const *a, float const *b, int w, int h, int dx, int dy)
{
	if (!dx && !dy)
		return 0.0f;
	constexpr int kTile = 64;
	const int x0 = std::max(0, -dx), x1 = std::min(w, w - dx);
	const int y0 = std::max(0, -dy), y1 = std::min(h, h - dy);
	long votes = 0, tiles = 0;
	for (int ty = y0; ty + kTile <= y1; ty += kTile) {
		for (int tx = x0; tx + kTile <= x1; tx += kTile) {
			double s0 = 0.0, sd = 0.0;
			for (int y = ty; y < ty + kTile; y += 2) {
				float const *pa = a + size_t(y) * w;
				float const *p0 = b + size_t(y) * w;
				float const *pd = b + size_t(y + dy) * w + dx;
				for (int x = tx; x < tx + kTile; x += 2) {
					s0 += std::fabs(pa[x] - p0[x]);
					sd += std::fabs(pa[x] - pd[x]);
				}
			}
			++tiles;
			if (sd < s0)
				++votes;
		}
	}
	return tiles ? float(double(votes) / double(tiles)) : 0.0f;
}

} /* namespace */

void GlobalMotion::buildPyramids(uint16_t const *cur, unsigned stridePx,
				 uint16_t const *ref, unsigned bayerW, unsigned bayerH,
				 unsigned threads)
{
	/*
	 * Estimate on a centred window, not the whole frame.
	 *
	 * Threading this build changed nothing on the Pi (25.07 -> 25.74 ms):
	 * it reads the entire Bayer frame plus the entire accumulator and
	 * writes a packed-resolution copy of each, ~17 MB a frame, so it is
	 * bandwidth-bound and more threads add no bandwidth. Reading a quarter
	 * of the area does.
	 *
	 * A global translation is by definition the same everywhere, and a
	 * half-by-half window still leaves ~265k pixels to fit two parameters
	 * from. The loss is that motion is inferred from the centre: if the
	 * centre is featureless, or the only moving thing is at an edge, the
	 * estimate is weaker. The cost check in estimate() rejects the vector
	 * when that shows up as no improvement.
	 */
	const unsigned fw = bayerW / 2, fh = bayerH / 2;
	const unsigned w0 = std::max(fw / 2, 32u), h0 = std::max(fh / 2, 32u);
	const unsigned ox = (fw - w0) / 2, oy = (fh - h0) / 2;
	for (unsigned l = 0; l < kLevels; ++l) {
		lw_[l] = std::max(w0 >> l, 1u);
		lh_[l] = std::max(h0 >> l, 1u);
		cur_[l].resize(size_t(lw_[l]) * lh_[l]);
		ref_[l].resize(size_t(lw_[l]) * lh_[l]);
	}

	/*
	 * Level 0 is the mean of each 2x2 CFA cell -- a grey proxy. Averaging
	 * the cell rather than picking one channel both halves the resolution
	 * (which is what makes the search cheap) and removes the CFA pattern
	 * itself, which would otherwise dominate any difference measure.
	 */
	auto level0 = [&](unsigned y0, unsigned y1) {
	for (unsigned y = y0; y < y1; ++y) {
		uint16_t const *c0 = cur + size_t(2 * (y + oy)) * stridePx + 2 * ox;
		uint16_t const *c1 = c0 + stridePx;
		uint16_t const *r0 = ref + size_t(2 * (y + oy)) * bayerW + 2 * ox;
		uint16_t const *r1 = r0 + bayerW;
		float *dc = cur_[0].data() + size_t(y) * w0;
		float *dr = ref_[0].data() + size_t(y) * w0;
		for (unsigned x = 0; x < w0; ++x) {
			dc[x] = 0.25f * (float(c0[2 * x]) + float(c0[2 * x + 1]) +
					 float(c1[2 * x]) + float(c1[2 * x + 1]));
			dr[x] = 0.25f * (r0[2 * x] + r0[2 * x + 1] +
					 r1[2 * x] + r1[2 * x + 1]);
		}
	}
	};

	/*
	 * This build reads the whole Bayer frame and the whole accumulator and
	 * writes a full packed-resolution copy of each -- ~17 MB of traffic,
	 * and it was the entire reason motion estimation cost 12 ms on the Pi
	 * while the searches themselves are trivial. It is per-row independent,
	 * so it threads for free.
	 */
	const unsigned nt = std::clamp(threads, 1u, 8u);
	if (nt <= 1) {
		level0(0, h0);
	} else {
		std::vector<std::thread> pool;
		for (unsigned i = 0; i < nt; ++i)
			pool.emplace_back(level0, h0 * i / nt, h0 * (i + 1) / nt);
		for (auto &t : pool)
			t.join();
	}

	for (unsigned l = 1; l < kLevels; ++l) {
		const unsigned pw = lw_[l - 1], w = lw_[l], h = lh_[l];
		std::vector<std::thread> pool;
		for (int which = 0; which < 2; ++which) {
			pool.emplace_back([&, which, pw, w, h] {
			std::vector<float> const &src = which ? ref_[l - 1] : cur_[l - 1];
			std::vector<float> &dst = which ? ref_[l] : cur_[l];
			for (unsigned y = 0; y < h; ++y) {
				float const *s0 = src.data() + size_t(2 * y) * pw;
				float const *s1 = s0 + pw;
				float *d = dst.data() + size_t(y) * w;
				for (unsigned x = 0; x < w; ++x)
					d[x] = 0.25f * (s0[2 * x] + s0[2 * x + 1] +
							s1[2 * x] + s1[2 * x + 1]);
			}
			});
		}
		for (auto &t : pool)
			t.join();
	}
}

GlobalMotion::Result GlobalMotion::estimate(uint16_t const *cur, unsigned stridePx,
					    uint16_t const *ref, unsigned bayerW,
					    unsigned bayerH, unsigned threads)
{
	Result r;
	if (!cur || !ref || bayerW < 32 || bayerH < 32)
		return r;

	buildPyramids(cur, stridePx, ref, bayerW, bayerH, threads);

	/*
	 * Coarsest level: exhaustive +/-8. Sub-sampled by 2 because at 1/8
	 * scale neighbouring pixels are already averages of 64 originals, so
	 * every second one carries almost the same information.
	 */
	int bx = 0, by = 0;
	float best = std::numeric_limits<float>::max();
	const int top = int(kLevels) - 1;
	for (int dy = -8; dy <= 8; ++dy)
		for (int dx = -8; dx <= 8; ++dx) {
			float c = mad(cur_[top].data(), ref_[top].data(), int(lw_[top]),
				      int(lh_[top]), dx, dy, 2);
			/*
			 * Mild bias toward the smaller shift. Periodic texture
			 * -- fences, tiling, brickwork -- matches equally well
			 * at every multiple of its period, and without a prior
			 * the search picks among those aliases arbitrarily.
			 * Measured on a 32 px checker it returned dy = 33 for
			 * a true dy of 0. 1% per unit breaks the tie without
			 * being able to overturn a genuinely better match.
			 */
			c *= 1.0f + 0.01f * std::sqrt(float(dx * dx + dy * dy));
			if (c < best) {
				best = c;
				bx = dx;
				by = dy;
			}
		}

	/* Refine +/-1 down the pyramid, doubling the vector at each step. */
	for (int l = top - 1; l >= 0; --l) {
		bx *= 2;
		by *= 2;
		const int step = l >= 1 ? 1 : 2;
		float bc = std::numeric_limits<float>::max();
		int nx = bx, ny = by;
		for (int dy = by - 1; dy <= by + 1; ++dy)
			for (int dx = bx - 1; dx <= bx + 1; ++dx) {
				float c = mad(cur_[l].data(), ref_[l].data(), int(lw_[l]),
					      int(lh_[l]), dx, dy, step);
				if (c < bc) {
					bc = c;
					nx = dx;
					ny = dy;
				}
			}
		bx = nx;
		by = ny;
		best = bc;
	}

	r.dx = bx;
	r.dy = by;
	r.cost = best;
	r.costZero = mad(cur_[0].data(), ref_[0].data(), int(lw_[0]), int(lh_[0]), 0, 0, 2);
	r.agree = agreeFraction(cur_[0].data(), ref_[0].data(), int(lw_[0]), int(lh_[0]),
				bx, by);
	return r;
}
