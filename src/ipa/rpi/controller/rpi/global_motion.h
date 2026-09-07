/* SPDX-License-Identifier: BSD-2-Clause */
/*
 * Copyright (C) 2026, Raspberry Pi Ltd
 *
 * Global motion estimation between a Bayer frame and the accumulator.
 */
#pragma once

#include <cstdint>
#include <vector>

namespace RPiController {

/*
 * One translation for the whole frame, found coarse-to-fine.
 *
 * WHY GLOBAL, AND WHY THAT IS ENOUGH HERE
 *
 * TemporalAccum's gate already handles things that move WITHIN a scene: those
 * blocks disagree with history, their weight drops, and they fall back to the
 * current frame. What the gate cannot handle is the whole frame moving, because
 * then every block disagrees at once and the accumulator resets -- temporal
 * denoising switches itself off exactly when the camera is in use. Removing the
 * global component first is what keeps it on.
 *
 * WHY PACKED PIXELS
 *
 * Shifting Bayer data by an odd number of pixels swaps R with G. Every vector
 * here is therefore in PACKED pixels (2 Bayer px), which keeps the CFA phase
 * exact and makes the warp a pure gather with no interpolation. The cost is 2
 * Bayer px of residual misalignment, which the gate then sees and prices in.
 *
 * WHY COARSE-TO-FINE
 *
 * Exhaustive search for +/-64 px at full resolution is ~17k candidates per
 * position. Searching +/-8 on a 1/8 pyramid level covers the same range in 289
 * candidates over 1/64 the pixels, then three +/-1 refinements land it exactly.
 */
class GlobalMotion
{
public:
	struct Result {
		int dx = 0;      /* packed pixels; accumulator -> current frame */
		int dy = 0;
		float cost = 0;  /* mean abs difference at the winning offset */
		float costZero = 0; /* ...and at (0,0), so callers can tell if it helped */
		/*
		 * Fraction of tiles that INDIVIDUALLY prefer the winning offset
		 * to (0,0). A whole-frame MAD is a sum, so one large or
		 * high-contrast moving object can carry the total on its own and
		 * win a shift that mis-registers the static background it
		 * outvoted -- which looks like the entire frame lurching when a
		 * single thing moves. A per-tile vote cannot be bought that way:
		 * real camera motion moves every tile, an object moves few.
		 */
		float agree = 0;
	};

	/*
	 * `ref` is the accumulator in Bayer layout (uint16 DN, stride = bayer
	 * width); `cur` is the incoming Bayer frame.
	 */
	/*
	 * `ref` is the accumulator, which is stored as uint16 DN (see AccT in
	 * temporal_accum.h) -- it used to be float. Only the level-0 pyramid
	 * build reads it, and it converts to float there as it always did.
	 */
	Result estimate(uint16_t const *cur, unsigned stridePx, uint16_t const *ref,
			unsigned bayerW, unsigned bayerH, unsigned threads);

	/* Largest displacement the search can see, in packed pixels. */
	static constexpr int kMaxPacked = 8 << 3;   /* +/-8 at 1/8 scale */

private:
	/* Level 0 is packed resolution (Bayer/2); each level halves again. */
	static constexpr unsigned kLevels = 4;
	std::vector<float> cur_[kLevels], ref_[kLevels];
	unsigned lw_[kLevels] = { 0 }, lh_[kLevels] = { 0 };

	void buildPyramids(uint16_t const *cur, unsigned stridePx, uint16_t const *ref,
			   unsigned bayerW, unsigned bayerH, unsigned threads);
};

} /* namespace RPiController */
