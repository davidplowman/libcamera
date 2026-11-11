/* SPDX-License-Identifier: BSD-2-Clause */
/*
 * Copyright (C) 2026, Raspberry Pi Ltd
 *
 * Global translation and rotation between two frames, from 1D projection
 * correlations on the left and right halves of the image (see
 * global_motion.md, and global_motion.py, of which this is a port).
 *
 * Both frames are given as "working images": small single-channel images,
 * each pixel the sum (or mean) of a block of Bayer samples, so that the
 * Bayer pattern doesn't matter. Coordinates are in working pixels, measured
 * from the working image's centre ((w - 1) / 2, (h - 1) / 2), with x to the
 * right and y downwards. The motion model is
 *
 *	content at q in the previous frame is at R q + t in the current one,
 *
 *	R = [[cos theta, -sin theta],
 *	     [sin theta,  cos theta]]
 *
 * so a positive theta turns +x towards +y (clockwise on screen). To align
 * the previous frame with the current one, sample it at R^T (p - t) for
 * each output position p.
 *
 * The per-half measurements only propose candidate motions: a rotation
 * (with its translation) and pure translations. The final choice between
 * them and no motion at all is made by checking each against the images
 * themselves: the previous
 * frame is warped by each candidate and compared with the current one, and
 * the one that aligns them best wins. A moving object can then pull the
 * per-half measurements away from the true (static) background without the
 * background being warped to follow it.
 */
#pragma once

namespace RPiController {

struct GlobalMotionConfig {
	/*
	 * Search range of each half's 1D correlations, in working pixels (+/-).
	 * 0 means there's no motion along that axis: it isn't searched, and
	 * counts as a perfect match. Estimating a rotation needs both.
	 */
	unsigned int maxShiftX = 16;
	unsigned int maxShiftY = 16;
	/*
	 * Whether to estimate a rotation. Without it, only the translations
	 * (and no motion) are candidates, and the halves' effective centres,
	 * needed only for the rotation, aren't measured.
	 */
	bool rotation = true;
	/*
	 * In the final check, a translation must align the images better than
	 * no motion, and a rotation better than either, by at least this
	 * fraction of the other's cost, so that noise alone doesn't pick a
	 * warp.
	 */
	double checkMargin = 0.02;
};

/* What one half of the image measured. */
struct GlobalMotionHalf {
	double dx = 0.0, dy = 0.0; /* its shift */
	double peakX = 0.0, peakY = 0.0; /* normalised correlation peaks */
	/* Effective centres of the horizontal (h) and vertical (v) measurements. */
	double chX = 0.0, chY = 0.0;
	double cvX = 0.0, cvY = 0.0;
	bool centresOk = false; /* false: geometric centres were used (or none measured) */
};

struct GlobalMotionResult {
	double tx = 0.0, ty = 0.0; /* translation, working pixels */
	double theta = 0.0; /* rotation, radians */
	enum class Kind {
		None, /* no motion */
		Translation, /* translation only */
		Rotation, /* translation and rotation */
	} kind = Kind::None;
	/*
	 * Why the rotation wasn't a candidate, or "" if it was (or rotation
	 * isn't being estimated).
	 */
	const char *reason = "";
	/*
	 * Each candidate's alignment cost in the final check (mean capped
	 * absolute difference; for translation, the best translation's), or -1
	 * if there was no such candidate.
	 */
	double costNone = -1.0;
	double costTranslation = -1.0;
	double costRotation = -1.0;
	/*
	 * sqrt(cos^2 + sin^2) from the 2x2 solve, 0 if not reached: ideally 1, so
	 * a measure of how well the halves' measurements fit a rotation.
	 */
	double scale = 0.0;
	GlobalMotionHalf half[2]; /* left, right */
};

/*
 * Estimate the motion from prev to cur, both w x h working images, row-major
 * with the given stride (in pixels). w and h must each be more than twice
 * the corresponding maximum shift.
 */
GlobalMotionResult estimateGlobalMotion(const float *cur, const float *prev, unsigned int w,
					unsigned int h, unsigned int stride,
					const GlobalMotionConfig &config);

} /* namespace RPiController */
