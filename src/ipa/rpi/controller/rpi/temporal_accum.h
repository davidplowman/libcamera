/* SPDX-License-Identifier: BSD-2-Clause */
/*
 * Copyright (C) 2026, Raspberry Pi Ltd
 *
 * Recursive temporal accumulation with a noise-model-gated per-pixel weight.
 */
#pragma once

#include "global_motion.h"

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace RPiController {

/*
 * One running accumulator per pixel, blended by how well the new frame agrees
 * with it, in units of the calibrated noise sigma.
 *
 * WHY RECURSIVE RATHER THAN THE 4-FRAME RING
 *
 * model_denoise keeps temporal_=4 packed frames and hands them to the model.
 * Averaging N frames can only reduce noise by sqrt(N), so a 4-frame ring is
 * capped at 2x, and it costs four buffers to get there. One recursive buffer
 * with weight w behaves like averaging (1+w)/(1-w) frames -- w=0.9 is 19
 * frames, w=0.95 is 39 -- for a quarter of the memory.
 *
 * That only pays once fixed-pattern noise is removed, which is what
 * SensorCalib does immediately upstream. Uncorrected, this sensor's averaging
 * saturates near N=21 because 81% of the residual variance at N=128 is
 * sensor-locked and averaging cannot touch it. Corrected, the measured gain is
 * +8.06 dB at N=256 and still climbing.
 *
 * WHY THE GATE IS IN SIGMA UNITS
 *
 * The only question per pixel is whether a disagreement with history is noise
 * or change. That is exactly a comparison against sigma, and sigma is known
 * analytically: var = a*signal + b, fitted per gain per Bayer channel
 * (r^2 >= 0.996). Without it the threshold is a guess that is wrong at one end
 * of the brightness range or the other.
 */
/*
 * ACCUMULATOR STORAGE: uint16 DN, NOT float.
 *
 * accum_ is the single largest array in the denoise pipeline -- one element per
 * Bayer pixel, read AND written every frame. At 1936x1100 that was 8.52 MB of
 * loads plus 8.52 MB of stores per frame as float. The stage is
 * DRAM-BANDWIDTH bound, not compute bound, and the CPU shares that bus with the
 * NPU: measured on this build, `pack` takes 22 ms while the net overlaps it and
 * 13 ms when it runs alone. So halving this array pays twice, exactly as the
 * per-pixel sample count n_ did when it went float -> __fp16.
 *
 * WHY uint16 IS NOT A PRECISION LOSS HERE, AND WHY fp16 WOULD BE.
 * The sensor is 12-bit presented left-shifted by 4 (black is 3200 = 200 << 4),
 * so every incoming sample is a multiple of 16 DN. uint16 storage resolves the
 * recursive mean to 1 DN -- SIXTEEN TIMES FINER than the quantisation of the
 * data going in. fp16 would have been the wrong choice despite being the
 * obvious one: its mantissa gives a step of 4 DN at 4096 and 64 DN at 65535,
 * i.e. it gets WORSE exactly where the signal is largest, and 64 DN is a
 * significant fraction of the residual of a deep average.
 *
 * The blend loop already round-trips through this exact representation: with
 * writeBack on it stores the blended value to the Bayer buffer as uint16 using
 * FCVTAU (nearest, ties away, saturating). Storing the accumulator the same way
 * reuses the proven idiom rather than inventing one.
 */
using AccT = uint16_t;

class TemporalAccum
{
public:
	/* var = a[c]*signal + b[c], signal and var in normalised units. */
	/*
	 * Correction on the modelled sigma, applied where the LUT is built so
	 * the block gate and the per-pixel gate stay consistent.
	 *
	 * MEASURED 2026-08-27 on a static scene at gain 512: mean|cur-acc| came
	 * out 1537 DN against a modelled sigma of 5866, a ratio of 0.262. Pure
	 * noise MUST give sqrt(2/pi) = 0.798, so the model overestimates sigma
	 * by ~3.05x. The consequence is that the gate's threshold sits ~3.8x
	 * above the real noise floor and essentially no motion ever reaches it
	 * -- everything blends into a 9-frame memory, which is the ghosting.
	 * kZ cannot fix this: halving it moved the gated fraction only from
	 * 0.0035 to 0.0056, because the threshold is far from the data either
	 * way.
	 */
	void setSigmaScale(float s) { sigma_scale_ = s; }
	void setNoise(float const a[4], float const b[4]);
	/* kZ is the gate threshold in STANDARD ERRORS of the per-block mean
	 * absolute difference, not in sigma -- see the note in apply(). */
	void setParams(float trust, float kZ);
	/* kPix in sigma; 0 disables the per-pixel gate. */
	void setPixelGate(float kPix) { kPix_ = kPix; }
	/* Per-pixel convergence ramp; false restores the old fixed-w blend. */
	void setRamp(bool on) { ramp_ = on; }
	float lastMeanN() const { return last_mean_n_; }

	/*
	 * FUSED PACK. accum's inner loop already holds every value that
	 * packMemory() and packTile() go back to DRAM for: the post-update
	 * state (which packMemory re-reads, 8.5 MB) and the raw frame (which
	 * packTile re-reads, 4.3 MB). Emitting the packed fp16 planes from
	 * here deletes both re-reads -- ~12.8 MB a frame on a bus that is the
	 * binding constraint (see the note above applyAccumInput: this stage
	 * is bandwidth bound, and overlapping it made BOTH sides slower).
	 *
	 * The emit runs per ROW, after that row's blend runs have finished, so
	 * it is correct regardless of block gating and it reads `acc` and
	 * `src` while they are still in L1. mem and cur are the two 4-plane
	 * feed slots; cur is only valid when writeBack is off, since writeBack
	 * overwrites the raw with the accumulated value.
	 */
	struct PackSink {
		__fp16 *mem = nullptr;    /* slot 0: post-update state */
		__fp16 *cur = nullptr;    /* slot 1: this frame's raw  */
		unsigned w = 0, h = 0;    /* infer_w_, infer_h_        */
		unsigned cx0 = 0, cy0 = 0;
		float black = 0.0f, inv = 0.0f, lo = -0.25f;
	};
	void setPackSink(const PackSink &s) { sink_ = s; }
	void clearPackSink() { sink_ = PackSink(); }

	float lastMeanDiff() const { return last_mean_diff_; }
	float lastMeanSigma() const { return last_mean_sigma_; }
	float lastGHalf() const { return last_g_half_; }
	float lastGQuarter() const { return last_g_quarter_; }
	void setMotion(bool on) { motion_enable_ = on; }
	/* Acceptance for a global shift: it must cut whole-frame cost to at
	 * most `ratio` of the cost at (0,0), AND at least `agree` of tiles must
	 * individually prefer it. The ratio alone was 0.98 -- a 2% improvement
	 * -- which one large moving object buys easily. */
	void setMotionAccept(float ratio, float agree)
	{
		motion_ratio_ = ratio;
		motion_agree_ = agree;
	}
	~TemporalAccum();
	/* Last estimated global shift, in packed pixels. */
	int lastDx() const { return lastDx_; }
	int lastDy() const { return lastDy_; }
	/* Discard history: mode switch, scene cut, geometry change. */
	void reset() { primed_ = false; }
	bool primed() const { return primed_; }

	/*
	 * In-place on full-resolution Bayer. Returns the fraction of pixels
	 * whose weight was driven to zero -- a frame-wide motion measure, which
	 * the caller can threshold to detect a cut.
	 */
	/*
	 * writeCols limits the write-back to the leftmost N Bayer columns, so a
	 * demo split still works with the model disabled: history is still
	 * accumulated across the WHOLE frame (the estimate must not be
	 * half-blind), only the pixels handed downstream are restricted.
	 *
	 * writeCols limits the write-back to the leftmost N Bayer columns, so a
	 * demo split still works with the model disabled: history is still
	 * accumulated across the WHOLE frame (the estimate must not be
	 * half-blind), only the pixels handed downstream are restricted.
	 *
	 * writeBack=false updates the accumulator from the frame but leaves the
	 * Bayer buffer untouched. That is what the memory-input model needs:
	 * the model's own output has already been written there by unpackTile,
	 * and the accumulator must keep averaging RAW frames -- averaging its
	 * own denoised output would feed the model back its own guess.
	 */
	float apply(uint16_t *bayer, unsigned stridePx, unsigned bayerW, unsigned bayerH,
		    float black, float white, unsigned threads, bool writeBack = true,
		    unsigned writeCols = 0);

	/* Accumulator state, Bayer layout, float DN, stride == bayerWidth().
	 * Null until primed. This is the "memory" the model consumes. */
	AccT const *state() const { return primed_ ? accum_.data() : nullptr; }
	unsigned stateWidth() const { return w_; }
	unsigned stateHeight() const { return h_; }

private:
	void buildLut(float black, float white);
	/* Run fn(lo, hi) over [0,n) split across the pool, and wait. */
	void parallelFor(unsigned n, std::function<void(unsigned, unsigned)> const &fn,
			 unsigned threads);
	void startPool(unsigned threads);
	void warpAccum(uint16_t const *bayer, unsigned stridePx, int dx, int dy);

	/*
	 * Persistent worker pool. apply() used to spawn a std::thread pool twice
	 * per frame (stat pass, blend pass) and GlobalMotion spawned more --
	 * measured 5.95 ms of p90-p10 spread on `accum` against <=0.72 ms for
	 * every other stage, i.e. this stage alone was the frame-time jitter.
	 * Creating and joining ~6 threads per frame under contention with
	 * ncnn's own pool is not free and is not predictable.
	 */
	std::vector<std::thread> pool_;
	std::mutex mu_;
	std::condition_variable cvWork_, cvDone_;
	std::function<void(unsigned, unsigned)> job_;
	unsigned jobN_ = 0, jobGen_ = 0, jobDone_ = 0, nThreads_ = 0;
	bool stop_ = false;

	std::vector<AccT> accum_;   /* full-res, DN, one per Bayer pixel */
	/*
	 * Effective sample count per pixel, 1..nMax. Drives both the blend
	 * weight and the gate threshold; scaled down by the gate rather than
	 * hard-reset, so a partially-moving block loses history in proportion.
	 */
	/*
	 * PER-PIXEL SAMPLE COUNT IN FP16. n lives in [1, 1/(1-trust)] -- 5.0 at
	 * the shipped accum_trust 0.8 -- so fp16's ~3 decimal digits are far
	 * more than the gate needs, and it was costing a full-resolution float
	 * array read AND written every frame: 8.5 MB of the ~47 MB this stage
	 * moves. That bandwidth is the binding constraint (the NPU inference
	 * stretches 12.6 -> 27.2 ms when it has to share the bus with this
	 * loop), so the saving shows up twice: once here and once in the net.
	 */
	std::vector<__fp16> n_;
	std::vector<__fp16> nwarp_;
	bool ramp_ = true;

	/*
	 * Global motion compensation. Without it the gate sees a panning camera
	 * as everything moving at once, drops every block, and switches
	 * temporal denoising off exactly when the camera is being used.
	 */
	GlobalMotion motion_;
	std::vector<AccT> warp_;   /* swapped with accum_, so same type */
	bool motion_enable_ = true;
	/*
	 * What the gated fraction WOULD be at a lower kZ, computed from this
	 * frame's blocks. r = (mean-expect)/(kZ*se) scales as 1/kZ, so a block
	 * gates at kZ' exactly when r >= kZ'/kZ -- and weight = trust*(1-r^2)
	 * inverts to give r per block. Reporting this predicts the effect of
	 * retuning kZ without a rebuild-and-restart cycle per value.
	 *
	 * A plain max-r is useless here: weight is CLAMPED to 0 once r>=1, so
	 * it saturates at exactly 1.0 the moment any single block gates.
	 */
	/*
	 * Frame-mean of the gate's own two quantities. On a STATIC scene pure
	 * noise gives mean|diff| = sqrt(2/pi)*sigma = 0.798*sigma exactly, so
	 * the ratio is a direct calibration check on the noise model: well
	 * below 0.798 means sigma is overestimated and the gate is dividing
	 * real motion down into the noise band.
	 */
	/*
	 * Mean effective frame count actually retained (nMax = 1/(1-trust) = 5
	 * at trust 0.8). This is the denoising strength the accumulator is
	 * really delivering: tightening the per-pixel gate to kill ghosting
	 * costs exactly this, so it is the number to watch while tuning
	 * accum_pixel_k rather than guessing at the grain/ghost trade.
	 */
	float last_mean_n_ = 0.0f;
	PackSink sink_;
	float last_mean_diff_ = 0.0f;
	float last_mean_sigma_ = 0.0f;
	float last_g_half_ = 0.0f;    /* gated fraction at kZ/2 */
	float last_g_quarter_ = 0.0f; /* ...and at kZ/4 */
	float motion_ratio_ = 0.90f;
	float motion_agree_ = 0.60f;
	int lastDx_ = 0, lastDy_ = 0;
	unsigned w_ = 0, h_ = 0;
	bool primed_ = false;

	float a_[4] = { 0, 0, 0, 0 };
	float b_[4] = { 0, 0, 0, 0 };
	float trust_ = 0.9f;
	float kZ_ = 10.0f;
	/*
	 * PER-PIXEL motion gate, in sigma. 0 disables it.
	 *
	 * The block gate above answers "is this REGION moving" reliably, because
	 * averaging |new - history| over n pixels shrinks its standard error by
	 * sqrt(n) -- that is why kZ can sit below one sigma. What it cannot do is
	 * localise: a small object inside a 16x16 block either drags 256 pixels'
	 * weight down or is averaged straight through, and the second case is
	 * ghosting.
	 *
	 * PiSP's own TDN gates PER PIXEL instead: it estimates noise as
	 * noise_constant + noise_slope * pixel and compares |current - LTA| against
	 * threshold x that. A single-pixel test is statistically weak -- the
	 * difference of two noisy pixels has sd sqrt(2)*sigma -- but each wrong
	 * decision costs one pixel and the errors do not correlate spatially, so it
	 * localises where the block gate cannot.
	 *
	 * Using both: the block gate decides how much history a region deserves,
	 * then this clamps individual pixels that moved far more than their own
	 * noise explains. Same noise model either way -- var = a*signal + b -- so
	 * the threshold means the same thing at every gain and signal level.
	 */
	float kPix_ = 0.0f;

	/*
	 * sigma(signal) needs a square root per pixel. Tabulating it against
	 * the raw value costs 16 KB and removes 2.1 M sqrts per frame; the
	 * curve is smooth, so 1024 steps is far finer than the noise it
	 * describes.
	 */
	/* Gate decision granularity, in Bayer pixels. Large enough that the
	 * per-pixel noise averages out of the decision, small enough to track a
	 * moving object's outline. */
	static constexpr unsigned kBlock = 16;

	static constexpr unsigned kLut = 1024;
	float sigma_scale_ = 1.0f;
	std::vector<float> lut_;     /* [4][kLut], sigma in DN */
	float lutBlack_ = -1.0f, lutWhite_ = -1.0f;
	float lutScale_ = 0.0f;
};

} /* namespace RPiController */
