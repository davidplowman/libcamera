/* SPDX-License-Identifier: BSD-2-Clause */
/*
 * Copyright (C) 2026, Raspberry Pi Ltd
 *
 * Simple NCNN-based full-frame denoise. No temporal accumulation, no
 * tiling, no brightness normalisation -- just: pack Bayer -> 4 planes,
 * run the network, unpack back over the same buffer.
 */
#pragma once

#include <string>
#include <vector>

#include <ncnn/net.h>

#include "../denoise_algorithm.h"

namespace RPiController {

class DenoiseNcnn : public DenoiseAlgorithm
{
public:
	DenoiseNcnn(Controller *controller);
	char const *name() const override;
	int read(const libcamera::ValueNode &params) override;
	void initialise() override;
	void switchMode(CameraMode const &cameraMode, Metadata *metadata) override;
	void prepare(Metadata *imageMetadata) override;
	void setMode(DenoiseMode mode) override;

private:
	void packBayer(const uint16_t *bayer16, unsigned stridePx);
	void packBayerDownscale(const uint16_t *bayer16, unsigned stridePx);
	void unpackBayer(uint16_t *bayer16, unsigned stridePx, const ncnn::Mat &out);
	void interleaveToRaw(uint16_t *bayer16, unsigned stridePx, const float *R, const float *G1,
			     const float *G2, const float *B, unsigned srcStride);
	void upscale2x(const ncnn::Mat &out);
	void upscaleChannel2x(const float *src, unsigned srcStride, unsigned srcW, unsigned srcH,
			       float *dst, unsigned dstStride, unsigned dstH);
	bool runNet(ncnn::Mat &out);
	void computeAlignment();

	std::string param_;
	std::string bin_;

	/*
	 * Same convention as model_denoise.cpp: erase the classic PISP
	 * hardware denoise stages' status on a frame we've handled ourselves,
	 * so they don't also run on top of it.
	 */
	bool sdn_disable_ = true;
	bool tdn_disable_ = true;
	bool cdn_disable_ = true;

	/*
	 * When set, box-average each Bayer colour component over an
	 * additional 2x2 (so each network-plane pixel covers a 4x4 raw
	 * block instead of 2x2), run the network at that quarter resolution,
	 * then bilinearly upscale its output back to half resolution before
	 * the usual raw-Bayer interleave. Halves netW_/netH_ in each
	 * dimension relative to a non-downscale network trained for the same
	 * sensor mode. Requires a network trained on quarter-res input.
	 */
	bool downscale_ = false;

	/* ncnn::Net's OpenMP thread count. Small/shallow networks (many
	 * cheap glue layers, little real per-layer compute) can end up
	 * *slower* with more threads -- the fixed per-region coordination
	 * cost outweighs the parallel work available -- so this is a tuning
	 * parameter, not a fixed constant. */
	int threads_ = 2;

	ncnn::Net net_;
	ncnn::PoolAllocator blobPool_;
	ncnn::PoolAllocator workPool_;

	bool init_ = false;
	unsigned width_ = 0; /* cameraMode.width, the raw Bayer width */
	unsigned height_ = 0; /* cameraMode.height, the raw Bayer height */

	/*
	 * Per-plane buffer geometry, set in switchMode(). netW_/netH_ are
	 * packW_/packH_ each rounded up to a multiple of alignW_/alignH_.
	 * validW_/validH_ (= width_/2, height_/2, regardless of downscale_)
	 * are the half-res Bayer-grid region that the final raw-Bayer
	 * interleave step reads/writes; packW_/packH_ are the net-plane-pixel
	 * region actually backed by real data -- equal to validW_/validH_
	 * when !downscale_, or half that (one net pixel per 4x4 raw block
	 * instead of 2x2) when downscale_. Anything beyond packW_/packH_ up
	 * to netW_/netH_ is zero-filled context. A smaller sensor mode than
	 * the network's own training resolution is fine; it just means a
	 * smaller (or more heavily padded) buffer.
	 */
	unsigned netW_ = 0;
	unsigned netH_ = 0;
	unsigned validW_ = 0;
	unsigned validH_ = 0;
	unsigned packW_ = 0;
	unsigned packH_ = 0;

	/*
	 * Required net-plane-pixel alignment, computed once in initialise()
	 * by computeAlignment(): the product of every Convolution/
	 * ConvolutionDepthWise stride in the .param file. Skip connections
	 * between the encoder and decoder need matching sizes at every
	 * downsampled scale, so the plane size must be an exact multiple of
	 * the total downsampling factor. Defaults to 8 (the deepest network
	 * seen so far) if the .param can't be parsed.
	 *
	 * This is a fallback only, used when fixedPlaneW_/fixedPlaneH_ (below)
	 * are unavailable: multiplying every stride together assumes they're
	 * all on one sequential downsampling chain, which is wrong for a
	 * network whose stride-2 layers aren't all on the path the visible
	 * skip-connections need to match (confirmed on a real network: this
	 * approach overcomputed 64 when the network actually needed exactly
	 * 968x548, silently corrupting the output -- all NaN -- once rounded
	 * up to a larger plane size).
	 */
	unsigned alignW_ = 8;
	unsigned alignH_ = 8;

	/*
	 * Exact required per-plane size (not just "a multiple of"), read
	 * directly from the largest fixed target height/width among the
	 * .param file's Interp layers (doubled -- see computeAlignment()).
	 * ncnn's Interp layer, unless using dynamic_target_size, bakes in an
	 * absolute pixel target rather than a scale factor, so this is a
	 * direct, reliable read of what the network's decoder actually
	 * needs, rather than an inference from encoder topology that can be
	 * wrong for branching/nested architectures. 0 if no such layer was
	 * found (falls back to alignW_/alignH_ instead).
	 */
	unsigned fixedPlaneW_ = 0;
	unsigned fixedPlaneH_ = 0;

	/* 4 planes (R, G1, G2, B), each netW_ x netH_, planar fp16. */
	std::vector<__fp16> buffer_;

	/*
	 * downscale_ only: 4 planes (R, G1, G2, B), each validW_ x validH_,
	 * planar fp32 -- the network's quarter-res output, bilinearly
	 * upscaled 2x in each dimension back to half-res, ready for
	 * interleaveToRaw(). Empty when !downscale_.
	 */
	std::vector<float> upscaleBuffer_;
};

} /* namespace RPiController */
