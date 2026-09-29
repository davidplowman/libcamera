/* SPDX-License-Identifier: BSD-2-Clause */
/*
 * Copyright (C) 2026, Raspberry Pi Ltd
 *
 * Simple NCNN-based full-frame denoise. No temporal accumulation, no
 * tiling, no brightness normalisation -- just: pack Bayer -> 4 planes,
 * space-to-depth -> 16 planes, run the network, unpack back over the same
 * buffer.
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
	void zeroBorder();
	void unpackBayer(uint16_t *bayer16, unsigned stridePx, const ncnn::Mat &out);
	void depthToRaw(uint16_t *bayer16, unsigned stridePx, const ncnn::Mat &out);
	void depthToSpace(const ncnn::Mat &out);
	void interleaveToRaw(uint16_t *bayer16, unsigned stridePx, const float *R, const float *G1,
			     const float *G2, const float *B, unsigned srcStride);
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
	 * additional 2x2 before the space-to-depth (so each network-plane
	 * pixel covers an 8x8 raw block instead of 4x4), run the network at
	 * that resolution, then depth-to-space and bilinearly upscale its
	 * output back to half resolution before the usual raw-Bayer
	 * interleave. Halves netW_/netH_ in each dimension relative to a
	 * non-downscale network trained for the same sensor mode. Requires a
	 * network trained on quarter-res input.
	 */
	bool downscale_ = false;

	/* ncnn::Net's OpenMP thread count. Small/shallow networks (many
	 * cheap glue layers, little real per-layer compute) can end up
	 * *slower* with more threads -- the fixed per-region coordination
	 * cost outweighs the parallel work available -- so this is a tuning
	 * parameter, not a fixed constant. Also used by the downscale_
	 * packing/unpacking loops; the full-res pack/unpack are memory-bound
	 * and always single-threaded. */
	int threads_ = 2;

	/*
	 * The pools must be declared before (so destroyed after) net_, which
	 * still holds memory from them until it is itself destroyed.
	 */
	ncnn::PoolAllocator blobPool_;
	ncnn::PoolAllocator workPool_;
	ncnn::Net net_;

	bool init_ = false;
	unsigned width_ = 0; /* cameraMode.width, the raw Bayer width */
	unsigned height_ = 0; /* cameraMode.height, the raw Bayer height */

	/*
	 * Per-plane buffer geometry, set in switchMode(). netW_/netH_ are
	 * the network's plane size (fixedPlaneW_/fixedPlaneH_, or failing
	 * that packW_/packH_ rounded up to a multiple of alignW_/alignH_).
	 * packW_/packH_ are the net-plane-pixel region actually backed by
	 * real data: one net pixel per 4x4 raw block, or per 8x8 raw block
	 * when downscale_. bayerW_/bayerH_ (= packW_*2, packH_*2) are the
	 * size of the 4 Bayer planes before space-to-depth -- half-res, or
	 * quarter-res when downscale_. validW_/validH_ (half-res) are the
	 * Bayer-grid region that the final write back to raw covers -- equal
	 * to bayerW_/bayerH_ when !downscale_, or twice that when downscale_.
	 * Anything beyond packW_/packH_ up to netW_/netH_ is zero-filled
	 * context. A smaller sensor mode than the network's own training
	 * resolution is fine; it just means a smaller (or more heavily
	 * padded) buffer.
	 */
	unsigned netW_ = 0;
	unsigned netH_ = 0;
	unsigned packW_ = 0;
	unsigned packH_ = 0;
	unsigned bayerW_ = 0;
	unsigned bayerH_ = 0;
	unsigned validW_ = 0;
	unsigned validH_ = 0;

	/*
	 * Distance between channel planes in buffer_, in fp16 samples:
	 * netW_ * netH_ rounded up to a 16-byte boundary, matching the cstep
	 * ncnn::Mat uses when wrapping external data.
	 */
	size_t planeStride_ = 0;

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
	 * .param file's Interp layers (see computeAlignment()).
	 * ncnn's Interp layer, unless using dynamic_target_size, bakes in an
	 * absolute pixel target rather than a scale factor, so this is a
	 * direct, reliable read of what the network's decoder actually
	 * needs, rather than an inference from encoder topology that can be
	 * wrong for branching/nested architectures. 0 if no such layer was
	 * found (falls back to alignW_/alignH_ instead).
	 */
	unsigned fixedPlaneW_ = 0;
	unsigned fixedPlaneH_ = 0;

	/*
	 * The network's input channel count, read from its first
	 * Convolution layer by computeAlignment() (0 if it couldn't be).
	 * Must be 16.
	 */
	unsigned inChannels_ = 0;

	/*
	 * 16 planes, each netW_ x netH_ (planeStride_ apart), planar fp16:
	 * the 4 Bayer planes (R, G1, G2, B) after space_to_depth(2), channel
	 * c*4 + i*2 + j being Bayer plane c's (row i, column j) sub-position.
	 */
	std::vector<__fp16> buffer_;

	/*
	 * downscale_ only: 4 planes (R, G1, G2, B), each bayerW_ x bayerH_,
	 * planar fp32 -- the network's output after depth_to_space(2), i.e.
	 * still quarter-res. Empty when !downscale_.
	 */
	std::vector<float> bayerBuffer_;

	/*
	 * downscale_ only: 4 planes (R, G1, G2, B), each validW_ x validH_,
	 * planar fp32 -- bayerBuffer_ bilinearly upscaled 2x in each
	 * dimension back to half-res, ready for interleaveToRaw(). Empty
	 * when !downscale_.
	 */
	std::vector<float> upscaleBuffer_;
};

} /* namespace RPiController */
