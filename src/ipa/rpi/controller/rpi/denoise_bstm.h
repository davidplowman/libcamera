/* SPDX-License-Identifier: BSD-2-Clause */
/*
 * Copyright (C) 2026, Raspberry Pi Ltd
 *
 * Simple full-frame denoise on the BSTM (Brainstorm) NPU, running a
 * TFLite model through Mesa's Teflon delegate. No temporal accumulation,
 * no tiling, no brightness normalisation -- just: pack Bayer -> 4 planes,
 * space-to-depth -> 16 channels (NHWC), run the network, unpack back over
 * the same buffer.
 */
#pragma once

#include <memory>
#include <string>
#include <vector>

#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/model.h>

#include "../denoise_algorithm.h"

namespace RPiController {

class DenoiseBstm : public DenoiseAlgorithm
{
public:
	DenoiseBstm(Controller *controller);
	~DenoiseBstm();
	char const *name() const override;
	int read(const libcamera::ValueNode &params) override;
	void initialise() override;
	void switchMode(CameraMode const &cameraMode, Metadata *metadata) override;
	void prepare(Metadata *imageMetadata) override;
	void setMode(DenoiseMode mode) override;

private:
	bool checkTensor(const TfLiteTensor *t, const char *what, bool &isInt8, float &scale,
			 int &zp);
	void packInput(const uint16_t *bayer16, unsigned stridePx);
	void unpackOutput(uint16_t *bayer16, unsigned stridePx);
	void packBayer(const uint16_t *bayer16, unsigned stridePx, float *in);
	template<typename T>
	void packBayerFull(const uint16_t *bayer16, unsigned stridePx, T *in);
	void packBayerDownscale(const uint16_t *bayer16, unsigned stridePx, float *in);
	template<typename T>
	void zeroBorder(T *in, T zero);
	void quantiseInput(const float *v, int8_t *q);
	void dequantiseOutput(const int8_t *q, float *v);
	void unpackBayer(uint16_t *bayer16, unsigned stridePx, const float *out);
	template<typename T>
	void depthToRaw(uint16_t *bayer16, unsigned stridePx, const T *out);
	void depthToSpace(const float *out);
	void interleaveToRaw(uint16_t *bayer16, unsigned stridePx, const float *R, const float *G1,
			     const float *G2, const float *B, unsigned srcStride);
	void upscaleChannel2x(const float *src, unsigned srcStride, unsigned srcW, unsigned srcH,
			      float *dst, unsigned dstStride, unsigned dstH);

	std::string model_;

	/* Path to Mesa's libteflon.so, the TFLite external delegate. */
	std::string delegate_;

	/*
	 * NPU arithmetic ("fp32", "bf16" or "fp16"), passed to the bstm
	 * driver as BSTM_FLOAT when the delegate is created. An existing
	 * BSTM_FLOAT in the environment takes precedence.
	 */
	std::string floatFormat_;

	/*
	 * Same convention as denoise_ncnn.cpp: erase the classic PISP
	 * hardware denoise stages' status on a frame we've handled ourselves,
	 * so they don't also run on top of it.
	 */
	bool sdn_disable_ = true;
	bool tdn_disable_ = true;
	bool cdn_disable_ = true;

	/*
	 * When set, box-average each Bayer colour component over an
	 * additional 2x2 before the space-to-depth (so each network pixel
	 * covers an 8x8 raw block instead of 4x4), run the network at that
	 * resolution, then depth-to-space and bilinearly upscale its output
	 * back to half resolution before the usual raw-Bayer interleave.
	 * Requires a network trained on quarter-res input.
	 */
	bool downscale_ = false;

	/*
	 * When set, everything runs as normal but only the right half of the
	 * denoised result is written back over the raw buffer, leaving the
	 * left half untouched for side-by-side comparison.
	 */
	bool demo_ = false;

	/*
	 * Set for a model exported to output only its residual (the
	 * correction to its input, e.g. DenoiseNet's self.out(feat) without the
	 * final "x +"), which quantises much more finely in an int8 model. The
	 * output is then added to the full-precision raw input on write-back,
	 * rather than replacing it. Must match how the model was exported.
	 */
	bool residual_ = false;

	/*
	 * Threads for the downscale_ packing/unpacking loops and for any
	 * operations the delegate leaves to the TFLite CPU kernels. The
	 * full-res pack/unpack are memory-bound and always single-threaded.
	 */
	int threads_ = 2;

	/*
	 * The interpreter must be destroyed before the delegate it was
	 * modified with (see the destructor).
	 */
	std::unique_ptr<tflite::FlatBufferModel> flatbuffer_;
	std::unique_ptr<tflite::Interpreter> interpreter_;
	TfLiteDelegate *tfDelegate_ = nullptr;

	bool init_ = false;
	unsigned width_ = 0; /* cameraMode.width, the raw Bayer width */
	unsigned height_ = 0; /* cameraMode.height, the raw Bayer height */

	/*
	 * Buffer geometry, as in denoise_ncnn.h. netW_/netH_ are the model's
	 * fixed input width/height (read from its input tensor). packW_/packH_
	 * are the network-pixel region actually backed by real data: one
	 * network pixel per 4x4 raw block, or per 8x8 raw block when
	 * downscale_. bayerW_/bayerH_ (= packW_*2, packH_*2) are the size of
	 * the 4 Bayer planes before space-to-depth. validW_/validH_ (half-res)
	 * are the Bayer-grid region written back to raw -- equal to
	 * bayerW_/bayerH_ when !downscale_, or twice that when downscale_.
	 * Anything beyond packW_/packH_ up to netW_/netH_ is zero-filled.
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

	/*
	 * The model's input and output tensor types, detected when it's
	 * loaded: float32, or int8 with a per-tensor scale and zero point
	 * (what litert_torch produces for a fully int8 model).
	 */
	bool inInt8_ = false;
	bool outInt8_ = false;
	float inScale_ = 0.0f;
	float outScale_ = 0.0f;
	int inZp_ = 0;
	int outZp_ = 0;

	/*
	 * downscale_ with an int8 input or output only: one float copy of the
	 * network tensor, packed or unpacked as float then (de)quantised.
	 */
	std::vector<float> scratch_;
};

} /* namespace RPiController */
