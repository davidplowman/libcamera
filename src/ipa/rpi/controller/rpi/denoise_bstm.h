/* SPDX-License-Identifier: BSD-2-Clause */
/*
 * Copyright (C) 2026, Raspberry Pi Ltd
 *
 * Recurrent (temporal) full-frame denoise on the BSTM (Brainstorm) NPU,
 * running an int8 or float32 TFLite model through Mesa's Teflon delegate
 * (a float model runs on the NPU as bf16 or fp16). The network
 * sees the square root of the current frame and its own previous result,
 * and returns a residual and a blend factor; the host does the square root,
 * the blend with the previous result and the squaring back to linear, all
 * in float (see temporal_host.md).
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
	bool checkTensor(const TfLiteTensor *t, const char *what, int channels, bool &isInt8,
			 float &scale, int &zp);
	void buildTables();
	void reset();
	bool checkContinuity(Metadata *imageMetadata);
	template<typename T>
	void packInput(const uint16_t *bayer16, unsigned stridePx, const T *table, T *in);
	void blendInt8(uint16_t *bayer16, unsigned stridePx);
	void blendFloat(uint16_t *bayer16, unsigned stridePx);
	void dumpFrame(const uint16_t *bayer16, unsigned stridePx, const char *what);

	std::string model_;

	/* Path to Mesa's libteflon.so, the TFLite external delegate. */
	std::string delegate_;

	/*
	 * NPU arithmetic for a float32 model, "bf16" or "fp16", passed to the
	 * bstm driver as BSTM_FLOAT when the delegate is created (plain fp32
	 * doesn't compile for this network). An existing BSTM_FLOAT in the
	 * environment takes precedence. Unused for an int8 model.
	 */
	std::string floatFormat_;

	/*
	 * How much of the classic PISP hardware denoise stages to keep on top
	 * of ours. 0 erases a stage's status, so it doesn't run at all on a
	 * frame we've handled ourselves; otherwise its noise constant and slope
	 * (for CDN, its threshold, which is linear in them) are multiplied by
	 * the factor, so 1 leaves the stage as it is. For SDN only the first
	 * noise constant/slope pair is scaled.
	 */
	double sdnFactor_ = 0.0;
	double tdnFactor_ = 0.0;
	double cdnFactor_ = 0.0;

	/*
	 * When set, everything runs as normal (so the recurrent state stays
	 * correct) but only the right half of the denoised result is written
	 * back over the raw buffer, leaving the left half untouched for
	 * side-by-side comparison.
	 */
	bool demo_ = false;

	/*
	 * Threads for the pack and blend passes, and for any operations the
	 * delegate leaves to the TFLite CPU kernels. Both passes run at close
	 * to the memory bandwidth with 2; more doesn't help.
	 */
	int threads_ = 2;

	/*
	 * Run the whole model on the CPU, with TFLite's own kernels (and its
	 * default XNNPACK delegate, where it applies), bypassing the NPU and
	 * the Teflon delegate entirely; float_format is then unused. The
	 * network runs with cpuThreads_ threads, the pack and blend passes
	 * still with threads_. 3 measured fastest for the int8 model on a Pi 5
	 * (4 competes with the rest of the camera stack).
	 */
	bool cpu_ = false;
	int cpuThreads_ = 3;

	/*
	 * Restart from frame 0 when exposure time x analogue gain changes by
	 * more than this factor (either way) from one frame to the next. The
	 * network was only trained on fixed-gain sequences.
	 */
	double resetRatio_ = 2.0;

	/*
	 * Debug only: write the raw frame before and after denoising, for the
	 * first dumpFrames_ frames after each mode switch, to dumpDir_ (as
	 * <dir>/in_NNNN.raw and out_NNNN.raw, packed uint16, width x height).
	 * <dir>/frames.txt lists each frame number and whether it restarted
	 * from frame 0.
	 */
	std::string dumpDir_;
	unsigned dumpFrames_ = 0;
	unsigned dumpCount_ = 0;

	/*
	 * The interpreter must be destroyed before the delegate it was
	 * modified with (see the destructor).
	 */
	std::unique_ptr<tflite::FlatBufferModel> flatbuffer_;
	std::unique_ptr<tflite::Interpreter> interpreter_;
	TfLiteDelegate *tfDelegate_ = nullptr;

	bool init_ = false;
	/* Set by switchMode() when the mode can be denoised. */
	bool active_ = false;
	unsigned width_ = 0; /* cameraMode.width, the raw Bayer width */
	unsigned height_ = 0; /* cameraMode.height, the raw Bayer height */

	/*
	 * netW_/netH_ are the model's fixed input width/height. packW_/packH_
	 * are the network-pixel region backed by real data, one network pixel
	 * per 4x4 raw block. Anything beyond that, up to netW_/netH_, holds
	 * the inputs' encoding of zero.
	 */
	unsigned netW_ = 0;
	unsigned netH_ = 0;
	unsigned packW_ = 0;
	unsigned packH_ = 0;

	/* Tensor indices, from the interpreter's input and output lists. */
	int inX_ = -1;
	int inPrev_ = -1;
	int outR_ = -1;
	int outA_ = -1;

	/*
	 * Whether the model's inputs and outputs are all float32, rather than
	 * all int8.
	 */
	bool isFloat_ = false;

	/* int8 only: per-tensor quantisation, read from the model. */
	float xScale_ = 0.0f;
	int xZp_ = 0;
	float prevScale_ = 0.0f;
	int prevZp_ = 0;
	float rScale_ = 0.0f;
	int rZp_ = 0;
	float aScale_ = 0.0f;
	int aZp_ = 0;

	/*
	 * The raw data is 12-bit, padded with zeros at the bottom, so these are
	 * indexed by raw >> 4. xfTable_ is the square root in float, for the
	 * blend (and the x input of a float model). int8 only: xqTable_ is the
	 * network's int8 x input, and alphaTable_ the sigmoid of a's
	 * dequantised value, indexed by a + 128.
	 */
	std::vector<int8_t> xqTable_;
	std::vector<float> xfTable_;
	std::vector<float> alphaTable_;

	/*
	 * int8 only: the previous frame's result, square-rooted, in packed
	 * NHWC order (packH_ x packW_ x 16), as 16-bit fixed point:
	 * round(prev * 65535).
	 * Never kept as 8-bit: in static areas each frame changes it only
	 * slightly, and those changes would round away. 16 bits is about 250x
	 * finer than the network's int8 input, and halves the blend pass's
	 * memory traffic for prev compared with float (the pass is
	 * memory-bound). A float model keeps the previous result in its float
	 * prev input tensor instead, which TFLite never writes.
	 */
	std::vector<uint16_t> prev_;
	bool firstFrame_ = true;

	/* exposure time x analogue gain of the last frame, for resetRatio_. */
	double lastExposure_ = 0.0;
};

} /* namespace RPiController */
