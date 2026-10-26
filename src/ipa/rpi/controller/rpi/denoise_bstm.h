/* SPDX-License-Identifier: BSD-2-Clause */
/*
 * Copyright (C) 2026, Raspberry Pi Ltd
 *
 * Recurrent (temporal) denoise on the BSTM (Brainstorm) NPU, running an int8
 * or float32 TFLite model through Mesa's Teflon delegate (a float model runs
 * on the NPU as bf16 or fp16). The network sees the square root of the
 * current frame and its own previous result, and returns a residual and a
 * blend factor; the host does the square root, the blend with the previous
 * result and the squaring back to linear, all in float (see
 * temporal_host.md).
 *
 * A frame larger than the model is processed as a grid of model-sized,
 * overlapping tiles, cross-faded where they overlap. Tiles small enough for
 * the NPU to keep all their intermediate tensors in its on-chip memory run
 * much faster than a whole frame, which streams every layer through DRAM.
 * Packing, inference and blending are pipelined across the tiles.
 */
#pragma once

#include <condition_variable>
#include <memory>
#include <mutex>
#include <stdlib.h>
#include <string>
#include <thread>
#include <vector>

#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/model.h>

#include "../denoise_algorithm.h"

struct DeviceStatus;

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

	/*
	 * The tile layout along one axis, in network (packed) pixels. Each
	 * position of the frame is covered by tile0[p] with weight weight0[p],
	 * and, where weight0[p] < 1, also by tile0[p] + 1 with the rest. The
	 * weights ramp linearly across the middle of each overlap. Tile i
	 * finishes positions [region[i], region[i + 1]): it's the last tile
	 * contributing to them, so they can be blended as soon as it's done.
	 */
	struct TileAxis {
		unsigned int count = 0; /* tiles along this axis */
		unsigned int valid = 0; /* network pixels per tile backed by the frame */
		std::vector<unsigned int> origin; /* each tile's first frame position */
		std::vector<unsigned int> region; /* count + 1 boundaries, from 0 to the frame size */
		std::vector<uint16_t> tile0;
		std::vector<float> weight0;
	};

private:
	bool checkTensor(const TfLiteTensor *t, const char *what, int channels, bool &isInt8,
			 float &scale, int &zp);
	void buildTables();
	void reset();
	void checkGain(double analogueGain);
	void checkForReset(const DeviceStatus &deviceStatus);
	template<typename T>
	void packTile(unsigned int k, const uint16_t *bayer16, unsigned int stridePx, const T *table);
	void blendRegionInt8(unsigned int k, uint16_t *bayer16, unsigned int stridePx);
	void blendRegionFloat(unsigned int k, uint16_t *bayer16, unsigned int stridePx);
	void dumpFrame(const uint16_t *bayer16, unsigned int stridePx, const char *what);

	void npuLoop();
	void startFrame(unsigned int tiles);
	void publishPacked(unsigned int packed);
	bool waitDone(unsigned int tiles);

	unsigned int tiles() const { return tilesX_.count * tilesY_.count; }
	uint8_t *tileBase(unsigned int k) const { return tileBuffers_.get() + k * tileBytes_; }
	template<typename T>
	T *tileX(unsigned int k) const { return reinterpret_cast<T *>(tileBase(k) + offX_); }
	template<typename T>
	T *tilePrev(unsigned int k) const { return reinterpret_cast<T *>(tileBase(k) + offPrev_); }
	template<typename T>
	T *tileR(unsigned int k) const { return reinterpret_cast<T *>(tileBase(k) + offR_); }
	template<typename T>
	T *tileA(unsigned int k) const { return reinterpret_cast<T *>(tileBase(k) + offA_); }

	/* Parameters from the tuning file, fixed once read. */
	struct Config {
		int read(const libcamera::ValueNode &params);

		/*
		 * The TFLite model (int8 or float32); its size is the tile size.
		 * The tuning file's "model" is either the path itself or the name
		 * of another key in the block whose value is the path, so that
		 * several models can be listed and one chosen by name.
		 */
		std::string model;

		/* Path to Mesa's libteflon.so, the TFLite external delegate. */
		std::string delegate;

		/*
		 * NPU arithmetic for a float32 model, "bf16" or "fp16", passed to
		 * the bstm driver as BSTM_FLOAT when the delegate is created (plain
		 * fp32 doesn't compile for this network). An existing BSTM_FLOAT
		 * in the environment takes precedence. Unused for an int8 model.
		 */
		std::string floatFormat;

		/*
		 * How much of the classic PISP hardware denoise stages to keep on
		 * top of ours. 0 erases a stage's status, so it doesn't run at all
		 * on a frame we've handled ourselves; otherwise its noise constants
		 * and slopes (for SDN, both pairs; for CDN, its threshold, which is
		 * linear in them) are multiplied by the factor, so 1 leaves the
		 * stage as it is.
		 */
		double sdnFactor = 0.0;
		double tdnFactor = 0.0;
		double cdnFactor = 0.0;

		/*
		 * When set, everything runs as normal (so the recurrent state stays
		 * correct) but only the right half of the denoised result is
		 * written back over the raw buffer, leaving the left half untouched
		 * for side-by-side comparison.
		 */
		bool demo = false;

		/*
		 * The minimum overlap between neighbouring tiles, in network pixels
		 * (4 raw pixels each). The tiles are spread evenly over the frame,
		 * so the actual overlap is usually larger. Unused when the model is
		 * as large as the frame.
		 */
		unsigned int minOverlap = 4;

		/*
		 * Run the whole model on the CPU, with TFLite's own kernels (and
		 * its default XNNPACK delegate, where it applies), bypassing the NPU
		 * and the Teflon delegate entirely; floatFormat is then unused. The
		 * network runs with cpuThreads threads. 3 measured fastest for the
		 * int8 model on a Pi 5 (4 competes with the rest of the camera
		 * stack).
		 */
		bool cpu = false;
		int cpuThreads = 3;

		/*
		 * Restart from frame 0 when exposure time x analogue gain changes
		 * by more than this factor (either way) from one frame to the next.
		 * The network was only trained on fixed-gain sequences.
		 */
		double resetRatio = 2.0;

		/*
		 * The network runs only at high analogue gain, where the classic
		 * hardware denoise stages fall short. It switches on when the gain
		 * reaches gainThresholdOn and off again when it drops below
		 * gainThresholdOff; between the two it stays as it was, so it
		 * doesn't flip back and forth. While off, the frame and the
		 * hardware stages' statuses are left untouched.
		 */
		double gainThresholdOn = 32.0;
		double gainThresholdOff = 24.0;

		/*
		 * Debug only: write the raw frame before and after denoising, for
		 * the first dumpFrames frames after each change of camera mode, to
		 * dumpDir (as <dir>/in_NNNN.raw and out_NNNN.raw, packed uint16,
		 * width x height). <dir>/frames.txt lists each frame number and
		 * whether it restarted from frame 0.
		 */
		std::string dumpDir;
		unsigned int dumpFrames = 0;
	};
	Config config_;

	/* Debug only: frames dumped since the last change of camera mode. */
	unsigned int dumpCount_ = 0;

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
	/* The camera mode last set up, to tell a real mode change from a restart. */
	CameraMode mode_ = {};
	unsigned int width_ = 0; /* cameraMode.width, the raw Bayer width */
	unsigned int height_ = 0; /* cameraMode.height, the raw Bayer height */

	/*
	 * netW_/netH_ are the model's fixed input width/height, which is the
	 * tile size. packW_/packH_ are the frame's size in network pixels, one
	 * per 4x4 raw block. Where the frame is smaller than the model, the
	 * single tile's remaining pixels hold the inputs' encoding of zero.
	 */
	unsigned int netW_ = 0;
	unsigned int netH_ = 0;
	unsigned int packW_ = 0;
	unsigned int packH_ = 0;

	TileAxis tilesX_;
	TileAxis tilesY_;

	/*
	 * Every tile's x and prev inputs and r and a outputs, allocated once
	 * per mode: tile k's buffers start at tileBase(k), each 64-byte
	 * aligned. Before each tile's inference, the interpreter's input and
	 * output tensors are pointed at its buffers, so nothing is copied (the
	 * Teflon delegate reads each tensor's data pointer at every Invoke()).
	 */
	struct FreeDeleter {
		void operator()(uint8_t *p) const { free(p); }
	};
	std::unique_ptr<uint8_t[], FreeDeleter> tileBuffers_;
	size_t tileBuffersSize_ = 0;
	size_t tileBytes_ = 0;
	size_t offX_ = 0;
	size_t offPrev_ = 0;
	size_t offR_ = 0;
	size_t offA_ = 0;

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
	 * The previous frame's result, square-rooted, in packed NHWC order
	 * (packH_ x packW_ x 16): the recurrent state. Each tile's prev input
	 * is packed from it, and only the blend writes it, so the tiles'
	 * buffers are pure inputs, never touched while the NPU may be using
	 * them. int8 model: prev_, as 16-bit fixed point, round(prev * 65535).
	 * Never kept as 8-bit: in static areas each frame changes it only
	 * slightly, and those changes would round away; 16 bits is about 250x
	 * finer than the network's int8 input, and half the memory traffic of
	 * float. prevQ_ is the same state quantised to the network's int8 prev
	 * input, made by the blend from the unrounded result and copied into
	 * each tile by packTile(). Float model: prevFloat_.
	 */
	std::vector<uint16_t> prev_;
	std::vector<int8_t> prevQ_;
	std::vector<float> prevFloat_;
	bool firstFrame_ = true;

	/* Whether the network is running, by Config::gainThresholdOn/Off. */
	bool aiOn_ = false;

	/* exposure time x analogue gain of the last frame, for Config::resetRatio. */
	double lastExposure_ = 0.0;

	/*
	 * Inference runs on npuThread_, the only thread that calls Invoke()
	 * once initialise() is done. prepare() packs tiles in raster order and
	 * publishes how many are ready (packed_); the thread runs them in that
	 * order and counts them off (done_), and prepare() blends the region
	 * each tile finishes as soon as it's done. All guarded by
	 * npuMutex_, with npuCv_ signalled on every change.
	 */
	std::thread npuThread_;
	std::mutex npuMutex_;
	std::condition_variable npuCv_;
	unsigned int packed_ = 0;
	unsigned int done_ = 0;
	bool npuError_ = false;
	bool npuExit_ = false;
	double npuMs_ = 0.0; /* time spent in Invoke() this frame */
};

} /* namespace RPiController */
