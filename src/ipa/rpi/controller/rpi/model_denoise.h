/* SPDX-License-Identifier: BSD-2-Clause */
/*
 * Copyright (C) 2026, Raspberry Pi Ltd
 *
 */

#include "denoise_algorithm.h"
#include "temporal_accum.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <cstring>
#include <future>
#include <array>
#include <map>
#include <string>
#include <vector>

#include <sys/ioctl.h>

#include <libcamera/base/log.h>
#include <libcamera/base/shared_fd.h>
#include <libcamera/base/span.h>

#include <linux/dma-buf.h>

#include <ncnn/net.h>

#ifdef RPI_HAVE_BSTM
#include "bstm_full.h"
#include "bstm_trunk.h"
#endif

#include "agc_status.h"
#include "black_level_status.h"
#include "denoise_status.h"
#include "device_status.h"
#include "lux_status.h"
#include "noise_status.h"

using namespace libcamera;

LOG_DEFINE_CATEGORY(RPiModelDenoise)

#define NAME "rpi.model_denoise"

namespace {

/* Bilateral student (bilateral_grid_480x270): single feed-forward I/O. */
constexpr const char *kInputBlob = "in0";
constexpr const char *kOutputBlob = "out0";

constexpr float kGainRef = 128.0f;

static void dmabufSyncStart(const SharedFD &fd)
{
	struct dma_buf_sync dma_sync {};
	dma_sync.flags = DMA_BUF_SYNC_START | DMA_BUF_SYNC_RW;

	::ioctl(fd.get(), DMA_BUF_IOCTL_SYNC, &dma_sync);
}

static void dmabufSyncEnd(const SharedFD &fd)
{
	struct dma_buf_sync dma_sync {};
	dma_sync.flags = DMA_BUF_SYNC_END | DMA_BUF_SYNC_RW;

	::ioctl(fd.get(), DMA_BUF_IOCTL_SYNC, &dma_sync);
}

inline float clampf(float v, float lo, float hi)
{
	return std::max(lo, std::min(hi, v));
}

/*
 * The gain plane handed to the predictor, log2(gain / 128).
 *
 * CAP IT AT WHAT THE MODEL WAS TRAINED ON. m9 was trained on gains 4..512,
 * i.e. a plane range of [-5, +2]. Above +2 it extrapolates with nothing
 * holding it back. Measured on the real weights 2026-09-01: the input_skip
 * coefficient w slides 0.431 -> 0.237 -> 0.172 as the plane goes +2 -> +4 ->
 * +5, so the fraction of the picture replaced by the 5x5 box blur climbs
 * 57% -> 76% -> 83%, retained high-frequency detail falls 0.935 -> 0.634 ->
 * 0.529 of the reference, and the UNBOUNDED b coefficient starts ADDING
 * low-frequency per-channel error rather than removing it (blotch ratio
 * 0.980 at gain 512, 1.008 at 2048, 1.035 at 4096). That is the reported
 * "it blurs and blotches above 512".
 *
 * The three training defences against exactly this -- A_MIN, the Sobel loss
 * and dark_masked_ab_loss -- only ever applied inside the trained range, so
 * nothing constrains the extrapolation. Capping the plane makes the model
 * behave at its best measured operating point instead of guessing.
 *
 * Default is unlimited, which is the historical behaviour.
 */
inline float encodeGain(float gain, float cap = std::numeric_limits<float>::infinity())
{
	return std::min(std::log2(std::max(gain, 1.0f) / kGainRef), cap);
}

} // namespace

namespace RPiController {

class ModelDenoise : public DenoiseAlgorithm
{
public:
	ModelDenoise(Controller *controller);
	/*
	 * MUST join the pipeline before any member dies. Members are destroyed in
	 * reverse declaration order, so pending_out_ goes before net_future_ has
	 * a chance to block -- an in-flight inference then writes into a freed
	 * ncnn::Mat. Joining here makes the thread finished before that starts.
	 */
	~ModelDenoise() override;
	char const *name() const override;
	int read(const libcamera::ValueNode &params) override;
	void initialise() override;
	void switchMode(CameraMode const &cameraMode, Metadata *metadata) override;
	void prepare(Metadata *imageMetadata) override;

	void setMode(DenoiseMode mode) override;

private:
	struct Tile {
		unsigned crop_x0;
		unsigned crop_y0;
	};

	void ensureFpn(float gain);
	void applyFpnFull(uint16_t *bayer16, unsigned stridePx, float black, float white);
	void applyAccum(uint16_t *bayer16, unsigned stridePx, float black, float gain);
	void applyAccumInput(uint16_t *bayer16, unsigned stridePx, float black, float gain,
			     bool writeBack = false, unsigned writeCols = 0);
	void resetTiles(unsigned tiles);
	/* Pack the accumulator's memory into ring slot `slot`, at the same
	 * geometry packTile uses. Source is float DN in Bayer layout. */
	bool packMemory(unsigned tile, unsigned slot, AccT const *mem, unsigned memStride,
			float black, float white);
	bool packTileSlot(unsigned tile, unsigned slot, const uint16_t *raw, unsigned stridePx,
			  float black, float white);
	bool packTile(unsigned tile, const uint16_t *raw, unsigned stridePx, float black,
		      float white);
	void buildFeed(unsigned tile, float gain);
	bool runNet(ncnn::Mat &out);
	void unpackTile(unsigned tile, uint16_t *bayer16, unsigned stridePx, const ncnn::Mat &out,
			float black, float white);

	/*
	 * DECLARED BEFORE net_ ON PURPOSE. Members destruct in REVERSE
	 * declaration order, so anything declared after net_ dies while the
	 * Net still holds Mats drawn from it -- ncnn then prints
	 * "pool allocator destroyed too early" for every outstanding block
	 * and the process segfaults on teardown. Declared here, the pools
	 * outlive net_ and are torn down last.
	 */
	/*
	 * ncnn allocates every intermediate blob from the heap on each forward
	 * unless it is given an allocator. At 968x548 the full-resolution
	 * intermediates are 4.2 MB apiece, so a frame churns tens of MB of
	 * fresh pages -- glibc hands them back to the kernel and they fault in
	 * again next frame. That is invisible in the median and shows up as a
	 * tail, which is exactly what a frame-rate flicker is made of.
	 * A pool keeps the buffers hot instead.
	 */
	/*
	 * Multiplier on the DSNU map's baked-in Wiener shrinkage alpha*.
	 *
	 * alpha* = var(signal)/var(map) is MSE-OPTIMAL, and MSE is not the
	 * objective here. What it leaves behind is FIXED pattern -- identical
	 * every frame -- which the temporal accumulator cannot touch, because
	 * averaging only kills things that change. What over-correcting adds
	 * instead is the map's own estimation noise, which IS independent per
	 * frame and is exactly what the accumulator removes (~19 frames at
	 * accum_trust 0.9). So pushing past alpha* trades a residual this
	 * pipeline can never fix for one it is built to fix.
	 *
	 * The effective alpha is clamped to 1.0: the map is an ESTIMATE of the
	 * pattern, so subtracting more than one of it over-corrects into an
	 * inverted pattern rather than removing more.
	 *
	 * The real fix is more darks -- alpha* rises 0.76 -> 0.99 going from 32
	 * to 1024 (nsa/calib.py) -- but that needs a recapture.
	 */
	float fpn_strength_ = 1.0f;
	ncnn::PoolAllocator blobPool_;
	ncnn::PoolAllocator workPool_;

	ncnn::Net net_;

	bool init_;
	std::string param_;
	std::string bin_;
	unsigned int lux_threshold_;
	bool sdn_disable_, cdn_disable_, tdn_disable_;
	bool demo_;
	/* Demo unpack: mix*ncnn + (1-mix)*original; 1.0 = full paint-over. */
	float blend_;

	unsigned infer_w_;
	unsigned infer_h_;
	unsigned temporal_;
	int threads_;
	/* CPU fp16: 1.5x on A76, visually lossless (84 dB vs fp32). */
	bool fp16_;
	/* Average the 4 CFA blocks per output pixel instead of
	 * subsampling: 2x less noise, 4x fewer pixels to infer. */
	bool bin2x2_;
	/* Fill plane 16 with sqrt(a*S+b) instead of encodeGain(). */
	bool sigma_plane_;
	/* Cap on the gain plane, log2 units. +2.0 = gain 512 = top of training. */
	float gain_plane_max_;
	/*
	 * Whether the net wants a trailing gain plane. False for the
	 * single-frame 4-channel models, whose graph has no Crop to throw
	 * it away.
	 */
	bool gain_plane_;
	/*
	 * h2 and the rest of the mem-sigma family take TWO extra conditioning
	 * planes describing the noise level of the MEMORY slot, so the net can
	 * tell a deep average from a single frame when the gain channel alone
	 * cannot. Training builds them in train_u8_split.py:mem_sigma_planes();
	 * the checkpoint reports macs.in_ch = 11 (8 image + 1 gain + 2 sigma).
	 * Feeding 9 to an 11-channel net fails at ncnn extract().
	 */
	bool mem_sigma_ = false;
	bool no_packing_ = false;
	unsigned feedCh() const
	{
		return temporal_ * 4u + (gain_plane_ ? 1u : 0u) + (mem_sigma_ ? 2u : 0u);
	}
	/*
	 * BSTM (BrainStorm) NPU offload of the predictor trunk. bstm_param_ is
	 * the same graph with layers convdw_48..conv_46 collapsed into one
	 * BstmTrunk layer (see splice_bstm_param.py); bstm_model_ is the
	 * precompiled bf16 .bstm the layer executes. Both must be set, and the
	 * NPU must actually come up, or we load the all-ncnn param instead --
	 * a wrong-but-running denoiser is worse than a slower correct one.
	 */
	bool bstm_enable_;
	std::string bstm_param_;
	std::string bstm_model_;
	/* True only once the spliced graph has loaded AND the NPU answered. */
	bool bstm_active_ = false;
	/*
	 * Full-model offload. Unlike bstm_param_/BstmTrunk, which put only the
	 * predictor on the NPU and left the full-res tail in ncnn, this runs the
	 * ENTIRE graph as one precompiled bf16 program per tile and takes ncnn
	 * out of the inference path. Requires infer_width/height to match the
	 * tile the .bstm was compiled for, and both to be multiples of 8 (the
	 * 2x-chain bilinear needs an exact power-of-two upscale).
	 */
	/*
	 * 0 = auto (the historical behaviour: prefer 2, drop to 1 only if the
	 * infer window will not fit). Must be forced to 1 for full-resolution
	 * output once tiles are small, because a small tile ALWAYS fits at
	 * step 2 and the auto rule then silently subsamples the Bayer -- which
	 * is a resolution reduction, not a tiling choice.
	 */
	unsigned pack_step_cfg_;
	bool bstm_full_enable_;
	std::string bstm_full_model_;
	bool bstm_full_active_ = false;
	/*
	 * Space-to-depth block size. 0 = the plain full-model path. 8 folds the
	 * 8x8 block into channels so nothing in the graph is full-resolution,
	 * which is what lets the NPU fill its channel-parallel array.
	 */
	unsigned bstm_s2d_block_;
	/* buildFeed + s2d fold in ONE pass, straight into the NPU payload. */
	bool foldFeedS2D(unsigned tile, float gain, float *payload);
	/* Set per tile by the caller; runNet() needs to know whether the
	 * payload was already filled by the merged path. */
	bool bstm_feed_merged_ = false;
#ifdef RPI_HAVE_BSTM
	BstmTrunkConfig bstm_cfg_;
	BstmFullModel bstm_full_;
#endif
	/* Per-pixel master-dark (DSNU) subtraction. */
	bool calib_enabled_;
	std::string calib_dir_;
	std::vector<float> fpn_;
	unsigned fpn_h_, fpn_w_;
	float fpn_scale_;
	int fpn_gain_;
	bool fpn_active_;
	/*
	 * Subtract the DSNU map across the WHOLE Bayer frame up front instead
	 * of inside packTile. packTile only touches the pixels the model
	 * samples, which is all the model needs -- but temporal accumulation
	 * reads and writes every pixel, and accumulating uncorrected data is
	 * what makes averaging saturate near N=21. With this on, packTile skips
	 * its own subtraction, so the model's input is unchanged.
	 */
	bool fpn_full_;

	TemporalAccum accum_;
	bool accum_enable_;
	bool accum_post_;
	/*
	 * "input" stage: the model consumes [memory, current] instead of a ring
	 * of raw frames. Memory is the accumulator state BEFORE this frame is
	 * folded in, matching how training built it (mean of the OTHER takes).
	 */
	bool accum_input_;
	float accum_trust_;
	float accum_k_z_;
	bool accum_motion_;
	float accum_reset_frac_;
	float net_min_gain_;
	float feed_lo_;
	int accum_gain_;
	double ms_fpn_ = 0.0, ms_accum_ = 0.0;
	/* Diagnostic: fraction of 16px blocks the motion gate rejected on the
	 * last frame. If this sits near 0 while the scene is moving, the gate
	 * is not detecting motion at all and anything downstream of it -- the
	 * halo dilation included -- cannot affect ghosting. */
	float last_gated_ = 0.0f;
	double ms_build_ = 0.0, ms_fold_ = 0.0;
	/*
	 * CPU/NPU PIPELINING. The stages inside one frame are a strict chain --
	 * accum -> pack -> feed -> net -> unpack -- so nothing within a frame can
	 * overlap. But the NPU and the CPU are separate engines, and frame N's
	 * inference does not depend on frame N+1's accumulator pass. So the net is
	 * launched asynchronously and joined at the TOP of the next frame, with
	 * accum running against it:
	 *
	 *   1 launch accum(N)            <- runs while net(N-1) is on the NPU
	 *   2 join net(N-1)
	 *   3 join accum(N)
	 *   4 pack(N)   reads RAW bayer16(N), packMemory from fresh accum state
	 *   5 feed(N)   writes the input payload (safe: net(N-1) has joined)
	 *   6 unpack(N-1) -> bayer16(N)  (safe: pack already read the raw data)
	 *   7 launch net(N)
	 *
	 * Measured serial cost is ~32.8 ms CPU against 12.6 ms NPU
	 * (net 15.3 = exec 12.64 + unfold 2.66), so this trades ONE FRAME OF
	 * LATENCY for roughly max() instead of sum().
	 *
	 * The input payload is single-buffered, which is why feed may only run
	 * after the join at step 2, and the output payload is read at step 6
	 * before being overwritten at step 7.
	 */
	bool net_pipeline_ = false;
	bool fused_pack_ = false;
	/* full-frame diagnostic reductions; one frame in 30 pays for
	 * all of them at once, which costs that frame its deadline. */
	bool probes_ = false;
	bool fused_pack_live_ = false;
	std::future<bool> net_future_;
	std::future<void> accum_future_;
	ncnn::Mat pending_out_;
	bool pending_valid_ = false;
	/*
	 * The brightness scale pending_out_ was ENCODED with. unpackTile divides
	 * by bnorm_scale_, and with the pipeline on it runs a frame late, so it
	 * was dividing frame N-1's pixels by frame N's scale. In steady state the
	 * two are equal and nothing shows; while bnorm is still ramping (a fresh
	 * IPA process, which is exactly what a still capture spawns) they differ
	 * by up to bnorm_max, and the frame comes out that many times too dark.
	 */
	float pending_bnorm_ = 1.0f;
	double ms_net_async_ = 0.0;
	void joinPipeline();

	bool feed_blend_ = true;
	bool accum_update_first_ = false;
	float accum_sigma_scale_ = 1.0f;
	float accum_motion_ratio_ = 0.90f;
	float accum_motion_agree_ = 0.60f;
	std::map<int, std::array<float, 4>> noise_a_, noise_b_;
	float black_default_;
	float white_default_;
	float gain_default_;
	/* Input brightness normalisation: the net trained at mean ~0.115;
	 * live dark scenes sit ~8x lower and it smooths detail away. */
	float bnorm_target_;
	float bnorm_max_;
	float bnorm_scale_;
	/* STATS_PATCH: mean of the 16 image planes before normalisation. */
	float bnorm_mean_ = 0.0f;
	/*
	 * BNORM FOLDED INTO PACK. The brightness scale used to be a separate
	 * read-modify-write over temporal_*4*plane fp16 elements -- 4.24M of
	 * them, ~17 MB of DRAM traffic -- purely to multiply by one scalar. The
	 * pack stages already write every one of those elements, so the scale
	 * rides along in their `inv` factor for free and the pass disappears.
	 * This pipeline is bandwidth bound (see the note above applyAccumInput:
	 * overlapping made BOTH sides slower), so removing bytes is the only
	 * thing that helps.
	 *
	 * It is the PREVIOUS frame's scale, because bmean is measured from the
	 * packed feed and so is not known until after the pack. That is a
	 * one-frame lag on a scene-brightness ratio; in practice bs sits pinned
	 * at the bnorm_max clamp (bmean ~0.0019 against a 0.115 target, ratio
	 * ~60), so the lag is unobservable. bnorm_scale_ always holds the scale
	 * ACTUALLY applied to the frame in flight, which is what unpackTile
	 * divides back out.
	 */
	float bnorm_pre_ = 1.0f;
	/*
	 * Cached state of the constant gain plane in feed_chw_. It is one value
	 * repeated 530k times and depends only on the analogue gain, so it is
	 * refilled on a gain transition rather than every frame. Invalidated
	 * wherever feed_chw_ is reallocated (resetTiles).
	 */
	float gplane_gch_ = 0.0f;
	bool gplane_valid_ = false;

	/* Dimensions of the incoming Bayer frame for this mode. */
	unsigned int width_;
	unsigned int height_;

	/* Bayer→packed sampling: CFA-cell step (1 or 2); bstep = 2 * step. */
	unsigned pack_step_;
	std::vector<Tile> tiles_;
	unsigned tiles_x_;
	unsigned tiles_y_;
	unsigned cover_bw_; /* Bayer width covered by all tiles */
	unsigned cover_bh_; /* Bayer height covered by all tiles */

	/*
	 * Per-tile ring of packed frames, each infer_h_ * infer_w_ * 4 floats
	 * (HWC). Shared ring_pos_/ring_filled_ — all tiles advance together.
	 */
	/*
	 * fp16 storage. The net runs fp16, so ncnn cast this buffer on every
	 * frame anyway -- holding it as float only bought an extra 19 MB of
	 * writes, 19 MB of reads and the cast itself. Arithmetic still happens
	 * in float: __fp16 promotes on every use, so only STORAGE rounds, which
	 * is exactly what the cast was doing at the end.
	 */
	using FeedT = __fp16;
	/*
	 * MEMORY-INPUT MODE BYPASSES THE RING. With accum_stage "input" the
	 * ring never rotates -- runFrame pins ring_pos_ to 0 every frame and
	 * writes slot 0 = memory, slot 1 = current. Staging those in rings_ and
	 * then copying them into feed_chw_ was ~17 MB a frame of pure transit.
	 * slotPtr() sends the pack straight at the feed planes instead; the
	 * blend in buildFeed is elementwise so it works in place unchanged.
	 * The rotating ring is still used for the 4-frame (non-memory) mode.
	 */
	FeedT *slotPtr(unsigned tile, unsigned slot)
	{
		if (accum_input_)
			return feed_chw_.data() +
			       size_t(slot) * 4u * infer_h_ * infer_w_;
		return rings_[tile][slot].data();
	}
	std::vector<std::vector<std::vector<FeedT>>> rings_;
	unsigned ring_pos_;
	unsigned ring_filled_;
	std::vector<FeedT> feed_chw_; /* 17 * H * W */
	/* fp32 staging, only for the use_fp16_storage=false path (see runNet) */
	std::vector<float> feed_f32_;

	unsigned long long n_frames_;
};

} // namespace RPiController
