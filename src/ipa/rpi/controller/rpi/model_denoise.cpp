/* SPDX-License-Identifier: BSD-2-Clause */
/*
 * Copyright (C) 2026 Raspberry Pi Ltd
 *
 * NCNN bilateral-grid denoise algorithm (packed Bayer, in0/out0).
 */

#include "model_denoise.h"

#include "bilateral_assemble.h"
#include "bilateral_assemble_lr.h"
#include "guide_blur5.h"
#include "kpn_assemble_lr.h"
#include "lf_locks.h"
#include "box_blur_r.h"
#include "input_shrink.h"

#include <cerrno>
#include <cstdio>
#include <cmath>
#include <string>
#include <chrono>
#include <thread>

#ifdef _OPENMP
#include <omp.h>
#endif

#if defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>
#endif

namespace {
#if defined(__ARM_NEON) && defined(__ARM_FP16_FORMAT_IEEE)
/*
 * Eight packed pixels of one Bayer channel: normalise, clamp, store fp16.
 * The CFA read that feeds this is a stride-2 walk, which vld2q_u16 does in
 * one instruction -- it de-interleaves even and odd lanes, which IS the
 * R/G1 (and G2/B) split, so no gather is needed.
 */
inline void packStore8(__fp16 *dst, uint16x8_t v, float32x4_t vblack,
		       float32x4_t vinv, float32x4_t lo, float32x4_t hi)
{
	float32x4_t a = vcvtq_f32_u32(vmovl_u16(vget_low_u16(v)));
	float32x4_t b = vcvtq_f32_u32(vmovl_u16(vget_high_u16(v)));
	a = vmulq_f32(vsubq_f32(a, vblack), vinv);
	b = vmulq_f32(vsubq_f32(b, vblack), vinv);
	a = vminq_f32(vmaxq_f32(a, lo), hi);
	b = vminq_f32(vmaxq_f32(b, lo), hi);
	vst1q_f16(dst, vcombine_f16(vcvt_f16_f32(a), vcvt_f16_f32(b)));
}
#endif
} /* namespace */

using namespace RPiController;
using namespace libcamera;

ModelDenoise::ModelDenoise(Controller *controller)
	: DenoiseAlgorithm(controller), init_(false), lux_threshold_(1),
	  sdn_disable_(true), cdn_disable_(true), tdn_disable_(true), demo_(false),
	  blend_(0.5f), infer_w_(480), infer_h_(270), temporal_(4), threads_(2), fp16_(true), bin2x2_(false), sigma_plane_(false), gain_plane_(true),
	  calib_enabled_(true), fpn_h_(0), fpn_w_(0),
	  fpn_scale_(0.0f), fpn_gain_(-1), fpn_active_(false), fpn_full_(false),
	  accum_enable_(false), accum_post_(true), accum_trust_(0.9f),
	  accum_k_z_(20.0f), accum_motion_(true), accum_reset_frac_(0.5f),
	  net_min_gain_(0.0f), feed_lo_(-0.25f), accum_gain_(0),
	  black_default_(3200.0f), white_default_(65535.0f), gain_default_(512.0f), bnorm_target_(0.115f),
	  bnorm_max_(32.0f), bnorm_scale_(1.0f),
	  width_(0), height_(0), pack_step_(2), tiles_x_(0), tiles_y_(0),
	  cover_bw_(0), cover_bh_(0), ring_pos_(0), ring_filled_(0), n_frames_(0)
{
}

char const *ModelDenoise::name() const
{
	return NAME;
}

int ModelDenoise::read([[maybe_unused]] const libcamera::ValueNode &params)
{
	param_ = params["param"].get<std::string>("");
	bin_ = params["bin"].get<std::string>("");
	bstm_enable_ = params["bstm"].get<bool>(false);
	bstm_param_ = params["bstm_param"].get<std::string>("");
	bstm_model_ = params["bstm_model"].get<std::string>("");
	pack_step_cfg_ = params["pack_step"].get<unsigned int>(0);
	bstm_full_enable_ = params["bstm_full"].get<bool>(false);
	bstm_full_model_ = params["bstm_full_model"].get<std::string>("");
	bstm_s2d_block_ = params["bstm_s2d_block"].get<unsigned int>(0);
	lux_threshold_ = params["lux_threshold"].get<unsigned int>(1);
	sdn_disable_ = params["sdn_disable"].get<bool>(true);
	tdn_disable_ = params["tdn_disable"].get<bool>(true);
	cdn_disable_ = params["cdn_disable"].get<bool>(true);
	demo_ = params["demo"].get<bool>(false);
	blend_ = params["blend"].get<float>(0.5f);
	if (blend_ < 0.0f)
		blend_ = 0.0f;
	else if (blend_ > 1.0f)
		blend_ = 1.0f;

	infer_w_ = params["infer_width"].get<unsigned int>(480);
	infer_h_ = params["infer_height"].get<unsigned int>(270);
	temporal_ = params["temporal"].get<unsigned int>(4);
	threads_ = params["threads"].get<int>(2);
	fp16_ = params["fp16"].get<bool>(true);
	bin2x2_ = params["bin2x2"].get<bool>(false);
	sigma_plane_ = params["sigma_plane"].get<bool>(false);
	mem_sigma_ = params["mem_sigma"].get<bool>(false);
	no_packing_ = params["no_packing"].get<bool>(false);
	gain_plane_ = params["gain_plane"].get<bool>(true);
	gain_plane_max_ = float(params["gain_plane_max"].get<double>(1e30));
	calib_enabled_ = params["calib"].get<bool>(true);
	calib_dir_ = params["calib_dir"].get<std::string>(
		"/usr/local/share/libcamera/ipa/rpi/pisp/fpn");
	fpn_full_ = params["fpn_full"].get<bool>(false);
	fpn_strength_ = clampf(params["fpn_strength"].get<double>(1.0), 0.0f, 8.0f);

	accum_enable_ = params["accum_enable"].get<bool>(false);
	{
		const std::string st = params["accum_stage"].get<std::string>("post");
		accum_input_ = (st == "input");
		accum_post_ = (st != "pre");
	}
	accum_trust_ = params["accum_trust"].get<double>(0.9);
	accum_k_z_ = params["accum_k_z"].get<double>(10.0);
	accum_motion_ = params["accum_motion"].get<bool>(true);
	accum_sigma_scale_ = params["accum_sigma_scale"].get<double>(1.0);
	accum_update_first_ = params["accum_update_first"].get<bool>(false);
	feed_blend_ = params["feed_blend"].get<bool>(true);
	net_pipeline_ = params["net_pipeline"].get<bool>(false);
	fused_pack_ = params["fused_pack"].get<bool>(false);
	probes_ = params["probes"].get<bool>(false);
	LOG(RPiModelDenoise, Info) << "fused_pack=" << fused_pack_;
	LOG(RPiModelDenoise, Info) << "net_pipeline=" << net_pipeline_;
	accum_motion_ratio_ = params["accum_motion_ratio"].get<double>(0.90);
	accum_motion_agree_ = params["accum_motion_agree"].get<double>(0.60);
	accum_reset_frac_ = params["accum_reset_frac"].get<double>(0.5);
	net_min_gain_ = params["net_min_gain"].get<double>(0.0);
	feed_lo_ = clampf(params["feed_lo"].get<double>(-0.25), -4.0f, 0.0f);
	accum_.setParams(accum_trust_, accum_k_z_);
	/*
	 * Per-pixel motion gate, in sigma. 0 = off, which is the shipped default
	 * because it changes the picture and has not been measured on a moving
	 * scene yet. PiSP's own TDN gates per pixel this way; our block gate
	 * decides regions reliably but cannot localise inside a 16x16 block,
	 * which is what a small moving object needs.
	 */
	accum_.setPixelGate(clampf(params["accum_pixel_k"].get<double>(0.0), 0.0f, 16.0f));
	accum_.setRamp(params["accum_ramp"].get<int>(1) != 0);
	accum_.setMotion(accum_motion_);
	accum_.setMotionAccept(accum_motion_ratio_, accum_motion_agree_);
	accum_.setSigmaScale(accum_sigma_scale_);
	for (auto const &[key, value] : params["noise_profiles"].asDict()) {
		std::array<float, 4> av{}, bv{};
		unsigned na = 0, nb = 0;
		for (auto const &v : value["a"].asList())
			if (na < 4)
				av[na++] = float(v.get<double>(0.0));
		for (auto const &v : value["b"].asList())
			if (nb < 4)
				bv[nb++] = float(v.get<double>(0.0));
		if (na != 4 || nb != 4) {
			LOG(RPiModelDenoise, Error)
				<< "noise_profiles[" << key << "] needs 4 a and 4 b";
			continue;
		}
		noise_a_[std::atoi(key.c_str())] = av;
		noise_b_[std::atoi(key.c_str())] = bv;
	}
	if (accum_enable_ && noise_a_.empty()) {
		LOG(RPiModelDenoise, Error)
			<< "accum_enable with no noise_profiles -- the gate has no sigma "
			   "to compare against; disabling temporal accumulation";
		accum_enable_ = false;
	}
	if (accum_enable_ && !fpn_full_)
		LOG(RPiModelDenoise, Warning)
			<< "accum_enable without fpn_full: the accumulator sees "
			   "uncorrected fixed pattern, which caps averaging near N=21";
	black_default_ = params["black_level"].get<float>(3200.0f);
	white_default_ = params["white_level"].get<float>(65535.0f);
	gain_default_ = params["gain_default"].get<float>(512.0f);
	bnorm_target_ = params["bnorm_target"].get<float>(0.115f);
	bnorm_max_ = params["bnorm_max"].get<float>(32.0f);

	if (infer_w_ < 8 || infer_h_ < 8 || (infer_w_ & 1) || (infer_h_ & 1)) {
		LOG(RPiModelDenoise, Error) << "infer size must be even and >= 8";
		return -EINVAL;
	}
	if (temporal_ < 1 || temporal_ > 8) {
		LOG(RPiModelDenoise, Error) << "temporal must be 1..8";
		return -EINVAL;
	}

	return 0;
}

void ModelDenoise::resetTiles(unsigned tiles)
{
	const unsigned plane4 = infer_h_ * infer_w_ * 4;
	rings_.assign(tiles, std::vector<std::vector<FeedT>>(
				     temporal_, std::vector<FeedT>(plane4, FeedT(0.0f))));
	ring_pos_ = 0;
	ring_filled_ = 0;
	feed_chw_.assign(feedCh() * infer_h_ * infer_w_, FeedT(0.0f));
	/* buildFeed caches the constant gain plane across frames; this wipes it. */
	gplane_valid_ = false;
}

void ModelDenoise::initialise()
{
	if (param_.empty() || bin_.empty()) {
		LOG(RPiModelDenoise, Error) << "No model param/bin supplied";
		return;
	}

	net_.opt.num_threads = std::max(1, threads_);
	/*
	 * ncnn threads elementwise work over CHANNELS, and with the packing
	 * layout a 4-channel full-res tensor becomes elempack=4 with Mat.c == 1
	 * -- so every full-resolution layer runs on ONE thread no matter what
	 * num_threads says (measured on h2: 1188 ms at threads=1 vs 1124 ms at
	 * threads=4). Turning packing off keeps c == 4 so the work can spread,
	 * at the cost of the packed SIMD path. Worth it only for graphs with a
	 * long un-fused full-res tail; m9's tail is two custom layers that do
	 * their own row-parallel threading, so it should stay ON there.
	 */
	if (no_packing_)
		net_.opt.use_packing_layout = false;
	net_.opt.blob_allocator = &blobPool_;
	net_.opt.workspace_allocator = &workPool_;
	net_.opt.use_vulkan_compute = false;
	/*
	 * fp16 on the A76 CPU: 182 ms -> 122 ms at 968x548, max output delta
	 * 0.075/255 (84 dB vs fp32). The old unconditional false came from the
	 * Vulkan V3D path (no fp16 ALU there), not from the CPU.
	 */
	net_.opt.use_fp16_arithmetic = fp16_;
	net_.opt.use_fp16_packed = fp16_;
	net_.opt.use_fp16_storage = fp16_;

	/*
	 * Nets exported with the fused tail name a BilateralAssemble layer,
	 * which is not an ncnn builtin. Registering it unconditionally is
	 * harmless for the unfused models -- they never reference it.
	 */
	net_.register_custom_layer("BilateralAssemble", BilateralAssemble_layer_creator);
	net_.register_custom_layer("GuideBlur5", GuideBlur5_layer_creator);
	net_.register_custom_layer("KpnAssembleLR", KpnAssembleLR_layer_creator);
	net_.register_custom_layer("LfLocks", LfLocks_layer_creator);
	net_.register_custom_layer("BoxBlurR", BoxBlurR_layer_creator);
	net_.register_custom_layer("InputShrink", InputShrink_layer_creator);
	net_.register_custom_layer("BilateralAssembleLR",
				   BilateralAssembleLR_layer_creator);

	/*
	 * BSTM offload. The spliced param carries a BstmTrunk layer in place of
	 * the 53 predictor layers, and that layer creates its NPU context in
	 * create_pipeline() -- which ncnn runs from load_model(). So a failure
	 * to reach the NPU surfaces as a load_model() error, and the recovery
	 * is to drop the whole net and load the all-ncnn param instead. Do NOT
	 * try to keep the half-loaded net: its BstmTrunk would return -1 every
	 * frame and the output would be an uninitialised blob.
	 */
	std::string param = param_;
	bstm_active_ = false;
	bstm_full_active_ = false;
#ifdef RPI_HAVE_BSTM
	/*
	 * Whole-model path first: if it loads there is no ncnn graph to build at
	 * all. Deliberately silent-and-fall-back rather than fatal -- a wrong
	 * tile size or a missing .bstm should degrade to the CPU pipeline, not
	 * take the camera down.
	 */
	if (bstm_full_enable_ && !bstm_full_model_.empty()) {
		const bool ok = bstm_s2d_block_
					? bstm_full_.loadS2D(bstm_full_model_, feedCh(),
							     infer_w_, infer_h_,
							     bstm_s2d_block_, 4u)
					: bstm_full_.load(bstm_full_model_, feedCh(),
							  infer_w_, infer_h_, 4u);
		if (ok) {
			bstm_full_active_ = true;
			bstm_full_.setThreads(std::max(threads_, 1));
			LOG(RPiModelDenoise, Info)
				<< "BSTM FULL model active: " << bstm_full_model_
				<< " tile=" << infer_w_ << "x" << infer_h_
				<< " in_ch=" << feedCh()
				<< (bstm_s2d_block_
					    ? " s2d block=" + std::to_string(bstm_s2d_block_)
					      + " grid=" + std::to_string(infer_w_ / bstm_s2d_block_)
					      + "x" + std::to_string(infer_h_ / bstm_s2d_block_)
					      + " uncovered_rows="
					      + std::to_string(bstm_full_.uncoveredRows())
					    : std::string());
		} else {
			LOG(RPiModelDenoise, Warning)
				<< "BSTM full model failed to load (tile must match the"
				<< " compiled " << infer_w_ << "x" << infer_h_ << "x"
				<< feedCh() << "); falling back";
		}
	}
	if (!bstm_full_active_ && bstm_enable_ && !bstm_param_.empty() &&
	    !bstm_model_.empty()) {
		bstm_cfg_.model_path = bstm_model_;
		net_.register_custom_layer("BstmTrunk", BstmTrunk_layer_creator,
					   BstmTrunk_layer_destroyer, &bstm_cfg_);
		if (net_.load_param(bstm_param_.c_str()) == 0 &&
		    net_.load_model(bin_.c_str()) == 0) {
			bstm_active_ = true;
			LOG(RPiModelDenoise, Info)
				<< "BSTM trunk active: " << bstm_param_
				<< " model=" << bstm_model_;
		} else {
			LOG(RPiModelDenoise, Warning)
				<< "BSTM offload unavailable (is bstorm-daemon running?),"
				<< " falling back to CPU ncnn: " << param_;
			net_.clear();
			net_.opt.blob_allocator = &blobPool_;
			net_.opt.workspace_allocator = &workPool_;
			net_.opt.use_vulkan_compute = false;
			net_.opt.use_fp16_arithmetic = fp16_;
			net_.opt.use_fp16_packed = fp16_;
			net_.opt.use_fp16_storage = fp16_;
			net_.register_custom_layer("BilateralAssemble", BilateralAssemble_layer_creator);
			net_.register_custom_layer("GuideBlur5", GuideBlur5_layer_creator);
			net_.register_custom_layer("KpnAssembleLR", KpnAssembleLR_layer_creator);
			net_.register_custom_layer("LfLocks", LfLocks_layer_creator);
			net_.register_custom_layer("BoxBlurR", BoxBlurR_layer_creator);
			net_.register_custom_layer("InputShrink", InputShrink_layer_creator);
			net_.register_custom_layer("BilateralAssembleLR",
						   BilateralAssembleLR_layer_creator);
		}
	}
#else
	if (bstm_enable_)
		LOG(RPiModelDenoise, Warning)
			<< "bstm requested but libcamera was built without BSTM support";
#endif

	if (!bstm_full_active_ && !bstm_active_ &&
	    (net_.load_param(param.c_str()) != 0 || net_.load_model(bin_.c_str()) != 0)) {
		LOG(RPiModelDenoise, Error) << "Failed to load NCNN model: " << param;
		return;
	}

	/* Single-tile placeholder until switchMode lays out demo tiles. */
	tiles_ = { { 0, 0 } };
	tiles_x_ = 1;
	tiles_y_ = 1;
	resetTiles(1);

	LOG(RPiModelDenoise, Info) << "Loaded bilateral NCNN " << infer_w_ << "x" << infer_h_
				   << " T=" << temporal_ << " blobs=" << kInputBlob << "/"
				   << kOutputBlob << " param="
				   << (bstm_active_ ? bstm_param_ : param_)
				   << " trunk=" << (bstm_full_active_ ? "BSTM-FULL"
						     : bstm_active_ ? "BSTM" : "CPU");
	init_ = true;
}

ModelDenoise::~ModelDenoise()
{
	joinPipeline();
}

void ModelDenoise::joinPipeline()
{
	/*
	 * Any in-flight inference references this object's payloads and the
	 * previous frame's buffers, so it MUST be joined before the geometry
	 * changes or the object dies.
	 */
	if (net_future_.valid())
		net_future_.wait();
	if (accum_future_.valid())
		accum_future_.wait();
	pending_valid_ = false;
}

void ModelDenoise::switchMode(CameraMode const &cameraMode, [[maybe_unused]] Metadata *metadata)
{
	/* an in-flight inference points at the OLD geometry's payloads */
	joinPipeline();
	width_ = cameraMode.width;
	height_ = cameraMode.height;
	n_frames_ = 0;

	/*
	 * Prefer stride-2 CFA packing (matches bilateral training). Only fall
	 * back to pack_step=1 when the Bayer frame cannot fit the infer window.
	 * Demo must stay on pack_step=2 — forcing step=1 feeds the net at the
	 * wrong spatial scale and paints mushy blur.
	 */
	pack_step_ = pack_step_cfg_ ? pack_step_cfg_ : 2;
	unsigned bstep = 2u * pack_step_;
	if (height_ / bstep < infer_h_ || width_ / bstep < infer_w_) {
		pack_step_ = 1;
		bstep = 2u * pack_step_;
	}

	if (height_ / bstep < infer_h_ || width_ / bstep < infer_w_) {
		LOG(RPiModelDenoise, Error)
			<< "Bayer " << width_ << "x" << height_
			<< " too small for infer " << infer_w_ << "x" << infer_h_;
		pack_step_ = 0;
		tiles_.clear();
		tiles_x_ = tiles_y_ = 0;
		cover_bw_ = cover_bh_ = 0;
		return;
	}

	const unsigned ph = height_ / bstep;
	const unsigned pw = width_ / bstep;
	/* Demo A/B: write only the left half of each packed tile (hailo-style). */
	const unsigned write_w = demo_ ? std::max(1u, infer_w_ / 2u) : infer_w_;
	tiles_.clear();

	if (demo_) {
		/*
		 * Full-height left-half coverage at pack_step=2. One tile spans
		 * nearly the full Bayer width when packed (480×4), so we keep a
		 * single horizontal tile and only unpack write_w columns
		 * (~960 Bayer ≈ left 50%). Stack tiles vertically for height.
		 */
		const unsigned cover_bh = height_ & ~1u;
		const unsigned cover_ph = cover_bh / bstep;

		tiles_x_ = 1;
		/*
		 * Ceil-tiling buys a whole extra full-size NCNN forward to cover a
		 * tiny remainder: cover_ph 550 vs infer_h 548 gives 2 tiles whose
		 * crops differ by 2 rows, i.e. 546/548 rows are computed twice for
		 * 0.36% more coverage. Only add the tile if it gains >5% of a tile.
		 */
		tiles_y_ = std::max(1u, cover_ph / infer_h_);
		if (cover_ph > tiles_y_ * infer_h_ &&
		    (cover_ph - tiles_y_ * infer_h_) * 20u > infer_h_)
			tiles_y_ += 1u;

		for (unsigned ty = 0; ty < tiles_y_; ++ty) {
			unsigned y0 = ty * infer_h_;
			if (ty + 1u == tiles_y_)
				y0 = cover_ph > infer_h_ ? cover_ph - infer_h_ : 0u;
			if (y0 + infer_h_ > ph)
				y0 = ph - infer_h_;
			/* Left-aligned crop; pack full infer_w_ for context. */
			tiles_.push_back({ 0u, y0 });
		}

		unsigned y_max = 0;
		for (const Tile &t : tiles_)
			y_max = std::max(y_max, (t.crop_y0 + infer_h_) * bstep);
		cover_bw_ = std::min(width_, write_w * bstep);
		cover_bh_ = std::min(height_, y_max);
	} else {
		/*
		 * 2D GRID. This used to be a single centred crop, which is correct
		 * only when infer_* already spans the frame -- as it did at
		 * 968x548. Small tiles are the whole point on BSTM: a full-res
		 * 4-channel bf16 tensor is ~1 MB against a 2 MB on-chip
		 * scratchpad, so one big inference spills every layer to DRAM.
		 * Measured on this Pi, 128x64 tiles move 74 bytes/px where
		 * 484x274 moves 876 -- an order of magnitude less traffic for the
		 * same pixels. So cover the frame with a grid instead.
		 *
		 * The >5% remainder rule is the vertical path's, kept verbatim:
		 * a whole extra full-size forward to cover a sliver is a bad
		 * trade, so only add the row/column if it earns more than 5% of a
		 * tile. The last tile on each axis is pulled back to fit, which
		 * overlaps its neighbour -- overlap is recomputed, never missing.
		 */
		auto count = [](unsigned span, unsigned tile) {
			unsigned n = std::max(1u, span / tile);
			if (span > n * tile && (span - n * tile) * 20u > tile)
				n += 1u;
			return n;
		};
		tiles_x_ = count(pw, infer_w_);
		tiles_y_ = count(ph, infer_h_);
		for (unsigned ty = 0; ty < tiles_y_; ++ty) {
			unsigned y0 = (ty + 1u == tiles_y_ && ph > infer_h_)
					      ? ph - infer_h_
					      : ty * infer_h_;
			if (y0 + infer_h_ > ph)
				y0 = ph > infer_h_ ? ph - infer_h_ : 0u;
			for (unsigned tx = 0; tx < tiles_x_; ++tx) {
				unsigned x0 = (tx + 1u == tiles_x_ && pw > infer_w_)
						      ? pw - infer_w_
						      : tx * infer_w_;
				if (x0 + infer_w_ > pw)
					x0 = pw > infer_w_ ? pw - infer_w_ : 0u;
				tiles_.push_back({ x0, y0 });
			}
		}
		unsigned x_max = 0, y_max = 0;
		for (const Tile &t : tiles_) {
			x_max = std::max(x_max, (t.crop_x0 + infer_w_) * bstep);
			y_max = std::max(y_max, (t.crop_y0 + infer_h_) * bstep);
		}
		cover_bw_ = std::min(width_, x_max);
		cover_bh_ = std::min(height_, y_max);
	}

	resetTiles(tiles_.size());
	/* History belongs to the OLD stream: different exposure, possibly a
	 * different crop. Keeping it means the first frames of the new mode
	 * are blended against, and guided by, a picture that is no longer
	 * true. Prime fresh from the first frame of the new mode instead. */
	accum_.reset();
	accum_gain_ = 0;

	const Tile &t0 = tiles_.front();
	LOG(RPiModelDenoise, Info) << "Mode " << width_ << "x" << height_
				   << " pack_step=" << pack_step_
				   << " tiles=" << tiles_x_ << "x" << tiles_y_
				   << " (" << tiles_.size() << ")"
				   << " cover_bayer=" << cover_bw_ << "x" << cover_bh_
				   << " crop0=(" << t0.crop_x0 << "," << t0.crop_y0 << ")"
				   << " blend=" << blend_
				   << (demo_ ? " demo" : "");
}

void ModelDenoise::setMode([[maybe_unused]] DenoiseMode mode)
{
}

namespace {

/*
 * Snap a continuous gain to the calibrated ladder, WITH HYSTERESIS.
 *
 * Plain nearest-in-log flips at the midpoint, and AGC hunting sits right on
 * midpoints. Each flip re-reads an 8.5 MB DSNU map from disk and (before this)
 * discarded the temporal accumulator, so a camera pointed at a scene near
 * gain ~360 oscillated between 19 and 9 FPS and never accumulated anything.
 *
 * Requiring the new rung to be better by a margin means the choice only moves
 * when the gain has genuinely committed to it.
 */
int snapGainLadder(float gain, int current)
{
	/*
	 * The ladder must cover every gain the sensor can actually apply. It used
	 * to stop at 512, so anything above snapped back to 512 and the
	 * accumulator kept the gain-512 noise model while the real noise went on
	 * growing. sigma was then UNDER-estimated, the gate read ordinary noise as
	 * motion, and temporal averaging switched itself off entirely -- measured
	 * at --gain 2048: 99.99% of blocks gated, meanN 1.00 (i.e. no averaging at
	 * all), mean|diff|/sigma 1.86 against the 0.798 that pure noise must give.
	 */
	static const int kGains[] = { 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048 };
	constexpr double kHysteresisOctaves = 0.25;

	const double lg = std::log2(std::max(double(gain), 1e-6));
	int best = kGains[0];
	double bd = 1e18;
	for (int g : kGains) {
		const double d = std::fabs(lg - std::log2(double(g)));
		if (d < bd) { bd = d; best = g; }
	}
	if (current > 0 && best != current) {
		const double dCur = std::fabs(lg - std::log2(double(current)));
		if (dCur - bd < kHysteresisOctaves)
			return current;   /* not decisively better -- stay put */
	}
	return best;
}

} /* namespace */

void ModelDenoise::ensureFpn(float gain)
{
	if (!calib_enabled_) {
		fpn_active_ = false;
		return;
	}
	const int best = snapGainLadder(gain, fpn_gain_);
	if (best == fpn_gain_)
		return;

	const std::string path = calib_dir_ + "/fpn_gain" + std::to_string(best) + ".bin";
	FILE *f = std::fopen(path.c_str(), "rb");
	if (!f) {
		LOG(RPiModelDenoise, Warning) << "no DSNU map at " << path << " (calib off)";
		fpn_active_ = false;
		fpn_gain_ = best;
		return;
	}
	int32_t hdr[4] = { 0, 0, 0, 0 };
	float sc = 0.0f;
	bool ok = std::fread(hdr, sizeof(int32_t), 4, f) == 4 &&
		  std::fread(&sc, sizeof(float), 1, f) == 1 &&
		  hdr[0] > 0 && hdr[1] > 0 && hdr[2] == 4;
	if (ok) {
		const size_t n = size_t(hdr[0]) * hdr[1] * 4;
		fpn_.resize(n);
		ok = std::fread(fpn_.data(), sizeof(float), n, f) == n;
		if (ok) {
			fpn_h_ = unsigned(hdr[0]);
			fpn_w_ = unsigned(hdr[1]);
			fpn_scale_ = clampf(sc * fpn_strength_, 0.0f, 1.0f);
		}
	}
	std::fclose(f);
	fpn_active_ = ok;
	fpn_gain_ = best;
	LOG(RPiModelDenoise, Info) << (ok ? "DSNU map loaded " : "DSNU map FAILED ")
				   << path << " " << fpn_h_ << "x" << fpn_w_
				   << " alpha*=" << sc << " x" << fpn_strength_
				   << " -> " << fpn_scale_ << " for gain " << gain;
}

/*
 * Subtract the DSNU map across the entire Bayer frame.
 *
 * packTile does its own subtraction, but only at the pixels the model samples
 * -- with pack_step=2 that is a quarter of them. Temporal accumulation reads
 * and writes every pixel, and fixed pattern is exactly what it cannot average
 * away: at gain 512, 81% of the residual variance at N=128 is sensor-locked,
 * so uncorrected accumulation saturates near N=21 frames however long it runs.
 *
 * When this is active packTile skips its own subtraction, so the values the
 * model sees are the same ones it would have seen -- only the rounding to
 * uint16 differs.
 */
void ModelDenoise::applyFpnFull(uint16_t *bayer16, unsigned stridePx, float black,
				float white)
{
	if (!fpn_active_ || !fpn_full_ || fpn_.empty())
		return;

	const float k = fpn_scale_ * std::max(white - black, 1.0f);
	const unsigned rows = std::min(fpn_h_, height_ / 2);
	unsigned cols = std::min(fpn_w_, width_ / 2);
	if (demo_)
		cols = std::min(cols, width_ / 4);  /* left half only, in packed cols */
	auto band = [&](unsigned y0, unsigned y1) {
	for (unsigned fy = y0; fy < y1; ++fy) {
		uint16_t *r0 = bayer16 + size_t(2 * fy) * stridePx;
		uint16_t *r1 = r0 + stridePx;
		const float *fp = fpn_.data() + size_t(fy) * fpn_w_ * 4;
		for (unsigned fx = 0; fx < cols; ++fx, fp += 4) {
			uint16_t *a = r0 + 2 * fx, *b = r1 + 2 * fx;
			a[0] = uint16_t(std::clamp(int(std::lround(float(a[0]) - k * fp[0])), 0, 65535));
			a[1] = uint16_t(std::clamp(int(std::lround(float(a[1]) - k * fp[1])), 0, 65535));
			b[0] = uint16_t(std::clamp(int(std::lround(float(b[0]) - k * fp[2])), 0, 65535));
			b[1] = uint16_t(std::clamp(int(std::lround(float(b[1]) - k * fp[3])), 0, 65535));
		}
	}
	};

	/* Row-disjoint, and it was 8.8 ms single-threaded on the Pi. */
	const unsigned nt = unsigned(std::clamp(threads_, 1, 8));
	if (nt <= 1) {
		band(0, rows);
	} else {
		std::vector<std::thread> pool;
		for (unsigned i = 0; i < nt; ++i)
			pool.emplace_back(band, rows * i / nt, rows * (i + 1) / nt);
		for (auto &t : pool)
			t.join();
	}
}

/*
 * As applyAccum, but by default updates memory ONLY -- the model consumes the
 * accumulator state as slot 0, so writing it back here would hand the model its
 * own history as the live frame. writeBack=true is the gain-gate path, where
 * there is no model run to carry memory into the image and the accumulator has
 * to deliver it directly.
 */
void ModelDenoise::applyAccumInput(uint16_t *bayer16, unsigned stridePx, float black, float gain,
				   bool writeBack, unsigned writeCols)
{
	/*
	 * accum_enable_ used to only gate applyAccum() (accum_stage post/pre),
	 * not this function -- so with accum_stage="input", "accum_enable":
	 * false was silently ignored and the accumulator kept running as long
	 * as noise_profiles was non-empty. Check it here too.
	 *
	 * NOTE: with accum_enable_ false, accum_.state() is never updated past
	 * whatever primed it (see the fallback prime in prepare()), so the
	 * model's memory slot goes stale rather than tracking the scene. This
	 * is fine for isolating whether temporal ghosting comes from the
	 * accumulator, but is not a general-purpose "run without an
	 * accumulator" mode -- accum_stage="input" feeds the model
	 * [memory, current] by design, and disabling the update leaves
	 * "memory" frozen rather than absent.
	 */
	if (!accum_enable_ || noise_a_.empty())
		return;
	const int best = snapGainLadder(gain, accum_gain_);
	if (best != accum_gain_ && noise_a_.count(best)) {
		accum_.setNoise(noise_a_[best].data(), noise_b_[best].data());
		accum_gain_ = best;
	}
	const float gated = accum_.apply(bayer16, stridePx, width_, height_, black,
					 white_default_, unsigned(std::max(threads_, 1)),
					 writeBack, writeCols);
	last_gated_ = gated;
	if (gated > accum_reset_frac_)
		accum_.reset();
}

/*
 * Recursive temporal accumulation, after the model has written its output
 * back. "post" keeps the model's input exactly the 4-frame ring it was trained
 * on; "pre" gates against the sigma that actually describes the data but hands
 * the model frames it never saw in training. Neither is settled -- it needs an
 * A/B on real footage.
 */
void ModelDenoise::applyAccum(uint16_t *bayer16, unsigned stridePx, float black, float gain)
{
	if (!accum_enable_ || noise_a_.empty())
		return;

	const int best = snapGainLadder(gain, accum_gain_);
	if (best != accum_gain_ && noise_a_.count(best)) {
		/*
		 * Update sigma, but do NOT discard history. Resetting here was
		 * wrong: sigma changes smoothly with gain, and if the exposure
		 * really did jump then every block disagrees at once and the
		 * gate's own scene-cut test resets it a frame later anyway.
		 * Resetting on every ladder change meant AGC hunting kept the
		 * accumulator permanently empty.
		 */
		accum_.setNoise(noise_a_[best].data(), noise_b_[best].data());
		accum_gain_ = best;
	}

	const float gated = accum_.apply(bayer16, stridePx, width_, height_, black,
					 white_default_, unsigned(std::max(threads_, 1)),
					 true, demo_ ? width_ / 2u : 0u);
	if (gated > accum_reset_frac_) {
		LOG(RPiModelDenoise, Debug)
			<< "temporal reset: " << (100.0f * gated) << "% of blocks gated";
		accum_.reset();
	}
}

bool ModelDenoise::packTile(unsigned tile, const uint16_t *raw, unsigned stridePx, float black,
			    float white)
{
	return packTileSlot(tile, ring_pos_, raw, stridePx, black, white);
}

/*
 * Memory -> ring slot. Same crop/step geometry as packTileSlot, but the source
 * is the accumulator: float, DN, full-resolution Bayer layout. Kept separate
 * rather than templated so the uint16 path stays byte-identical.
 */
bool ModelDenoise::packMemory(unsigned tile, unsigned slot, AccT const *mem,
			      unsigned memStride, float black, float white)
{
	if (!pack_step_ || tile >= tiles_.size() || !mem || slot >= rings_[tile].size())
		return false;

	const unsigned bstep = 2u * pack_step_;
	/* bnorm folded in; see bnorm_pre_. Exact: for bs >= 1 and shared
	 * bounds, clamp(bs*clamp(u,lo,hi),lo,hi) == clamp(bs*u,lo,hi). */
	const float inv = bnorm_pre_ / std::max(white - black, 1.0f);
	const unsigned crop_x0 = tiles_[tile].crop_x0;
	const unsigned crop_y0 = tiles_[tile].crop_y0;
	const unsigned plane = infer_h_ * infer_w_;
	FeedT *p0 = slotPtr(tile, slot), *p1 = p0 + plane, *p2 = p1 + plane,
	      *p3 = p2 + plane;

#ifdef _OPENMP
	#pragma omp parallel for num_threads(std::max(threads_, 1)) schedule(static)
#endif
	for (int y_ = 0; y_ < int(infer_h_); ++y_) {
		const unsigned y = unsigned(y_);
		for (unsigned x = 0; x < infer_w_; ++x) {
			const size_t o = size_t(y) * infer_w_ + x;
			const unsigned sy = (crop_y0 + y) * bstep;
			const unsigned sx = (crop_x0 + x) * bstep;
			if (sy + 1 >= height_ || sx + 1 >= width_) {
				p0[o] = p1[o] = p2[o] = p3[o] = 0.0f;
				continue;
			}
			auto const *r0 = mem + size_t(sy) * memStride + sx;
			auto const *r1 = mem + size_t(sy + 1) * memStride + sx;
			p0[o] = clampf((float(r0[0]) - black) * inv, -0.25f, 1.0f);
			p1[o] = clampf((float(r0[1]) - black) * inv, -0.25f, 1.0f);
			p2[o] = clampf((float(r1[0]) - black) * inv, -0.25f, 1.0f);
			p3[o] = clampf((float(r1[1]) - black) * inv, -0.25f, 1.0f);
		}
	}
	return true;
}

bool ModelDenoise::packTileSlot(unsigned tile, unsigned slot, const uint16_t *raw,
				unsigned stridePx, float black, float white)
{
	if (!pack_step_ || tile >= tiles_.size() || slot >= rings_[tile].size())
		return false;

	const unsigned bstep = 2u * pack_step_;
	/* bnorm folded in; see bnorm_pre_. Exact: for bs >= 1 and shared
	 * bounds, clamp(bs*clamp(u,lo,hi),lo,hi) == clamp(bs*u,lo,hi). */
	const float inv = bnorm_pre_ / std::max(white - black, 1.0f);
	const unsigned crop_x0 = tiles_[tile].crop_x0;
	const unsigned crop_y0 = tiles_[tile].crop_y0;

	/* planar ring: store as 4 contiguous planes so buildFeed is a memcpy. */
	const unsigned plane = infer_h_ * infer_w_;
	FeedT *p0 = slotPtr(tile, slot);
	FeedT *p1 = p0 + plane;
	FeedT *p2 = p1 + plane;
	FeedT *p3 = p2 + plane;
	/* nb=2 averages the 4 CFA blocks that pack_step=2 spans (binning);
	 * nb=1 is the plain one-block read. */
	const unsigned nb = (bin2x2_ && pack_step_ >= 2) ? 2u : 1u;
	const float binw = 1.0f / float(nb * nb);
	/*
	 * Fast path for the shipped geometry: no binning, DSNU already removed
	 * full-frame, and a 2-pixel CFA step. That is a stride-2 uint16 read,
	 * which is exactly vld2q_u16 -- eight output pixels per iteration with
	 * no gather and no per-pixel bounds test.
	 */
#if defined(__ARM_NEON) && defined(__ARM_FP16_FORMAT_IEEE)
	const bool simple = (nb == 1u) && bstep == 2u &&
			    !(fpn_active_ && !fpn_full_);
	const float32x4_t vblack = vdupq_n_f32(black);
	const float32x4_t vinv = vdupq_n_f32(inv);
	const float32x4_t vlo = vdupq_n_f32(-0.25f);
	const float32x4_t vhi = vdupq_n_f32(1.0f);
#else
	const bool simple = false;
#endif

#ifdef _OPENMP
	#pragma omp parallel for num_threads(std::max(threads_, 1)) schedule(static)
#endif
	for (int y_ = 0; y_ < int(infer_h_); ++y_) {
		const unsigned y = unsigned(y_);
		unsigned x = 0;
#if defined(__ARM_NEON) && defined(__ARM_FP16_FORMAT_IEEE)
		if (simple) {
			const unsigned sy = (crop_y0 + y) * bstep;
			if (sy + 1 < height_) {
				const uint16_t *r0 = raw + size_t(sy) * stridePx
						     + crop_x0 * bstep;
				const uint16_t *r1 = r0 + stridePx;
				/* largest x whose sx+1 is still inside the frame */
				unsigned xlim = infer_w_;
				while (xlim > 0 &&
				       (crop_x0 + xlim - 1) * bstep + 1 >= width_)
					--xlim;
				const size_t row = size_t(y) * infer_w_;
				for (; x + 8 <= xlim; x += 8) {
					const uint16x8x2_t a =
						vld2q_u16(r0 + size_t(x) * 2);
					const uint16x8x2_t b =
						vld2q_u16(r1 + size_t(x) * 2);
					packStore8(p0 + row + x, a.val[0], vblack,
						   vinv, vlo, vhi);
					packStore8(p1 + row + x, a.val[1], vblack,
						   vinv, vlo, vhi);
					packStore8(p2 + row + x, b.val[0], vblack,
						   vinv, vlo, vhi);
					packStore8(p3 + row + x, b.val[1], vblack,
						   vinv, vlo, vhi);
				}
			}
		}
#endif
		for (; x < infer_w_; ++x) {
			const size_t o = size_t(y) * infer_w_ + x;
			float ar = 0.0f, ag1 = 0.0f, ag2 = 0.0f, ab = 0.0f;
			for (unsigned dy = 0; dy < nb; ++dy) {
				for (unsigned dx = 0; dx < nb; ++dx) {
					const unsigned sy = (crop_y0 + y) * bstep + dy * 2u;
					const unsigned sx = (crop_x0 + x) * bstep + dx * 2u;
					if (sy + 1 >= height_ || sx + 1 >= width_)
						continue;
					const uint16_t *row0 = raw + sy * stridePx + sx;
					const uint16_t *row1 = raw + (sy + 1) * stridePx + sx;
					float vr = (float(row0[0]) - black) * inv;
					float vg1 = (float(row0[1]) - black) * inv;
					float vg2 = (float(row1[0]) - black) * inv;
					float vb = (float(row1[1]) - black) * inv;
					if (fpn_active_ && !fpn_full_) {
						/* DSNU is per-pixel: subtract before averaging */
						const unsigned fy = (crop_y0 + y) * pack_step_ + dy;
						const unsigned fx = (crop_x0 + x) * pack_step_ + dx;
						if (fy < fpn_h_ && fx < fpn_w_) {
							const float *fp = fpn_.data() +
								(size_t(fy) * fpn_w_ + fx) * 4;
							vr -= fpn_scale_ * fp[0];
							vg1 -= fpn_scale_ * fp[1];
							vg2 -= fpn_scale_ * fp[2];
							vb -= fpn_scale_ * fp[3];
						}
					}
					ar += vr; ag1 += vg1; ag2 += vg2; ab += vb;
				}
			}
			p0[o] = clampf(ar * binw, -0.25f, 1.0f);  /* R */
			p1[o] = clampf(ag1 * binw, -0.25f, 1.0f); /* G1 */
			p2[o] = clampf(ag2 * binw, -0.25f, 1.0f); /* G2 */
			p3[o] = clampf(ab * binw, -0.25f, 1.0f);  /* B */
		}
	}
	return true;
}

namespace {

/* Per-gain means of noise_model.heteroscedastic.{a,b} (imx662h, opt_imx662). */
struct AbRow { int gain; float a; float b; };
const AbRow kAbTable[] = {
	{   4, 0.00063367f, 0.00000032f}, {   8, 0.00131243f, 0.00000126f},
	{  12, 0.00193779f, 0.00000218f}, {  16, 0.00269128f, 0.00000005f},
	{  24, 0.00402081f, 0.00000783f}, {  32, 0.00526220f, 0.00001190f},
	{  48, 0.00806187f, 0.00004374f}, {  64, 0.01083641f, 0.00004039f},
	{  96, 0.01607404f, 0.00016150f}, { 128, 0.02119305f, 0.00028296f},
	{ 192, 0.03170913f, 0.00081509f}, { 256, 0.04232473f, 0.00132535f},
	{ 384, 0.06684152f, 0.00279391f}, { 512, 0.09315149f, 0.00411326f},
};

/* Nearest gain in log2, matching the training-side lookup. */
void abForGain(float gain, float &av, float &bv)
{
	const double lg = std::log2(std::max(gain, 1.0f));
	double best = 1e18;
	av = kAbTable[0].a;
	bv = kAbTable[0].b;
	for (const AbRow &r : kAbTable) {
		const double d = std::fabs(lg - std::log2(double(r.gain)));
		if (d < best) { best = d; av = r.a; bv = r.b; }
	}
}

} // namespace

/*
 * buildFeed() and the s2d fold, merged.
 *
 * Separately they wrote feed_chw_ (9 planes, ~9.5 MB) and then immediately read
 * it back to scatter into the NPU's block layout -- two traversals of the same
 * 4.7 M elements, 5.8 + 7.9 ms. This does the temporal blend while writing the
 * payload, so the intermediate never exists. The stage is bandwidth-bound
 * (nsa-pipeline-is-bandwidth-bound), so removing a traversal is the whole point.
 *
 * Destination is walked sequentially: a block's CIN floats are contiguous, so
 * the scattering lands on the READ side where it is cheaper.
 */
bool ModelDenoise::foldFeedS2D(unsigned tile, float gain, float *payload)
{
	const unsigned r = bstm_s2d_block_;
	if (!payload || r == 0 || infer_w_ % r || temporal_ == 0)
		return false;
	const unsigned sub = r * r, gw = infer_w_ / r, gh = infer_h_ / r;
	const unsigned plane = infer_h_ * infer_w_;
	const unsigned CIN = feedCh() * sub;

	const float gch = encodeGain(gain, gain_plane_max_);

	/* Identical blend parameters to buildFeed(); see the rationale there. */
	const unsigned newest = (ring_filled_ < temporal_)
					? (ring_filled_ - 1u)
					: ((ring_pos_ + temporal_ - 1u) % temporal_);
	const FeedT *cur = slotPtr(tile, newest);
	float sigma = 0.0f;
	if (temporal_ > 1 && ring_filled_ > 1) {
		const unsigned prev = (newest + temporal_ - 1u) % temporal_;
		const FeedT *pv = slotPtr(tile, prev);
		double acc = 0.0;
		unsigned n = 0;
		for (unsigned i = 0; i < plane * 4u; i += 37u) {
			acc += std::fabs(float(cur[i]) - float(pv[i]));
			++n;
		}
		sigma = n ? float(acc / n) : 0.0f;
	}
	const float lo = 2.5f * sigma, hi = 6.0f * sigma;
	const float inv_span = (hi > lo) ? 1.0f / (hi - lo) : 0.0f;

	const FeedT *slot[8];
	bool blend[8];
	for (unsigned t = 0; t < temporal_; ++t) {
		const unsigned idx = (ring_filled_ < temporal_)
					     ? (t % ring_filled_)
					     : ((ring_pos_ + t) % temporal_);
		slot[t] = slotPtr(tile, idx);
		blend[t] = (idx != newest) && (inv_span != 0.0f);
	}

	for (unsigned by = 0; by < gh; ++by) {
		for (unsigned bx = 0; bx < gw; ++bx) {
			float *d = payload + (size_t(by) * gw + bx) * CIN;
			for (unsigned t = 0; t < temporal_; ++t) {
				for (unsigned c = 0; c < 4u; ++c) {
					const FeedT *sp = slot[t] + size_t(c) * plane;
					const FeedT *cp = cur + size_t(c) * plane;
					for (unsigned dy = 0; dy < r; ++dy) {
						const size_t o = size_t(by * r + dy) * infer_w_
								 + bx * r;
						if (!blend[t]) {
							for (unsigned dx = 0; dx < r; ++dx)
								*d++ = float(sp[o + dx]);
						} else {
							for (unsigned dx = 0; dx < r; ++dx) {
								const float s = float(sp[o + dx]);
								const float k = float(cp[o + dx]);
								float w = 1.0f - (std::fabs(s - k) - lo) * inv_span;
								w = w > 1.0f ? 1.0f : (w < 0.0f ? 0.0f : w);
								*d++ = w * s + (1.0f - w) * k;
							}
						}
					}
				}
			}
			if (gain_plane_)
				for (unsigned i = 0; i < sub; ++i)
					*d++ = gch;
		}
	}
	return true;
}

void ModelDenoise::buildFeed(unsigned tile, float gain)
{
	const unsigned plane = infer_h_ * infer_w_;
	const float gch = encodeGain(gain, gain_plane_max_);
	/*
	 * motion-adaptive temporal blend. The net averages the T frames, so a
	 * moving object is averaged with where it used to be (measured: -4.4 dB
	 * at 12 px/frame, while stacking is worth +5.8 dB when static). Keep the
	 * old sample where the temporal difference is noise-sized; fall back to
	 * the current sample where it is motion. Static pixels are unchanged.
	 */
	const unsigned newest = (ring_filled_ < temporal_)
					? (ring_filled_ - 1u)
					: ((ring_pos_ + temporal_ - 1u) % temporal_);
	const FeedT *cur = slotPtr(tile, newest);

	/* Noise estimate: mean |newest - previous| on a subsampled grid. */
	float sigma = 0.0f;
	/*
	 * ONLY WHEN SOMETHING CONSUMES IT. sigma's only consumers are lo/hi,
	 * which feed inv_span, which is forced to 0 whenever feed_blend_ is
	 * false -- the deployed value, because the accumulator is the memory
	 * slot and already applies a calibrated gate (see inv_span below).
	 * The loop strides i += 37 over plane*4 __fp16, i.e. 74 bytes, so
	 * every iteration touches a fresh 64-byte line of BOTH cur and pv:
	 * it sweeps ~7.3 MB of DRAM per frame to produce a float that is then
	 * discarded. The bytes matter more than the cycles here, because the
	 * CPU and the NPU contend for one bus -- measured on this build, pack
	 * is 22.3 ms while the net overlaps it and 13.1 ms when it runs alone.
	 */
	if (feed_blend_ && temporal_ > 1 && ring_filled_ > 1) {
		const unsigned prev = (newest + temporal_ - 1u) % temporal_;
		const FeedT *pv = slotPtr(tile, prev);
		double acc = 0.0;
		unsigned n = 0;
		for (unsigned i = 0; i < plane * 4u; i += 37u) {
			acc += std::fabs(float(cur[i]) - float(pv[i]));
			++n;
		}
		sigma = n ? float(acc / n) : 0.0f;
	}
	/* Below lo => pure noise, keep history. Above hi => motion, drop it. */
	const float lo = 2.5f * sigma;
	const float hi = 6.0f * sigma;
	/*
	 * REDUNDANT WHEN THE ACCUMULATOR IS THE MEMORY SLOT. With
	 * accum_stage="input", slot 0 is TemporalAccum's own state, which
	 * already applies a calibrated per-block and per-pixel motion gate.
	 * This blend is a second, cruder gate on top -- its sigma is a
	 * subsampled mean|newest-prev| rather than the sensor noise model --
	 * and it costs a full pass over 4 planes with a fabs and two multiplies
	 * per pixel. feed_blend=false skips it and takes the memcpy path.
	 */
	const float inv_span = (feed_blend_ && hi > lo) ? 1.0f / (hi - lo) : 0.0f;

	for (unsigned t = 0; t < temporal_; ++t) {
		unsigned idx;
		if (ring_filled_ < temporal_)
			idx = t % ring_filled_;
		else
			idx = (ring_pos_ + t) % temporal_;
		const FeedT *src = slotPtr(tile, idx);
		FeedT *dst0 = feed_chw_.data() + (t * 4) * plane;
		if (idx == newest || inv_span == 0.0f) {
			/* already there when the pack wrote straight at the feed */
			if (src != dst0)
				for (unsigned c = 0; c < 4; ++c)
					std::memcpy(dst0 + c * plane, src + c * plane,
						    plane * sizeof(FeedT));
			continue;
		}
#ifdef _OPENMP
		#pragma omp parallel for num_threads(std::max(threads_, 1)) schedule(static)
#endif
		for (int c = 0; c < 4; ++c) {
			const FeedT *sp = src + unsigned(c) * plane;
			const FeedT *cp = cur + unsigned(c) * plane;
			FeedT *dp = dst0 + unsigned(c) * plane;
			for (unsigned i = 0; i < plane; ++i) {
				const float d = std::fabs(float(sp[i]) - float(cp[i]));
				float w = 1.0f - (d - lo) * inv_span;
				w = w > 1.0f ? 1.0f : (w < 0.0f ? 0.0f : w);
				dp[i] = FeedT(w * float(sp[i]) + (1.0f - w) * float(cp[i]));
			}
		}
	}
	/*
	 * Brightness-normalise the 16 image planes (never the gain plane). The
	 * net trained at mean ~0.115; a live dark scene sits ~8x lower, which it
	 * answers by smoothing all detail away (measured offline: gradient energy
	 * 0.0242 -> 0.0109). Scaling up to the training level restores it fully
	 * (0.0242). unpackTile() divides the scale back out.
	 */
	/*
	 * Only pay for the mean when something consumes it. This is a serial
	 * double-accumulate over temporal_*4*plane floats -- 4.2M of them at
	 * T=2, ~7 ms -- and with bnorm_target 0.0 (the deployed value) the
	 * result was computed and then discarded on every single frame.
	 */
	/*
	 * Subsampled. The full walk was a SERIAL double-accumulate over
	 * temporal_*4*plane floats -- 4.2M at T=2, ~7 ms a frame -- and its
	 * result feeds a brightness estimate, which does not need every pixel.
	 * Sampling every 37th ROW rather than every 37th element: an element
	 * stride of 37 floats is 148 bytes, which touches every cache line of
	 * the 17 MB buffer and costs the full memory sweep to read 1/37 of the
	 * data. Whole rows are contiguous, so the sample really is 1/37 of the
	 * traffic. 37 is coprime with the 8 planes and with 2, so the sampled
	 * rows sweep planes and Bayer phases rather than landing on one.
	 * ~14k samples still puts the standard error far below the 1% that
	 * matters, and it keeps the STATS bmean readable, which is what
	 * diagnosed the out-of-distribution colour cast at low light.
	 */
	double bsum = 0.0;
	unsigned bn = 0;
	for (unsigned pl = 0; pl < temporal_ * 4u; ++pl) {
		const FeedT *pp = feed_chw_.data() + size_t(pl) * plane;
		for (unsigned r = 0; r < infer_h_; r += 37u) {
			const FeedT *rp = pp + size_t(r) * infer_w_;
			for (unsigned i = 0; i < infer_w_; ++i)
				bsum += float(rp[i]);
			bn += infer_w_;
		}
	}
	const float bmean = bn ? static_cast<float>(bsum / bn) : 0.0f;
	/* DIAG: per-plane means of the PRE-bnorm feed. slot0 = memory,
	 * slot1 = current; if these two disagree the model is being handed an
	 * inconsistent pair and bmean is meaningless. */
	if (probes_ && (n_frames_ % 30ull) == 0ull) {
		std::string ms;
		for (unsigned pl = 0; pl < temporal_ * 4u + 1u; ++pl) {
			const FeedT *pp = feed_chw_.data() + size_t(pl) * plane;
			double s = 0.0;
			for (unsigned i = 0; i < plane; i += 7u)
				s += float(pp[i]);
			ms += " " + std::to_string(s / double((plane + 6u) / 7u));
		}
		LOG(RPiModelDenoise, Info) << "FEEDMEAN bmean=" << bmean << " planes:" << ms;
	}
	float bs = 1.0f;
	if (bnorm_target_ > 0.0f && bmean > 1e-6f)
		/*
		 * Two-sided. The lower bound used to be 1.0, so this could only
		 * ever brighten a frame. Measured on the camera, bmean is
		 * 0.185 in a dark room and 0.631 in a bright one against a
		 * target of 0.115 -- both ABOVE target, so the ratio was
		 * clamped to 1 and no normalisation happened at all, on every
		 * real frame. The model then saw scenes 3-12x brighter than
		 * the ~0.05 mean of its training crops, which is exactly the
		 * regime where the guided filter's a-coefficient is wrong,
		 * because a depends on local variance measured against the
		 * expected noise variance and both scale with signal level.
		 *
		 * Allowing bs < 1 scales a bright frame down onto the target
		 * instead. unpackTile already divides by bnorm_scale_ on the
		 * way out (inv_bn), so the output level is unchanged either
		 * way -- this only moves the model's INPUT into distribution.
		 */
		bs = clampf(bnorm_target_ / (bmean / std::max(bnorm_pre_, 1e-6f)),
			    1.0f / bnorm_max_, bnorm_max_);
	/*
	 * bmean was measured from the ALREADY-SCALED feed, so divide out the
	 * scale pack applied to recover the raw scene mean. bnorm_scale_ is what
	 * the frame in flight actually carries (unpackTile divides by it);
	 * bnorm_pre_ is what the next frame's pack will apply. The elementwise
	 * scale+clamp pass that used to live here is gone.
	 */
	bnorm_scale_ = bnorm_pre_;
	bnorm_pre_ = bs;
	bnorm_mean_ = bmean;   /* STATS_PATCH */

	FeedT *gplane = gain_plane_
			 ? feed_chw_.data() + temporal_ * 4u * plane : nullptr;
	if (!gain_plane_) {
		/* net has no gain channel; skip building one it would discard */
	} else if (sigma_plane_) {
		/*
		 * sigma = sqrt(a*S + b) with S = mean of the 4 packed channels of the
		 * CURRENT frame - must match how training built plane 16, or the
		 * conditioning input is out of distribution.
		 */
		float av = 0.0f, bv = 0.0f;
		abForGain(gain, av, bv);
		for (unsigned i = 0; i < plane; ++i) {
			const float S = 0.25f * (float(cur[i]) + float(cur[plane + i])
						 + float(cur[2u * plane + i]) + float(cur[3u * plane + i]));
			const float v = av * (S > 0.0f ? S : 0.0f) + bv;
			gplane[i] = FeedT(std::sqrt(v > 1e-12f ? v : 1e-12f));
		}
	} else if (!gplane_valid_ || gplane_gch_ != gch) {
		/*
		 * ONE CONSTANT, 1.06 MB OF STORES. gch depends only on the
		 * analogue gain, so this plane is byte-identical to the previous
		 * frame's whenever the gain has not moved -- which is every
		 * frame of a static exposure. feed_chw_ persists across frames
		 * and nothing else writes this plane, so refilling it costs
		 * 1.06 MB of stores (plus the read-for-ownership of the same
		 * lines) to change nothing. Refill only on a gain transition.
		 */
		std::fill(gplane, gplane + plane, FeedT(gch));
		gplane_gch_ = gch;
		gplane_valid_ = true;
	}

	if (mem_sigma_) {
		/*
		 * Two noise-level planes for the MEMORY slot, matching
		 * train_u8_split.py:mem_sigma_planes() exactly -- S is the mean of
		 * the four memory channels, sigma = sqrt(a*S+b)/sqrt(k), and the
		 * pair is [log2(sigma) + 7, sigma * 16]. Built from the SLOT, not
		 * feed_chw_, so brightness normalisation does not shift it, which
		 * is how training saw it.
		 *
		 * k is the effective sample count of the memory. The accumulator
		 * measures it per frame (meanN, ~5 at trust 0.8); training used 39
		 * for a deep memory and 1 for a single frame, so it matters.
		 */
		float av = 0.0f, bv = 0.0f;
		abForGain(gain, av, bv);
		const float kEff = std::max(accum_.lastMeanN(), 1.0f);
		const float invSqrtK = 1.0f / std::sqrt(kEff);
		const FeedT *mem = slotPtr(tile, 0);
		FeedT *lg = feed_chw_.data()
			    + size_t(temporal_ * 4u + (gain_plane_ ? 1u : 0u)) * plane;
		FeedT *lin = lg + plane;
		for (unsigned i = 0; i < plane; ++i) {
			const float S = 0.25f * (float(mem[i]) + float(mem[plane + i])
						 + float(mem[2u * plane + i])
						 + float(mem[3u * plane + i]));
			float var = av * (S > 0.0f ? S : 0.0f) + bv;
			if (var < 1e-12f)
				var = 1e-12f;
			const float sig = std::sqrt(var) * invSqrtK;
			lg[i] = FeedT(std::log2(sig > 1e-9f ? sig : 1e-9f) + 7.0f);
			lin[i] = FeedT(sig * 16.0f);
		}
	}
}

bool ModelDenoise::runNet(ncnn::Mat &out)
{
	/*
	 * feed_chw_ is already contiguous planar CHW and w*h*4 is 16-byte
	 * aligned for both 480x270 and 968x548, so hand NCNN the buffer
	 * directly instead of allocating + memcpying 17 planes (36 MB per
	 * forward at 968x548) on every frame.
	 */
	/*
	 * fp16 buffer handed straight to ncnn -- no cast layer, no staging.
	 * Only legal when use_fp16_storage is on: with it off ncnn reads an
	 * elemsize-2 external buffer as fp32 and walks off the end (measured:
	 * malloc(): corrupted top size), so that path materialises fp32.
	 */
#ifdef RPI_HAVE_BSTM
	if (bstm_full_active_) {
		/* No ncnn at all: feed_chw_ straight to the NPU program. run() is
		 * templated on FeedT, so this covers the fp16 and fp32 feeds. */
		const bool ok = bstm_s2d_block_
					? (bstm_feed_merged_ ? bstm_full_.execUnfold(out)
						  : bstm_full_.runS2D(feed_chw_.data(), out))
					: bstm_full_.run(feed_chw_.data(), out);
		LOG(RPiModelDenoise, Info) << "ok = " << ok;
		return ok && out.c >= 4 && (unsigned)out.w == infer_w_ &&
		       (unsigned)out.h == infer_h_;
	}
#endif

	ncnn::Mat in;
	if (fp16_) {
		in = ncnn::Mat(int(infer_w_), int(infer_h_), int(feedCh()),
			       (void *)feed_chw_.data(), size_t(2u));
	} else {
		feed_f32_.resize(feed_chw_.size());
		for (size_t i = 0; i < feed_chw_.size(); ++i)
			feed_f32_[i] = float(feed_chw_[i]);
		in = ncnn::Mat(int(infer_w_), int(infer_h_), int(feedCh()),
			       (void *)feed_f32_.data(), size_t(4u));
	}

	ncnn::Extractor ex = net_.create_extractor();
	if (ex.input(kInputBlob, in) != 0) {
		LOG(RPiModelDenoise, Error) << "ex.input(" << kInputBlob << ") failed";
		return false;
	}
	if (ex.extract(kOutputBlob, out) != 0) {
		LOG(RPiModelDenoise, Error) << "ex.extract(" << kOutputBlob << ") failed";
		return false;
	}
	return out.c >= 4 && (unsigned)out.w == infer_w_ && (unsigned)out.h == infer_h_;
}

void ModelDenoise::unpackTile(unsigned tile, uint16_t *bayer16, unsigned stridePx,
			      const ncnn::Mat &out, float black, float white)
{
	const unsigned bstep = 2u * pack_step_;
	const float scale = std::max(white - black, 1.0f);
	const unsigned crop_x0 = tiles_[tile].crop_x0;
	const unsigned crop_y0 = tiles_[tile].crop_y0;
	/*
	 * Demo: left half of the packed crop (out_w = infer_w_/2) so the right
	 * Bayer half stays raw for A/B. With pack_step>1, same-phase densify
	 * fills offsets of 2 within the pack block so the left half is densely
	 * written (visible A/B). Blend keeps original detail: mix*ncnn +
	 * (1-mix)*original — avoids full paint-mush from densify alone.
	 */
	const unsigned out_w = demo_ ? std::max(1u, infer_w_ / 2u) : infer_w_;
	const float *R = out.channel(0);
	const float *G1 = out.channel(1);
	const float *G2 = out.channel(2);
	const float *B = out.channel(3);
	const float inv_bn = 1.0f / std::max(bnorm_scale_, 1e-6f);
	const float mix = blend_;
	const float keep = 1.0f - mix;
	const unsigned densify_n = (pack_step_ > 1) ? pack_step_ : 1u;  /* DENSIFY_FIX */

#ifdef _OPENMP
	#pragma omp parallel for num_threads(std::max(threads_, 1)) schedule(static)
#endif
	for (int y_ = 0; y_ < int(infer_h_); ++y_) {
		const unsigned y = unsigned(y_);
		for (unsigned x = 0; x < out_w; ++x) {
			const size_t i = size_t(y) * infer_w_ + size_t(x);
			const float nr = R[i] * inv_bn * scale + black;
			const float ng1 = G1[i] * inv_bn * scale + black;
			const float ng2 = G2[i] * inv_bn * scale + black;
			const float nb = B[i] * inv_bn * scale + black;

			for (unsigned dy = 0; dy < densify_n; ++dy) {
				for (unsigned dx = 0; dx < densify_n; ++dx) {
					const unsigned sy = (crop_y0 + y) * bstep + dy * 2u;
					const unsigned sx = (crop_x0 + x) * bstep + dx * 2u;
					if (sy + 1 >= height_ || sx + 1 >= width_)
						continue;
					uint16_t *row0 = bayer16 + sy * stridePx + sx;
					uint16_t *row1 = bayer16 + (sy + 1) * stridePx + sx;
					row0[0] = static_cast<uint16_t>(
						clampf(mix * nr + keep * float(row0[0]), 0.0f, 65535.0f));
					row0[1] = static_cast<uint16_t>(
						clampf(mix * ng1 + keep * float(row0[1]), 0.0f, 65535.0f));
					row1[0] = static_cast<uint16_t>(
						clampf(mix * ng2 + keep * float(row1[0]), 0.0f, 65535.0f));
					row1[1] = static_cast<uint16_t>(
						clampf(mix * nb + keep * float(row1[1]), 0.0f, 65535.0f));
				}
			}
		}
	}
}

void ModelDenoise::prepare([[maybe_unused]] Metadata *imageMetadata)
{
	std::pair<SharedFD, Span<uint8_t>> bayer;

	imageMetadata->get("global.bayer_buffer", bayer);
	LOG(RPiModelDenoise, Debug) << "Model Denoise buffer: "
				    << (uint64_t)bayer.second.begin()
				    << " size: " << bayer.second.size();

	if (!init_ || !bayer.second.data() || !pack_step_ || !width_ || !height_ ||
	    tiles_.empty()) {
		LOG(RPiModelDenoise, Info)
			<< "ModelDenoise::prepare not running:"
			<< (!init_ ? " init_=false" : "")
			<< (!bayer.second.data() ? " no_bayer_buffer" : "")
			<< (!pack_step_ ? " pack_step_=0" : "")
			<< (!width_ ? " width_=0" : "")
			<< (!height_ ? " height_=0" : "")
			<< (tiles_.empty() ? " tiles_empty" : "");
		return;
	}

	LuxStatus lux{};
	imageMetadata->get("lux.status", lux);
	if (lux.lux > lux_threshold_) {
		LOG(RPiModelDenoise, Info) << "Disable AI Denoise at lux level " << lux.lux;
		return;
	}

	float black = black_default_;
	BlackLevelStatus blStatus{};
	if (imageMetadata->get("black_level.status", blStatus) == 0) {
		black = 0.25f * float(blStatus.blackLevelR + blStatus.blackLevelG +
				      blStatus.blackLevelG + blStatus.blackLevelB);
		if (black <= 0.0f)
			black = black_default_;
	}

	float gain = gain_default_;
	float dgain = 1.0f;
	{
		/*
		 * Digital gain is applied downstream by the ISP, not to the
		 * Bayer we see -- but knowing it is the difference between "the
		 * raw is fine and the display is amplifying it" and "our sigma
		 * is wrong by that factor". Log it so the distinction is
		 * observable instead of argued about.
		 */
		AgcStatus ag{};
		if (imageMetadata->get("agc.delayed_status", ag) == 0 ||
		    imageMetadata->get("agc.status", ag) == 0)
			if (ag.digitalGain > 0.0)
				dgain = float(ag.digitalGain);
	}
	DeviceStatus deviceStatus{};
	if (imageMetadata->get("device.status", deviceStatus) == 0 &&
	    deviceStatus.analogueGain > 0.0)
		gain = static_cast<float>(deviceStatus.analogueGain);
	else {
		AgcStatus agc{};
		if (imageMetadata->get("agc.delayed_status", agc) == 0 && agc.analogueGain > 0.0)
			gain = static_cast<float>(agc.analogueGain);
		else if (imageMetadata->get("agc.status", agc) == 0 && agc.analogueGain > 0.0)
			gain = static_cast<float>(agc.analogueGain);
	}
	gain *= 3.24;

	dmabufSyncStart(bayer.first);

	uint16_t *bayer16 = reinterpret_cast<uint16_t *>(bayer.second.begin());
	const unsigned stridePx = (bayer.second.size() / height_) / sizeof(uint16_t);

	/*
	 * Accept row PADDING but reject a compressed buffer.
	 *
	 * The service gets SRGGB16 padded to 1952 px per row (4294400 bytes for
	 * 1936x1100), so demanding size == width*height*2 rejected a perfectly good
	 * buffer. rpicam-* instead gets PISP_COMP1 at 2182400 bytes = 992 px/row,
	 * which makes stridePx nonsense and corrupts the frame. So the test is
	 * bytes-per-row >= width, not an exact size.
	 */
	const size_t rowBytes = bayer.second.size() / height_;
	const size_t minRow = size_t(width_) * sizeof(uint16_t);
	if (bayer.second.size() % height_ || rowBytes < minRow) {
		static bool warned = false;
		if (!warned) {
			warned = true;
			LOG(RPiModelDenoise, Error)
				<< "raw buffer " << bayer.second.size() << " bytes = " << rowBytes
				<< " B/row, need >= " << minRow << " for " << width_
				<< " px wide. Compressed raw? Skipping model_denoise.";
		}
		dmabufSyncEnd(bayer.first);
		return;
	}
	/* RAWSTATS_PATCH */
	if (probes_ && (n_frames_ % 30ull) == 0ull) {
		uint16_t rmin = 0xffff, rmax = 0;
		double racc = 0.0; unsigned rn = 0;
		for (unsigned y = height_ / 4; y < 3 * height_ / 4; y += 8) {
			const uint16_t *row = bayer16 + size_t(y) * stridePx;
			for (unsigned x = width_ / 4; x < 3 * width_ / 4; x += 8) {
				const uint16_t v = row[x];
				if (v < rmin) rmin = v;
				if (v > rmax) rmax = v;
				racc += v; ++rn;
			}
		}
		LOG(RPiModelDenoise, Info)
			<< "RAW16 min=" << rmin << " mean=" << (rn ? racc / rn : 0.0)
			<< " max=" << rmax << " (black=" << black
			<< " white=" << white_default_ << " stridePx=" << stridePx
			<< " bytes=" << bayer.second.size() << ")";
	}

	ensureFpn(gain);
	auto tFpnStart = std::chrono::steady_clock::now();
	applyFpnFull(bayer16, stridePx, black, white_default_);
	ms_fpn_ = std::chrono::duration<double, std::milli>(
		std::chrono::steady_clock::now() - tFpnStart).count();

	if (!accum_post_)
		applyAccum(bayer16, stridePx, black, gain);

	/* STAGE_TIMING */
	/*
	 * GAIN GATE. Below net_min_gain the network earns almost nothing --
	 * measured +0.02 dB at ag4 and +0.05 at ag8, against the accumulator's
	 * +10.9 -- while still costing the whole pack/feed/infer/unpack budget.
	 * Skipping it there is not a quality trade, it is the same picture
	 * sooner. The accumulator still runs; it just writes its own result
	 * back rather than handing it to the model as slot 0, because in
	 * accum_stage="input" mode memory reaches the image ONLY through the
	 * model -- bypass without write-back would emit raw frames.
	 *
	 * Gate on the SNAPPED ladder gain, not the raw one. snapGainLadder
	 * carries the hysteresis added when AGC hunting around a threshold
	 * flipped the pipeline every frame and made the FPS read 19/9/19; a
	 * bare comparison here would reintroduce exactly that.
	 */
	if (accum_input_ && net_min_gain_ > 0.0f &&
	    float(snapGainLadder(gain, accum_gain_)) < net_min_gain_) {
		auto tA = std::chrono::steady_clock::now();
		/* honour demo here as well, or the A/B split silently shows the
		 * accumulator on BOTH halves and reads as 'no denoising'. */
		applyAccumInput(bayer16, stridePx, black, gain, true,
				demo_ ? width_ / 2u : 0u);
		ms_accum_ = std::chrono::duration<double, std::milli>(
			std::chrono::steady_clock::now() - tA).count();
		dmabufSyncEnd(bayer.first);
		if ((++n_frames_ % 30ull) == 0ull)
			LOG(RPiModelDenoise, Info)
				<< "net bypassed below gain " << net_min_gain_
				<< " (again=" << gain << ") accum=" << ms_accum_ << "ms";
		if (sdn_disable_)
			imageMetadata->erase("sdn.status");
		if (tdn_disable_)
			imageMetadata->erase("tdn.status");
		if (cdn_disable_)
			imageMetadata->erase("cdn.status");
		return;
	}

	auto tp0 = std::chrono::steady_clock::now();
	/*
	 * MEMORY-INPUT MODE. The model consumes [memory, current] rather than a
	 * ring of raw frames: slot 0 is the accumulator state BEFORE this frame
	 * is folded in (matching how training built it -- the mean of the OTHER
	 * takes), slot 1 is the live frame.
	 *
	 * Two frames instead of four means buildFeed assembles 9 channels
	 * instead of 17, which is most of `feed` (measured 27.5 ms at native
	 * resolution). And memory carries ~19 frames of history in 4 channels,
	 * against 4 frames in 16.
	 */
	if (accum_input_) {
		/*
		 * UPDATE BEFORE PACKING. packMemory used to hand the model the
		 * accumulator as it stood at the END OF THE PREVIOUS FRAME, and
		 * applyAccumInput ran ~50 lines later -- so the memory slot the
		 * network reads was always one frame (~53 ms) stale. Since the
		 * full-res path operates on that slot, the PICTURE was built
		 * from stale memory, and no amount of gating could prevent it:
		 * this frame's rejection happened after the model had already
		 * consumed the old state.
		 *
		 * That is why ghosting appeared only ABOVE A CERTAIN SPEED --
		 * one frame of staleness is a small displacement for slow
		 * motion and a large one for fast motion.
		 */
		if (accum_update_first_) {
			auto tA0 = std::chrono::steady_clock::now();
			/*
			 * FUSED PACK. Hand the accumulator the two feed slots so
			 * its own row loop emits them, instead of packMemory()
			 * re-reading the state and packTileSlot() re-reading the
			 * raw -- ~12.8 MB a frame off a saturated bus. Only for
			 * the single-tile, no-writeBack, no-partial-fpn geometry
			 * the camera actually runs; anything else falls back.
			 */
			fused_pack_live_ = false;
			/*
			 * The sink emits from the blend loop, which a cold
			 * accumulator never reaches: apply() copies the frame
			 * in and returns early while !primed(). A still capture
			 * starts a FRESH IPA process, so that is exactly the
			 * case that shipped a black frame. Fall back to the
			 * separate packs until the state is real and matches
			 * this mode -- the same guard packMemory's caller uses.
			 */
			if (accum_enable_ && fused_pack_ && accum_input_ && tiles_.size() == 1 &&
			    pack_step_ == 1 && !bin2x2_ && !(fpn_active_ && !fpn_full_) &&
			    accum_.primed() && accum_.state() &&
			    accum_.stateWidth() == width_ &&
			    accum_.stateHeight() == height_) {
				/*
				 * The sink is filled by applyAccumInput()'s own blend
				 * loop below -- which now returns immediately when
				 * accum_enable_ is false (see applyAccumInput()). Without
				 * this guard, fused_pack_live_ would still skip
				 * packMemory()/packTileSlot() in the per-tile loop further
				 * down, so the feed slots would never get written at all.
				 */
				TemporalAccum::PackSink sk;
				const size_t pl = size_t(infer_h_) * infer_w_;
				sk.mem = slotPtr(0, 0);
				sk.cur = slotPtr(0, 1);
				sk.w = infer_w_; sk.h = infer_h_;
				sk.cx0 = tiles_[0].crop_x0; sk.cy0 = tiles_[0].crop_y0;
				sk.black = black;
				sk.inv = bnorm_pre_ / std::max(white_default_ - black, 1.0f);
				sk.lo = feed_lo_;
				(void)pl;
				accum_.setPackSink(sk);
				fused_pack_live_ = true;
			} else {
				accum_.clearPackSink();
			}
			if (net_pipeline_) {
				/*
				 * Steps 1-3. accum(N) is launched FIRST so it runs
				 * against net(N-1), which is still on the NPU. Both
				 * are joined before anything touches the payloads or
				 * overwrites the raw frame.
				 */
				accum_future_ = std::async(std::launch::async,
					[this, bayer16, stridePx, black, gain] {
						applyAccumInput(bayer16, stridePx, black, gain);
					});
				if (net_future_.valid() && !net_future_.get())
					pending_valid_ = false;
				accum_future_.get();
			} else {
				applyAccumInput(bayer16, stridePx, black, gain);
			}
			ms_accum_ = std::chrono::duration<double, std::milli>(
				std::chrono::steady_clock::now() - tA0).count();
		}
		AccT const *mem = accum_.state();
		if (!mem || accum_.stateWidth() != width_ ||
		    accum_.stateHeight() != height_) {
			/* No history yet: prime from this frame so slot 0 is at
			 * least the right scene, then fall through. */
			accum_.apply(bayer16, stridePx, width_, height_, black,
				     white_default_, unsigned(std::max(threads_, 1)), false);
			mem = accum_.state();
		}
		for (unsigned t = 0; t < tiles_.size(); ++t) {
			if (fused_pack_live_)
				continue;   /* accum already emitted both slots */
			if (!mem ||
			    !packMemory(t, 0, mem, accum_.stateWidth(), black, white_default_) ||
			    !packTileSlot(t, 1, bayer16, stridePx, black, white_default_)) {
				dmabufSyncEnd(bayer.first);
				return;
			}
		}
		/* buildFeed walks (ring_pos_ + t) % temporal_; pin it so slot 0
		 * is memory and slot 1 is current, every frame. */
		ring_pos_ = 0;
		ring_filled_ = temporal_;
	} else {
		for (unsigned t = 0; t < tiles_.size(); ++t) {
			if (!packTile(t, bayer16, stridePx, black, white_default_)) {
				dmabufSyncEnd(bayer.first);
				return;
			}
		}
	}
	const double ms_pack =
		std::chrono::duration<double, std::milli>(
			std::chrono::steady_clock::now() - tp0).count();

	if (!accum_input_) {
		ring_pos_ = (ring_pos_ + 1) % temporal_;
		if (ring_filled_ < temporal_)
			ring_filled_++;
	}

	if (ring_filled_ < temporal_) {
		dmabufSyncEnd(bayer.first);
		return;
	}

	/*
	 * Fold the RAW frame into memory BEFORE inference writes over it.
	 * The accumulator must average sensor frames; averaging the model's own
	 * output would feed it back its own guess and compound its errors.
	 * writeBack=false leaves the Bayer buffer alone.
	 */
	/*
	 * DO NOT overlap this with inference on a thread. It is independent of the
	 * net -- packMemory() already froze the pre-update state into slot 0 -- so
	 * it looks like free parallelism, and it was TRIED and MEASURED:
	 *
	 *     sequential   accum 11.2  net 25.2  TOTAL 39.2 ms
	 *     overlapped   accum 21-25 net 34-36 TOTAL 52-54 ms
	 *
	 * Both sides got slower. This whole stage is memory-BANDWIDTH bound, not
	 * compute bound: the NPU alone pulls ~151 MB per inference across the same
	 * DRAM the CPU passes use. Adding concurrency just splits one saturated bus
	 * two ways. The same reason OpenMP made the s2d fold slower, and why
	 * threads=1 beats threads=4. To go faster, move fewer BYTES.
	 */
	if (accum_input_ && !accum_update_first_) {
		auto tA = std::chrono::steady_clock::now();
		applyAccumInput(bayer16, stridePx, black, gain);
		ms_accum_ = std::chrono::duration<double, std::milli>(
			std::chrono::steady_clock::now() - tA).count();
	}

	double ms_total = 0.0, ms_feed = 0.0, ms_unpack = 0.0;  /* STAGE_TIMING */
	bool all_ok = true;
	for (unsigned t = 0; t < tiles_.size(); ++t) {
		auto tf0 = std::chrono::steady_clock::now();
		/*
		 * foldFeedS2D() merges buildFeed and the s2d fold into one pass,
		 * removing a whole traversal of 4.7 M elements. It is MEASURABLY
		 * SLOWER and is left disabled:
		 *
		 *     separate   feed  5.8   net 25.4   TOTAL 39.5 ms
		 *     merged     feed 19.3   net 17.8   TOTAL 44.6 ms
		 *
		 * The fold really did vanish (net -7.6 ms), but feed grew +13.5.
		 * The separate fold reads feed_chw_, nine contiguous planes walked
		 * row-wise; the merged one reads scattered slot memory AND does the
		 * temporal blend on that scattered access. On a bandwidth-bound
		 * stage, locality beats trip count. See
		 * nsa-pipeline-is-bandwidth-bound.
		 */
		bstm_feed_merged_ = false;
#ifdef RPI_HAVE_BSTM
		/*
		 * Split-input model: fold writes the image slot and the block means
		 * directly, so the NPU never receives the memory slots or the gain
		 * plane it only ever averages. buildFeed still runs -- it applies the
		 * temporal blend and the brightness normalisation that the fold reads.
		 */
		if (bstm_full_active_ && bstm_s2d_block_ && bstm_full_.inputCount() == 2) {
			auto tb0 = std::chrono::steady_clock::now();
			buildFeed(t, gain);
			auto tb1 = std::chrono::steady_clock::now();
			bstm_feed_merged_ = bstm_full_.foldSplit(feed_chw_.data(),
								 bstm_full_.inputPayload(0),
								 bstm_full_.inputPayload(1));
			auto tb2 = std::chrono::steady_clock::now();
			ms_build_ = std::chrono::duration<double, std::milli>(tb1 - tb0).count();
			ms_fold_ = std::chrono::duration<double, std::milli>(tb2 - tb1).count();
		}
#endif
		if (!bstm_feed_merged_)
			buildFeed(t, gain);
		auto tf1 = std::chrono::steady_clock::now();
		ms_feed += std::chrono::duration<double, std::milli>(tf1 - tf0).count();
		ncnn::Mat out;
		if (net_pipeline_ && tiles_.size() == 1) {
			/*
			 * Step 6: this frame's buffer receives the PREVIOUS frame's
			 * denoised pixels -- one frame of latency, by design. Safe
			 * here because pack() above has already taken the raw data
			 * into the slots, and net(N-1) was joined before that.
			 */
			auto tu0 = std::chrono::steady_clock::now();
			if (pending_valid_) {
				/* unpack with the scale these pixels carry,
				 * not this frame's. See pending_bnorm_. */
				const float cur_bn = bnorm_scale_;
				bnorm_scale_ = pending_bnorm_;
				unpackTile(t, bayer16, stridePx, pending_out_,
					   black, white_default_);
				bnorm_scale_ = cur_bn;
			} else if (runNet(pending_out_)) {
				/*
				 * First frame after a start or a switchMode:
				 * there is no previous inference to emit, and
				 * deferring would hand the application a RAW
				 * frame. A still capture is exactly one frame
				 * after a mode switch, so that path returned an
				 * undenoised (and, before joinPipeline cleared
				 * the Mat, a black) image. Pay one synchronous
				 * inference here; steady state is unaffected.
				 */
				unpackTile(t, bayer16, stridePx, pending_out_,
					   black, white_default_);
			}
			ms_unpack += std::chrono::duration<double, std::milli>(
				std::chrono::steady_clock::now() - tu0).count();
			/* Step 7: launch net(N). Only now may the output payload be
			 * overwritten, and the input payload is complete. */
			net_future_ = std::async(std::launch::async, [this] {
				auto n0 = std::chrono::steady_clock::now();
				const bool r = runNet(pending_out_);
				ms_net_async_ = std::chrono::duration<double, std::milli>(
					std::chrono::steady_clock::now() - n0).count();
				return r;
			});
			pending_valid_ = true;
			pending_bnorm_ = bnorm_scale_;
			/*
			 * Deliberately NOT added to ms_total. The async net of the
			 * PREVIOUS frame overlapped this frame's CPU work, so folding
			 * it back into TOTAL re-counts ~15 ms that never cost wall
			 * clock -- TOTAL then reads ~60 ms against a ~32 ms frame and
			 * makes a working overlap look like no change. Reported
			 * separately as netAsync= instead; the inference RATE from the
			 * n= counter is the honest throughput number either way.
			 */
			continue;
		}
		auto t0 = std::chrono::steady_clock::now();
		const bool ok = runNet(out);
		auto t1 = std::chrono::steady_clock::now();
		ms_total += std::chrono::duration<double, std::milli>(t1 - t0).count();
		if (!ok) {
			LOG(RPiModelDenoise, Error) << "NCNN infer failed tile=" << t;
			all_ok = false;
			break;
		}
		auto tu0 = std::chrono::steady_clock::now();
		unpackTile(t, bayer16, stridePx, out, black, white_default_);
		ms_unpack += std::chrono::duration<double, std::milli>(
			std::chrono::steady_clock::now() - tu0).count();

		/* STATS_PATCH */
		if (probes_ && (n_frames_ % 30ull) == 0ull) {
			const unsigned pl = infer_h_ * infer_w_;
			double im[4] = {}, om[4] = {};
			float imin = 1e9f, imax = -1e9f, omin = 1e9f, omax = -1e9f;
			for (unsigned ch = 0; ch < 4; ++ch) {
				const FeedT *ip = feed_chw_.data() + ch * pl;
				const float *op = out.channel(ch);
				for (unsigned i = 0; i < pl; ++i) {
					im[ch] += float(ip[i]); om[ch] += op[i];
					if (float(ip[i]) < imin) imin = float(ip[i]);
					if (float(ip[i]) > imax) imax = float(ip[i]);
					if (op[i] < omin) omin = op[i];
					if (op[i] > omax) omax = op[i];
				}
				im[ch] /= double(pl); om[ch] /= double(pl);
			}
			LOG(RPiModelDenoise, Info)
				<< "STATS tile=" << t << " bmean=" << bnorm_mean_
				<< " bs=" << bnorm_scale_
				<< " IN R=" << im[0] << " G1=" << im[1] << " G2=" << im[2]
				<< " B=" << im[3] << " range=" << imin << ".." << imax
				<< " | OUT R=" << om[0] << " G1=" << om[1] << " G2=" << om[2]
				<< " B=" << om[3] << " range=" << omin << ".." << omax
				<< " | gplane="
				<< (gain_plane_ ? float(*(feed_chw_.data() + temporal_ * 4u * pl))
						: -1.0f);
		}
	}

	/*
	 * Accumulate on what the model actually produced. Skipped when the
	 * inference failed: half a frame of model output blended into
	 * history would persist for as long as the accumulator takes to
	 * decay, long after the failing frame is gone.
	 */
	if (accum_post_ && all_ok && !accum_input_) {
		/* Only overwrite the timer when this path actually ran --
		 * input mode already recorded its accumulator cost earlier. */
		auto tAccStart = std::chrono::steady_clock::now();
		applyAccum(bayer16, stridePx, black, gain);
		ms_accum_ = std::chrono::duration<double, std::milli>(
			std::chrono::steady_clock::now() - tAccStart).count();
	}

	dmabufSyncEnd(bayer.first);

	if (!all_ok) {
		/*
		 * The timing log used to sit AFTER this return, so a failed
		 * inference produced neither denoising NOR any log line -- the
		 * pipeline ran silently with the net dead. That made every
		 * truncated-graph measurement return nothing and cost two wrong
		 * guesses about where h2's time goes. Log first, then bail.
		 */
		LOG(RPiModelDenoise, Info)
			<< "infer FAILED n=" << n_frames_
			<< " [pack=" << ms_pack << " feed=" << ms_feed
			<< " net=" << ms_total << " unpack=" << ms_unpack
			<< " accum=" << ms_accum_ << "]";
		return;
	}

	/*
	 * OUTPUT PROBE. `probes` measures the RAW buffer on the way IN; this
	 * measures it on the way OUT, which is the only thing that answers
	 * "did the model help or hurt".
	 *
	 * With demo=true the left half of the frame is the model's output and
	 * the right half is untouched sensor data, in the SAME frame, at the
	 * same instant. So the ratio of high-frequency energy between the two
	 * halves is a direct, self-referencing measure of how much detail the
	 * model removed -- no ground truth, no capture, no encoder.
	 *
	 * hf is the mean |second difference| along a row of ONE Bayer channel
	 * (stride 2 to stay on R), sampled every 8th row. Cheap enough to leave
	 * behind the probes flag: measured well under 0.5 ms.
	 *
	 * DO NOT measure this with rpicam-vid. The encoder's DRAM traffic on
	 * top of a bandwidth-bound pipeline took this Pi down hard, twice.
	 */
	if (probes_ && (n_frames_ % 30ull) == 0ull && cover_bw_ > 32 && cover_bh_ > 16) {
		const unsigned half = cover_bw_ / 2u;
		double hfL = 0.0, hfR = 0.0;
		unsigned nL = 0, nR = 0;
		for (unsigned y = 8; y + 8 < cover_bh_; y += 8) {
			const uint16_t *row = bayer16 + size_t(y) * stridePx;
			for (unsigned x = 2; x + 2 < half; x += 2) {
				hfL += std::fabs(2.0 * row[x] - row[x - 2] - row[x + 2]);
				++nL;
			}
			for (unsigned x = half + 2; x + 2 < cover_bw_; x += 2) {
				hfR += std::fabs(2.0 * row[x] - row[x - 2] - row[x + 2]);
				++nR;
			}
		}
		if (nL && nR) {
			const double L = hfL / nL, R = hfR / nR;
			LOG(RPiModelDenoise, Info)
				<< "OUTPROBE n=" << n_frames_
				<< " hf_model=" << L << " hf_raw=" << R
				<< " kept=" << (R > 1e-9 ? L / R : 0.0)
				<< " gainplane=" << encodeGain(gain, gain_plane_max_)
				<< " again=" << gain;
		}
	}

	/* Per-frame stage timings at Debug: the 30-frame summary hides variance,
	 * and variance is exactly what a frame-rate wobble is made of. */
	LOG(RPiModelDenoise, Debug)
		<< "FRAMETIME pack=" << ms_pack << " feed=" << ms_feed
		<< " net=" << ms_total << " unpack=" << ms_unpack
		<< " fpn=" << ms_fpn_ << " accum=" << ms_accum_
		<< " warp=" << (accum_.lastDx() || accum_.lastDy() ? 1 : 0);

	if ((++n_frames_ % 30ull) == 0ull) {
		LOG(RPiModelDenoise, Info)
			<< "infer ok n=" << n_frames_ << " " << ms_total << "ms"
			<< " [pack=" << ms_pack << " feed=" << ms_feed
			<< " net=" << ms_total << " unpack=" << ms_unpack << " fpn=" << ms_fpn_ << " accum=" << ms_accum_
			<< " TOTAL=" << (ms_pack + ms_feed + ms_total + ms_unpack) << "ms]"
			<< " netAsync=" << ms_net_async_
			<< " gated=" << last_gated_
			<< " gHalf=" << accum_.lastGHalf()
			<< " gQtr=" << accum_.lastGQuarter()
			<< " mdiff=" << accum_.lastMeanDiff()
			<< " sig=" << accum_.lastMeanSigma()
			<< " meanN=" << accum_.lastMeanN()
#ifdef RPI_HAVE_BSTM
			<< " exec=" << bstm_full_.lastExecMs() << " unfold=" << bstm_full_.lastUnfoldMs()
#endif
			<< " build=" << ms_build_ << " fold=" << ms_fold_
			<< " warp=" << (accum_.lastDx() || accum_.lastDy() ? 1 : 0)
			<< " tiles=" << tiles_.size() << " again=" << gain << " dgain=" << dgain
			<< " black=" << black << " cover=" << cover_bw_ << "x" << cover_bh_
			<< (demo_ ? " demo" : "");
	}

	/* Pedestal restored in unpack — keep existing black_level.status. */

	if (sdn_disable_)
		imageMetadata->erase("sdn.status");
	if (tdn_disable_)
		imageMetadata->erase("tdn.status");
	if (cdn_disable_)
		imageMetadata->erase("cdn.status");
}

static Algorithm *Create(Controller *controller)
{
	return (Algorithm *)new ModelDenoise(controller);
}
static RegisterAlgorithm reg(NAME, &Create);
