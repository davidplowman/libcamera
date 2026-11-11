/* SPDX-License-Identifier: BSD-2-Clause */
/*
 * Copyright (C) 2026, Raspberry Pi Ltd
 *
 * Recurrent, tiled denoise control algorithm on the BSTM NPU
 */

#include "denoise_bstm.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <errno.h>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <type_traits>
#include <utility>

#include <linux/dma-buf.h>

#include <libcamera/base/file.h>
#include <libcamera/base/log.h>
#include <libcamera/base/shared_fd.h>
#include <libcamera/base/span.h>

#include <tensorflow/lite/builtin_ops.h>
#include <tensorflow/lite/delegates/external/external_delegate.h>
#include <tensorflow/lite/interpreter_builder.h>
#include <tensorflow/lite/kernels/register.h>

#include "../denoise_status.h"
#include "../device_status.h"

#if defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>
#endif

using namespace RPiController;
using namespace libcamera;

LOG_DEFINE_CATEGORY(RPiDenoiseBstm)

#define NAME "rpi.denoise_bstm"

namespace {

/*
 * A model given by file name rather than absolute path is looked for in the
 * source tree when running uninstalled from the build directory, then where
 * it's installed, preferring /usr/local. If it's nowhere, the name is
 * returned unchanged and loading it fails later.
 */
std::string findModel(const std::string &name)
{
	if (name.empty() || name[0] == '/')
		return name;

	std::vector<std::string> paths;
	const std::string root = utils::libcameraSourcePath();
	if (!root.empty())
		paths.push_back(root + "src/ipa/rpi/pisp/data/tflite/" + name);
	for (const char *dir : { "/usr/local/share/libcamera", LIBCAMERA_DATA_DIR,
				 "/usr/share/libcamera" }) {
		const std::string path = std::string(dir) + "/ipa/rpi/pisp/tflite/" + name;
		if (std::find(paths.begin(), paths.end(), path) == paths.end())
			paths.push_back(path);
	}

	for (const std::string &path : paths) {
		if (File::exists(path))
			return path;
	}

	LOG(RPiDenoiseBstm, Error)
		<< "Model " << name << " not found in any of: " << utils::join(paths, ", ");
	return name;
}

const std::pair<const char *, BayerFormat::Order> kBayerOrders[] = {
	{ "BGGR", BayerFormat::BGGR },
	{ "GBRG", BayerFormat::GBRG },
	{ "GRBG", BayerFormat::GRBG },
	{ "RGGB", BayerFormat::RGGB },
};

/*
 * The four Bayer orders are numbered so that mirroring horizontally flips
 * bit 0 and mirroring vertically flips bit 1: BGGR mirrored horizontally is
 * GBRG, vertically GRBG, and both ways RGGB.
 */
static_assert(BayerFormat::BGGR == 0 && BayerFormat::GBRG == 1 && BayerFormat::GRBG == 2 &&
	      BayerFormat::RGGB == 3);

bool isBayer(BayerFormat::Order order)
{
	return order <= BayerFormat::RGGB;
}

std::string bayerOrderName(BayerFormat::Order order)
{
	for (const auto &[name, value] : kBayerOrders) {
		if (value == order)
			return name;
	}
	return "non-Bayer (" + std::to_string(order) + ")";
}

void dmabufSyncStart(const SharedFD &fd)
{
	struct dma_buf_sync dma_sync{};
	dma_sync.flags = DMA_BUF_SYNC_START | DMA_BUF_SYNC_RW;

	::ioctl(fd.get(), DMA_BUF_IOCTL_SYNC, &dma_sync);
}

void dmabufSyncEnd(const SharedFD &fd)
{
	struct dma_buf_sync dma_sync{};
	dma_sync.flags = DMA_BUF_SYNC_END | DMA_BUF_SYNC_RW;

	::ioctl(fd.get(), DMA_BUF_IOCTL_SYNC, &dma_sync);
}

/*
 * The network's x and prev inputs, and its r output, are the 4 Bayer planes
 * (raw 2x2-cell positions (0,0), (1,0), (0,1), (1,1)) further packed by
 * space_to_depth(2): channel c*4 + i*2 + j holds Bayer plane c's (row i,
 * column j) sub-position within each 2x2 block of that plane. The tensors
 * are NHWC, so each network pixel's 16 channels are consecutive, and a
 * network pixel covers one 4x4 raw block. This returns the channel holding
 * raw offset (dx, dy) within that block.
 */
constexpr unsigned int rawToChannel(unsigned int dx, unsigned int dy)
{
	return 4 * (2 * (dy & 1) + (dx & 1)) + 2 * (dy >> 1) + (dx >> 1);
}

constexpr unsigned int kChannels = 16;

/* 12-bit raw data, padded with zeros at the bottom: tables are indexed by raw >> 4. */
constexpr unsigned int kRawShift = 4;
constexpr unsigned int kTableSize = 65536 >> kRawShift;

using Clock = std::chrono::steady_clock;

double msSince(Clock::time_point t)
{
	return std::chrono::duration<double, std::milli>(Clock::now() - t).count();
}

/*
 * Scale the classic PISP denoise stage's status (tag) by factor, using
 * scale(status, factor), or with a factor of 0 erase it so the stage doesn't
 * run at all on a frame we've already denoised. 1 leaves it untouched.
 */
template<typename T, typename F>
void scaleStatus(Metadata *imageMetadata, const char *tag, double factor, F scale)
{
	if (factor == 0.0) {
		imageMetadata->erase(tag);
		return;
	}
	if (factor == 1.0)
		return;

	T status;
	if (imageMetadata->get(tag, status) == 0) {
		scale(status, factor);
		imageMetadata->set(tag, status);
	}
}

/*
 * Whether two camera modes deliver frames of the same size and layout from
 * the same sensor pixels, so that the tile layout and recurrent state carry
 * over. Frame duration and exposure limits don't matter.
 */
bool cameraModesMatch(const CameraMode &a, const CameraMode &b)
{
	return a.bitdepth == b.bitdepth && a.width == b.width && a.height == b.height &&
	       a.sensorWidth == b.sensorWidth && a.sensorHeight == b.sensorHeight &&
	       a.binX == b.binX && a.binY == b.binY && a.cropX == b.cropX && a.cropY == b.cropY &&
	       a.scaleX == b.scaleX && a.scaleY == b.scaleY && a.transform == b.transform &&
	       a.bayerOrder == b.bayerOrder;
}

/*
 * Lay tiles of tileSize along an axis of frameSize, both in network pixels.
 * A frame no larger than the model is a single tile at 0. Otherwise the
 * fewest tiles that overlap by at least minOverlap are spread evenly from 0
 * to frameSize - tileSize, at even positions (but the last, which ends
 * flush), so that their half-resolution grids line up. Each pair of
 * neighbours is cross-faded linearly across the middle of its overlap, over
 * no more than the distance between them, so that no position is ever
 * covered by more than two tiles. Tile i + 1 is the last to contribute from
 * the start of its cross-fade with tile i, so that is where its blending
 * starts (TileAxis::blendStart).
 */
void layoutAxis(unsigned int frameSize, unsigned int tileSize, unsigned int minOverlap,
		DenoiseBstm::TileAxis &axis)
{
	axis.tileStart.clear();

	if (frameSize <= tileSize) {
		axis.tileCount = 1;
		axis.valid = frameSize;
		axis.tileStart.push_back(0);
	} else {
		const unsigned int last = frameSize - tileSize;
		const unsigned int maxStep = tileSize - minOverlap;
		axis.valid = tileSize;
		axis.tileCount = (frameSize - minOverlap + maxStep - 1) / maxStep;

		for (;; axis.tileCount++) {
			axis.tileStart.clear();
			for (unsigned int i = 0; i < axis.tileCount; i++) {
				const double ideal = double(i) * last / (axis.tileCount - 1);
				const unsigned int even = 2 * static_cast<unsigned int>(std::lround(ideal / 2));
				axis.tileStart.push_back(i + 1 == axis.tileCount ? last : even);
			}
			bool ok = true;
			for (unsigned int i = 0; i + 1 < axis.tileCount; i++)
				ok &= axis.tileStart[i + 1] - axis.tileStart[i] <= maxStep;
			if (ok)
				break;
		}
	}

	axis.pixelTile.assign(frameSize, 0);
	axis.pixelWeight.assign(frameSize, 1.0f);
	axis.blendStart.assign(1, 0);
	unsigned int p = 0;
	for (unsigned int i = 0; i + 1 < axis.tileCount; i++) {
		const unsigned int end = axis.tileStart[i] + tileSize; /* tile i ends here */
		const unsigned int overlap = end - axis.tileStart[i + 1]; /* with tile i + 1 */
		unsigned int len = std::min(overlap, axis.tileStart[i + 1] - axis.tileStart[i]);
		const unsigned int start = std::max(axis.tileStart[i + 1] + (overlap - len) / 2, p);
		len = std::min(len, end - start);
		for (; p < start; p++)
			axis.pixelTile[p] = i;
		axis.blendStart.push_back(p);
		for (unsigned int j = 0; j < len; j++, p++) {
			axis.pixelTile[p] = i;
			axis.pixelWeight[p] = 1.0f - (j + 0.5f) / len;
		}
	}
	for (; p < frameSize; p++)
		axis.pixelTile[p] = axis.tileCount - 1;
	axis.blendStart.push_back(frameSize);
}

/*
 * The raw column where network column x's 4x4 block starts, in a frame
 * width raw pixels wide, mirrored for HFlip (see DenoiseBstm::hflip_), and
 * the raw offset from there of the block's column dx.
 */
template<bool HFlip>
inline unsigned int blockColumn(unsigned int width, unsigned int x)
{
	return HFlip ? width - 4 - 4 * x : 4 * x;
}

template<bool HFlip>
constexpr unsigned int columnOffset(unsigned int dx)
{
	return HFlip ? 3 - dx : dx;
}

/*
 * The current frame's square root for the network pixel whose block starts
 * at raw column col, in channel order, looked up from the raw values
 * (rows[dy] being the raw row of the block's row dy).
 */
template<bool HFlip>
inline void gatherSqrt(uint16_t *const rows[4], unsigned int col, const float *table, float *xf)
{
	for (unsigned int dy = 0; dy < 4; ++dy) {
		const uint16_t *r = rows[dy] + col;
		for (unsigned int dx = 0; dx < 4; ++dx)
			xf[rawToChannel(dx, dy)] = table[r[columnOffset<HFlip>(dx)] >> kRawShift];
	}
}

#if defined(__ARM_NEON)
/*
 * Square a network pixel's 16 channels (in [0, 1]) back to linear, scale to
 * 16 bits and write them over its 4x4 raw block, at raw column col.
 * FCVTAU rounds to nearest and saturates, and vqmovn_u32 saturates again to
 * uint16, so no clamp is needed. The 4 channel registers hold raw row dy's
 * samples, in dx order (reversed for HFlip), as (see rawToChannel()):
 * dy=0: zip1(c0-3, c4-7) = 0,4,1,5    dy=1: zip1(c8-11, c12-15) = 8,12,9,13
 * dy=2: zip2(c0-3, c4-7) = 2,6,3,7    dy=3: zip2(c8-11, c12-15) = 10,14,11,15
 */
template<bool HFlip>
inline void writeRaw(uint16_t *const rows[4], unsigned int col, const float32x4x4_t &out)
{
	const float32x4_t vRawMax = vdupq_n_f32(65535.0f);
	float32x4_t lin[4];
	for (unsigned int i = 0; i < 4; ++i)
		lin[i] = vmulq_f32(out.val[i], vmulq_f32(out.val[i], vRawMax));
	const float32x4_t rowv[4] = { vzip1q_f32(lin[0], lin[1]), vzip1q_f32(lin[2], lin[3]),
				      vzip2q_f32(lin[0], lin[1]), vzip2q_f32(lin[2], lin[3]) };
	for (unsigned int dy = 0; dy < 4; ++dy) {
		uint16x4_t v = vqmovn_u32(vcvtaq_u32_f32(rowv[dy]));
		if constexpr (HFlip)
			v = vrev64_u16(v);
		vst1_u16(rows[dy] + col, v);
	}
}

/*
 * An int8 residual's 16 channels minus its zero point (within [-255, 255]:
 * exact in int16), as float.
 */
inline float32x4x4_t residualMinusZp(const int8_t *rq, int16x8_t vRZp)
{
	const int8x16_t q = vld1q_s8(rq);
	const int16x8_t lo = vsubq_s16(vmovl_s8(vget_low_s8(q)), vRZp);
	const int16x8_t hi = vsubq_s16(vmovl_s8(vget_high_s8(q)), vRZp);
	auto cvt = [](int16x4_t v) { return vcvtq_f32_s32(vmovl_s16(v)); };
	return { cvt(vget_low_s16(lo)), cvt(vget_high_s16(lo)),
		 cvt(vget_low_s16(hi)), cvt(vget_high_s16(hi)) };
}

/* prev_'s 16-bit fixed point, round(prev * 65535), to float. */
inline float32x4x4_t loadPrev(const uint16_t *p, float32x4_t vInvMax)
{
	const uint16x8x2_t pu = vld1q_u16_x2(p);
	auto cvt = [&](uint16x4_t v) { return vmulq_f32(vcvtq_f32_u32(vmovl_u16(v)), vInvMax); };
	return { cvt(vget_low_u16(pu.val[0])), cvt(vget_high_u16(pu.val[0])),
		 cvt(vget_low_u16(pu.val[1])), cvt(vget_high_u16(pu.val[1])) };
}

/* out is in [0, 1], so no clamp is needed (see writeRaw()). */
inline void storePrev(uint16_t *p, const float32x4x4_t &out, float32x4_t vRawMax)
{
	uint16x4_t pu[4];
	for (unsigned int i = 0; i < 4; ++i)
		pu[i] = vqmovn_u32(vcvtaq_u32_f32(vmulq_f32(out.val[i], vRawMax)));
	vst1q_u16_x2(p, uint16x8x2_t{ vcombine_u16(pu[0], pu[1]), vcombine_u16(pu[2], pu[3]) });
}

/*
 * The next frame's int8 prev input, quantised as TFLite would: round half
 * away from zero (FCVTAS), plus the zero point, saturated to int8.
 */
inline int8x16_t quantisePrev(const float32x4x4_t &out, float32x4_t vInvScale, int32x4_t vZp)
{
	int16x4_t q16[4];
	for (unsigned int i = 0; i < 4; ++i)
		q16[i] = vqmovn_s32(vqaddq_s32(vcvtaq_s32_f32(vmulq_f32(out.val[i], vInvScale)), vZp));
	return vcombine_s8(vqmovn_s16(vcombine_s16(q16[0], q16[1])),
			   vqmovn_s16(vcombine_s16(q16[2], q16[3])));
}
#else
template<bool HFlip>
inline void writeRaw(uint16_t *const rows[4], unsigned int col, const float *out)
{
	for (unsigned int dy = 0; dy < 4; ++dy) {
		uint16_t *r = rows[dy] + col;
		for (unsigned int dx = 0; dx < 4; ++dx) {
			const float o = out[rawToChannel(dx, dy)];
			r[columnOffset<HFlip>(dx)] =
				uint16_t(std::lround(std::min(o * (o * 65535.0f), 65535.0f)));
		}
	}
}
#endif

/*
 * The tiles covering one network pixel along one axis: one, or two (the
 * second being the next tile) where neighbours are cross-faded.
 */
struct Cover {
	unsigned int n;
	unsigned int tile[2];
	unsigned int local[2]; /* position within each tile */
	float weight[2];
};

inline Cover cover(const DenoiseBstm::TileAxis &axis, unsigned int p)
{
	const unsigned int t = axis.pixelTile[p];
	const float w = axis.pixelWeight[p];
	if (w < 1.0f)
		return { 2, { t, t + 1 }, { p - axis.tileStart[t], p - axis.tileStart[t + 1] }, { w, 1.0f - w } };
	return { 1, { t, t }, { p - axis.tileStart[t], 0 }, { 1.0f, 0.0f } };
}

/*
 * Shift an axis of n network pixels (2n plane pixels) by shift plane pixels:
 * position p reads from p - shift, clamped to the frame.
 */
void buildShiftMap(DenoiseBstm::ShiftMap &map, unsigned int n, int shift)
{
	const int last = int(2 * n) - 1;
	map.plane.resize(2 * n);
	map.whole.resize(n);
	for (unsigned int p = 0; p < 2 * n; p++)
		map.plane[p] = std::clamp(int(p) - shift, 0, last);
	for (unsigned int i = 0; i < n; i++) {
		const unsigned int p = map.plane[2 * i];
		map.whole[i] = p % 2 == 0 && map.plane[2 * i + 1] == p + 1 ? int(p / 2) : -1;
	}
}

/*
 * The 16 channels of network pixel (x, y) of the shifted state, in channel
 * order. Channel c*4 + i*2 + j is Bayer plane c at plane position
 * (2y + i, 2x + j), so it reads plane c at (mapY.plane[2y + i],
 * mapX.plane[2x + j]). When that's a whole network pixel of the state, it's
 * returned directly; otherwise it's gathered into tmp.
 */
template<typename T>
inline const T *shiftedPrev(const T *state, unsigned int packW, const DenoiseBstm::ShiftMap &mapY,
			    const DenoiseBstm::ShiftMap &mapX, unsigned int y, unsigned int x, T *tmp)
{
	const int wy = mapY.whole[y], wx = mapX.whole[x];
	if (wy >= 0 && wx >= 0)
		return state + (size_t(wy) * packW + wx) * kChannels;

	for (unsigned int i = 0; i < 2; i++) {
		const unsigned int p = mapY.plane[2 * y + i];
		for (unsigned int j = 0; j < 2; j++) {
			const unsigned int q = mapX.plane[2 * x + j];
			const T *s = state + (size_t(p >> 1) * packW + (q >> 1)) * kChannels +
				     (p & 1) * 2 + (q & 1);
			for (unsigned int c = 0; c < 4; c++)
				tmp[c * 4 + i * 2 + j] = s[c * 4];
		}
	}
	return tmp;
}

/*
 * The map for a rotation by theta and a translation by (tx, ty) quads, in a
 * frame of packW x packH network pixels (2 packW x 2 packH quads): output
 * quad p, measured from the frame's centre, reads the quad nearest
 * R^T (p - t).
 */
void buildRotationMap(DenoiseBstm::RotationMap &map, unsigned int packW, unsigned int packH,
		      double tx, double ty, double theta)
{
	const int planeW = 2 * packW, planeH = 2 * packH;
	const double cx = (planeW - 1) / 2.0, cy = (planeH - 1) / 2.0;
	const double c = std::cos(theta), s = std::sin(theta);
	auto fixed = [](double v) { return int32_t(std::lround(v * 65536.0)); };

	map.rowX.resize(planeH);
	map.rowY.resize(planeH);
	const double u0 = -cx - tx; /* column 0, from the centre, less the translation */
	for (int p = 0; p < planeH; p++) {
		const double v = p - cy - ty;
		map.rowX[p] = fixed(c * u0 + s * v + cx) + 0x8000;
		map.rowY[p] = fixed(-s * u0 + c * v + cy) + 0x8000;
	}
	map.dX = fixed(c);
	map.dY = fixed(-s);
	map.maxQ = planeW - 1;
	map.maxP = planeH - 1;
}

/*
 * As shiftedPrev(), for a rotation: the four quads of network pixel (x, y)
 * each read wherever map says, the whole network pixel being returned
 * directly when that's where they all read from, in order.
 */
template<typename T>
inline const T *rotatedPrev(const T *state, unsigned int packW, const DenoiseBstm::RotationMap &map,
			    unsigned int y, unsigned int x, T *tmp)
{
	int p[2][2], q[2][2];
	for (unsigned int i = 0; i < 2; i++) {
		const int32_t rx = map.rowX[2 * y + i], ry = map.rowY[2 * y + i];
		for (unsigned int j = 0; j < 2; j++) {
			const int32_t c = int32_t(2 * x + j);
			q[i][j] = std::clamp((rx + c * map.dX) >> 16, 0, map.maxQ);
			p[i][j] = std::clamp((ry + c * map.dY) >> 16, 0, map.maxP);
		}
	}

	if (p[0][0] % 2 == 0 && q[0][0] % 2 == 0 && p[0][1] == p[0][0] && q[0][1] == q[0][0] + 1 &&
	    p[1][0] == p[0][0] + 1 && q[1][0] == q[0][0] && p[1][1] == p[0][0] + 1 &&
	    q[1][1] == q[0][0] + 1)
		return state + (size_t(p[0][0] >> 1) * packW + (q[0][0] >> 1)) * kChannels;

	for (unsigned int i = 0; i < 2; i++) {
		for (unsigned int j = 0; j < 2; j++) {
			const T *s = state + (size_t(p[i][j] >> 1) * packW + (q[i][j] >> 1)) * kChannels +
				     (p[i][j] & 1) * 2 + (q[i][j] & 1);
			for (unsigned int c = 0; c < 4; c++)
				tmp[c * 4 + i * 2 + j] = s[c * 4];
		}
	}
	return tmp;
}

} /* namespace */

DenoiseBstm::DenoiseBstm(Controller *controller)
	: DenoiseAlgorithm(controller)
{
}

DenoiseBstm::~DenoiseBstm()
{
	if (npuThread_.joinable()) {
		{
			std::lock_guard<std::mutex> lock(npuMutex_);
			npuExit_ = true;
		}
		npuCv_.notify_all();
		npuThread_.join();
	}

	/* The interpreter still references the delegate until it's gone. */
	interpreter_.reset();
	if (tfDelegate_)
		TfLiteExternalDelegateDelete(tfDelegate_);
}

char const *DenoiseBstm::name() const
{
	return NAME;
}

int DenoiseBstm::read(const libcamera::ValueNode &params)
{
	return config_.read(params);
}

int DenoiseBstm::Config::read(const libcamera::ValueNode &params)
{
	model = params["model"].get<std::string>("");
	/* "model" may instead name another key that holds the model's path. */
	if (!model.empty() && params.contains(model))
		model = params[model].get<std::string>("");
	model = findModel(model);
	delegate = params["delegate"].get<std::string>(RPI_TEFLON_PATH);
	floatFormat = params["float_format"].get<std::string>("bf16");
	const std::string bayerOrder = params["model_bayer_order"].get<std::string>("");
	sdnFactor = params["sdn_factor"].get<double>(0.0);
	tdnFactor = params["tdn_factor"].get<double>(0.0);
	cdnFactor = params["cdn_factor"].get<double>(0.0);

	minOverlap = params["min_overlap"].get<unsigned int>(4);
	cpu = params["cpu"].get<bool>(false);
	cpuThreads = params["cpu_threads"].get<int>(3);
	demo = params["demo"].get<bool>(false);
	resetRatio = params["reset_ratio"].get<double>(2.0);
	gainThresholdOn = params["gain_threshold_on"].get<double>(32.0);
	gainThresholdOff = params["gain_threshold_off"].get<double>(24.0);
	const int rangeX = params["motion_range_x"].get<int>(0);
	const int rangeY = params["motion_range_y"].get<int>(0);
	motionRotation = params["motion_rotation"].get<bool>(false);
	motionCheckMargin = params["motion_check_margin"].get<double>(0.02);

	dumpDir = params["dump_dir"].get<std::string>("");
	dumpFrames = params["dump_frames"].get<unsigned int>(0);

	if (model.empty()) {
		LOG(RPiDenoiseBstm, Error) << "\"model\" must be supplied";
		return -EINVAL;
	}

	auto order = std::find_if(std::begin(kBayerOrders), std::end(kBayerOrders),
				  [&](const auto &o) { return bayerOrder == o.first; });
	if (order == std::end(kBayerOrders)) {
		LOG(RPiDenoiseBstm, Error)
			<< "\"model_bayer_order\" must be \"RGGB\", \"GRBG\", \"GBRG\" or \"BGGR\"";
		return -EINVAL;
	}
	modelBayerOrder = order->second;

	if (sdnFactor < 0.0 || tdnFactor < 0.0 || cdnFactor < 0.0) {
		LOG(RPiDenoiseBstm, Error)
			<< "\"sdn_factor\", \"tdn_factor\" and \"cdn_factor\" must not be negative";
		return -EINVAL;
	}

	if (floatFormat != "bf16" && floatFormat != "fp16") {
		LOG(RPiDenoiseBstm, Error)
			<< "\"float_format\" must be \"bf16\" or \"fp16\", not \"" << floatFormat << "\"";
		return -EINVAL;
	}

	if (!(resetRatio > 1.0)) {
		LOG(RPiDenoiseBstm, Error) << "\"reset_ratio\" must be greater than 1";
		return -EINVAL;
	}

	if (gainThresholdOff > gainThresholdOn) {
		LOG(RPiDenoiseBstm, Error)
			<< "\"gain_threshold_off\" must not be greater than \"gain_threshold_on\"";
		return -EINVAL;
	}

	if (rangeX < 0 || rangeY < 0 || rangeX % 2 || rangeY % 2) {
		LOG(RPiDenoiseBstm, Error)
			<< "\"motion_range_x\" and \"motion_range_y\" must be even and not negative";
		return -EINVAL;
	}
	motionRangeX = rangeX;
	motionRangeY = rangeY;

	if (motionCheckMargin < 0.0 || motionCheckMargin >= 1.0) {
		LOG(RPiDenoiseBstm, Error) << "\"motion_check_margin\" must be at least 0 and less than 1";
		return -EINVAL;
	}

	return 0;
}

/*
 * A model input or output must be [1, H, W, channels], float32 or int8. An
 * int8 tensor must be quantised per tensor (a single scale and zero point),
 * which are returned in scale and zp.
 */
bool DenoiseBstm::checkTensor(const TfLiteTensor *t, const char *what, int channels,
			      bool &isInt8, float &scale, int &zp)
{
	if (!t || !t->dims || t->dims->size != 4 || t->dims->data[0] != 1 ||
	    t->dims->data[3] != channels) {
		LOG(RPiDenoiseBstm, Error)
			<< config_.model << ": " << what << " must be [1, H, W, " << channels << "]";
		return false;
	}

	if (t->type == kTfLiteFloat32) {
		isInt8 = false;
		return true;
	}

	if (t->type != kTfLiteInt8) {
		LOG(RPiDenoiseBstm, Error)
			<< config_.model << ": " << what << " must be float32 or int8, not "
			<< TfLiteTypeGetName(t->type);
		return false;
	}

	const auto *q = static_cast<const TfLiteAffineQuantization *>(t->quantization.params);
	if (t->quantization.type != kTfLiteAffineQuantization || !q || !q->scale ||
	    q->scale->size != 1 || !(t->params.scale > 0.0f)) {
		LOG(RPiDenoiseBstm, Error)
			<< config_.model << ": " << what << " must have a single (per-tensor) scale";
		return false;
	}

	isInt8 = true;
	scale = t->params.scale;
	zp = t->params.zero_point;
	return true;
}

void DenoiseBstm::initialise()
{
	flatbuffer_ = tflite::FlatBufferModel::BuildFromFile(config_.model.c_str());
	if (!flatbuffer_) {
		LOG(RPiDenoiseBstm, Error) << "Failed to load TFLite model: " << config_.model;
		return;
	}

	/*
	 * On the CPU, BuiltinOpResolverWithXNNPACK asks TFLite to apply its
	 * default XNNPACK delegate, for float and int8 alike, which is about 3x
	 * faster for this network. Debian's libtensorflow-lite is built without
	 * XNNPACK, though (its MaybeCreateXNNPACKDelegate() returns nullptr),
	 * so there this runs on TFLite's builtin kernels, and the log shows 0
	 * nodes delegated. The resolver must outlive AllocateTensors(), where
	 * a default delegate would be applied.
	 */
	std::unique_ptr<tflite::ops::builtin::BuiltinOpResolver> resolver;
	if (config_.cpu)
		resolver = std::make_unique<tflite::ops::builtin::BuiltinOpResolverWithXNNPACK>();
	else
		resolver = std::make_unique<tflite::ops::builtin::BuiltinOpResolver>();
	tflite::InterpreterBuilder(*flatbuffer_, *resolver)(&interpreter_);
	if (!interpreter_) {
		LOG(RPiDenoiseBstm, Error) << "Failed to build TFLite interpreter for " << config_.model;
		return;
	}
	/*
	 * Threads for TFLite's CPU kernels (and its default XNNPACK delegate).
	 * On the NPU the whole graph is one Teflon delegate node, which runs on
	 * the calling thread, so the number makes no difference there.
	 */
	interpreter_->SetNumThreads(config_.cpu ? config_.cpuThreads : 1);

	if (!config_.cpu) {
		/*
		 * Read by the bstm driver when the delegate opens the NPU, so it
		 * must be set before then; the inputs' type is already known.
		 */
		if (!interpreter_->inputs().empty() &&
		    interpreter_->tensor(interpreter_->inputs()[0])->type == kTfLiteFloat32)
			setenv("BSTM_FLOAT", config_.floatFormat.c_str(), 0);

		TfLiteExternalDelegateOptions options =
			TfLiteExternalDelegateOptionsDefault(config_.delegate.c_str());
		tfDelegate_ = TfLiteExternalDelegateCreate(&options);
		if (!tfDelegate_) {
			LOG(RPiDenoiseBstm, Error)
				<< "Failed to create the Teflon delegate from " << config_.delegate;
			return;
		}

		/* This is where the driver compiles the model (or finds it in its disk cache). */
		if (interpreter_->ModifyGraphWithDelegate(tfDelegate_) != kTfLiteOk) {
			LOG(RPiDenoiseBstm, Error)
				<< "Failed to apply the Teflon delegate to " << config_.model;
			return;
		}
	}

	if (interpreter_->AllocateTensors() != kTfLiteOk) {
		LOG(RPiDenoiseBstm, Error) << "Failed to allocate tensors for " << config_.model;
		return;
	}

	/*
	 * With config_.cpu, any delegate nodes are TFLite's own default CPU delegate
	 * (XNNPACK), applied by AllocateTensors(); otherwise they're the NPU.
	 */
	unsigned int delegated = 0;
	const std::vector<int> &plan = interpreter_->execution_plan();
	for (int node : plan) {
		const auto *nr = interpreter_->node_and_registration(node);
		if (nr && nr->second.builtin_code == kTfLiteBuiltinDelegate)
			delegated++;
	}
	if (!config_.cpu && !delegated)
		LOG(RPiDenoiseBstm, Warning) << "No part of " << config_.model << " runs on the NPU";
	else if (!config_.cpu && delegated != plan.size())
		LOG(RPiDenoiseBstm, Warning)
			<< config_.model << ": " << plan.size() - delegated << " of " << plan.size()
			<< " execution-plan nodes run on the CPU";

	if (interpreter_->inputs().size() != 2 || interpreter_->outputs().size() != 2) {
		LOG(RPiDenoiseBstm, Error)
			<< config_.model << " must have exactly two inputs (x, prev) and two outputs (r, a)";
		return;
	}

	/*
	 * The inputs have the same shape, so they're taken in the order the
	 * interpreter lists them, x then prev (convert_temporal.py --verify
	 * checks this). The outputs are told apart by their last dimension.
	 */
	inX_ = interpreter_->inputs()[0];
	inPrev_ = interpreter_->inputs()[1];
	for (int idx : interpreter_->outputs()) {
		const TfLiteTensor *t = interpreter_->tensor(idx);
		const int c = t && t->dims && t->dims->size == 4 ? t->dims->data[3] : 0;
		if (c == int(kChannels))
			outR_ = idx;
		else if (c == 1)
			outA_ = idx;
	}
	if (outR_ < 0 || outA_ < 0) {
		LOG(RPiDenoiseBstm, Error)
			<< config_.model << ": outputs must be r [1, H, W, 16] and a [1, H, W, 1]";
		return;
	}

	const TfLiteTensor *x = interpreter_->tensor(inX_);
	const TfLiteTensor *prev = interpreter_->tensor(inPrev_);
	const TfLiteTensor *r = interpreter_->tensor(outR_);
	const TfLiteTensor *a = interpreter_->tensor(outA_);
	bool int8[4];
	if (!checkTensor(x, "input x", kChannels, int8[0], xScale_, xZp_) ||
	    !checkTensor(prev, "input prev", kChannels, int8[1], prevScale_, prevZp_) ||
	    !checkTensor(r, "output r", kChannels, int8[2], rScale_, rZp_) ||
	    !checkTensor(a, "output a", 1, int8[3], aScale_, aZp_))
		return;
	if (int8[1] != int8[0] || int8[2] != int8[0] || int8[3] != int8[0]) {
		LOG(RPiDenoiseBstm, Error)
			<< config_.model << ": inputs and outputs must be all int8 or all float32";
		return;
	}
	isFloat_ = !int8[0];

	netH_ = x->dims->data[1];
	netW_ = x->dims->data[2];
	for (const TfLiteTensor *t : { prev, r, a }) {
		if (static_cast<unsigned int>(t->dims->data[1]) != netH_ ||
		    static_cast<unsigned int>(t->dims->data[2]) != netW_) {
			LOG(RPiDenoiseBstm, Error)
				<< config_.model << ": all inputs and outputs must be the same size";
			return;
		}
	}

	if (2 * config_.minOverlap >= std::min(netW_, netH_)) {
		LOG(RPiDenoiseBstm, Error)
			<< "\"min_overlap\" (" << config_.minOverlap << ") must be less than half the model's size ("
			<< netW_ << "x" << netH_ << ")";
		return;
	}

	buildTables();

	LOG(RPiDenoiseBstm, Info)
		<< "Loaded " << config_.model << " (" << kChannels << "x" << netH_ << "x" << netW_
		<< "), " << delegated << "/" << plan.size()
		<< (config_.cpu ? " nodes delegated to XNNPACK, all on the CPU (" : " nodes on the NPU")
		<< (config_.cpu ? std::to_string(config_.cpuThreads) + " threads)" : "");
	if (isFloat_ && !config_.cpu)
		LOG(RPiDenoiseBstm, Info)
			<< "float32 model, BSTM_FLOAT=" << getenv("BSTM_FLOAT");
	else if (isFloat_)
		LOG(RPiDenoiseBstm, Info) << "float32 model";
	else
		LOG(RPiDenoiseBstm, Info)
			<< "int8 model: x (" << xScale_ << ", " << xZp_ << "), prev (" << prevScale_
			<< ", " << prevZp_ << "), r (" << rScale_ << ", " << rZp_ << "), a (" << aScale_
			<< ", " << aZp_ << ")";

	npuThread_ = std::thread(&DenoiseBstm::npuLoop, this);
	init_ = true;
}

/*
 * The input tables don't depend on the model except through x's
 * quantisation; the sigmoid table belongs to a's scale and zero point. All
 * are computed in double, then rounded once. A float model needs only
 * xfTable_: its a output isn't limited to 256 values, so the sigmoid is
 * computed directly.
 */
void DenoiseBstm::buildTables()
{
	xfTable_.resize(kTableSize);
	for (unsigned int i = 0; i < kTableSize; i++)
		xfTable_[i] = float(std::sqrt(double(i << kRawShift) / 65535.0));

	if (isFloat_) {
		xqTable_.clear();
		alphaTable_.clear();
		return;
	}

	xqTable_.resize(kTableSize);
	for (unsigned int i = 0; i < kTableSize; i++) {
		const double v = std::sqrt(double(i << kRawShift) / 65535.0);
		xqTable_[i] = int8_t(std::clamp(int(std::lround(v / xScale_)) + xZp_, -128, 127));
	}

	alphaTable_.resize(256);
	for (int q = -128; q <= 127; q++)
		alphaTable_[q + 128] = float(1.0 / (1.0 + std::exp(-double(q - aZp_) * aScale_)));
}

void DenoiseBstm::switchMode(CameraMode const &cameraMode, [[maybe_unused]] Metadata *metadata)
{
	/*
	 * initialise() failed (the model couldn't be loaded or isn't usable,
	 * already logged): stay inactive, and let the camera run without us.
	 */
	if (!init_)
		return;

	/*
	 * Restarting the camera in the same mode keeps the tile layout, the
	 * buffers and the recurrent state (checkForReset() still restarts
	 * from frame 0 on a large jump in exposure). active_ is only set once
	 * a mode has been set up, so the first call always sets up.
	 */
	if (active_ && cameraModesMatch(cameraMode, mode_)) {
		LOG(RPiDenoiseBstm, Debug) << "Camera mode unchanged: keeping the temporal state";
		return;
	}

	mode_ = cameraMode;
	width_ = cameraMode.width;
	height_ = cameraMode.height;
	active_ = false;

	/* Mirror the frame to the model's Bayer order (see hflip_). */
	if (!isBayer(cameraMode.bayerOrder)) {
		LOG(RPiDenoiseBstm, Error)
			<< "Raw images are " << bayerOrderName(cameraMode.bayerOrder)
			<< ", not a Bayer order: can't denoise";
		return;
	}
	const unsigned int flips = cameraMode.bayerOrder ^ config_.modelBayerOrder;
	hflip_ = flips & 1;
	vflip_ = flips & 2;
	if ((hflip_ && width_ % 2) || (vflip_ && height_ % 2)) {
		LOG(RPiDenoiseBstm, Error)
			<< "Can't mirror a " << width_ << "x" << height_ << " Bayer frame";
		return;
	}

	/* One network pixel per 4x4 raw block; tiled when larger than the model. */
	packW_ = width_ / 4;
	packH_ = height_ / 4;
	if (!packW_ || !packH_)
		return;

	layoutAxis(packW_, netW_, config_.minOverlap, tilesX_);
	layoutAxis(packH_, netH_, config_.minOverlap, tilesY_);

	/* Every tile's x, prev, r and a, each 64-byte aligned. */
	auto align = [](size_t n) { return (n + 63) & ~size_t(63); };
	const size_t elem = isFloat_ ? sizeof(float) : sizeof(int8_t);
	const size_t n16 = size_t(netW_) * netH_ * kChannels * elem;
	const size_t n1 = size_t(netW_) * netH_ * elem;
	offX_ = 0;
	offPrev_ = align(n16);
	offR_ = offPrev_ + align(n16);
	offA_ = offR_ + align(n16);
	tileBytes_ = offA_ + align(n1);
	const size_t bytes = tileBytes_ * tiles();
	if (bytes != tileBuffersSize_) {
		tileBuffers_.reset(static_cast<uint8_t *>(aligned_alloc(64, bytes)));
		tileBuffersSize_ = tileBuffers_ ? bytes : 0;
		if (!tileBuffers_) {
			LOG(RPiDenoiseBstm, Error) << "Failed to allocate " << bytes << " bytes of tile buffers";
			return;
		}
	}

	/*
	 * Only the part of each tile backed by the frame is ever packed, so
	 * where the frame is smaller than the model, the rest of its inputs
	 * keeps the encoding of zero filled here for as long as this mode lasts.
	 */
	const size_t n = size_t(netW_) * netH_ * kChannels;
	for (unsigned int k = 0; k < tiles(); k++) {
		if (isFloat_) {
			std::fill(tileX<float>(k), tileX<float>(k) + n, 0.0f);
			std::fill(tilePrev<float>(k), tilePrev<float>(k) + n, 0.0f);
		} else {
			std::fill(tileX<int8_t>(k), tileX<int8_t>(k) + n, int8_t(xZp_));
			std::fill(tilePrev<int8_t>(k), tilePrev<int8_t>(k) + n, int8_t(prevZp_));
		}
	}

	/* Every frame position is finished by exactly one tile. */
	for (const TileAxis *axis : { &tilesX_, &tilesY_ }) {
		ASSERT(axis->blendStart.size() == axis->tileCount + 1);
		ASSERT(axis->blendStart.front() == 0);
		ASSERT(std::is_sorted(axis->blendStart.begin(), axis->blendStart.end()));
	}
	ASSERT(tilesX_.blendStart.back() == packW_ && tilesY_.blendStart.back() == packH_);

	const size_t state = size_t(packW_) * packH_ * kChannels;
	prev_.assign(isFloat_ ? 0 : state, 0);
	prevQ_.assign(isFloat_ ? 0 : state, int8_t(prevZp_));
	prevFloat_.assign(isFloat_ ? state : 0, 0.0f);

	/*
	 * Motion compensation, in plane pixels. Each axis's search must leave at
	 * least half the frame to compare, or it's left uncompensated.
	 */
	motionX_ = config_.motionRangeX / 2;
	motionY_ = config_.motionRangeY / 2;
	if (4 * motionX_ > 2 * packW_ || 4 * motionY_ > 2 * packH_) {
		LOG(RPiDenoiseBstm, Warning)
			<< "Motion ranges " << config_.motionRangeX << "x" << config_.motionRangeY
			<< " too large for a " << 4 * packW_ << "x" << 4 * packH_
			<< " frame: not compensating motion along that axis";
		if (4 * motionX_ > 2 * packW_)
			motionX_ = 0;
		if (4 * motionY_ > 2 * packH_)
			motionY_ = 0;
	}
	const bool motion = motionX_ || motionY_;
	prevLast_.assign(motion && !isFloat_ ? state : 0, 0);
	prevFloatLast_.assign(motion && isFloat_ ? state : 0, 0.0f);
	shiftX_ = shiftY_ = 0;
	shifted_ = false;

	/*
	 * Each half's search ranges are the motion ranges, in working pixels
	 * (4 raw pixels, or 2 plane pixels, each), rounded up; 0 for an axis
	 * that isn't compensated. The checks above leave each half at least
	 * twice as wide as its horizontal search, and the frame at least twice
	 * as high as its vertical one.
	 */
	rotation_ = config_.motionRotation && motionX_ && motionY_;
	if (config_.motionRotation && motion && !rotation_)
		LOG(RPiDenoiseBstm, Warning)
			<< "Motion rotation needs both motion ranges: estimating translation only";
	globalMotion_.maxShiftX = (motionX_ + 1) / 2;
	globalMotion_.maxShiftY = (motionY_ + 1) / 2;
	globalMotion_.rotation = rotation_;
	globalMotion_.checkMargin = config_.motionCheckMargin;
	workCur_.assign(motion ? size_t(packW_) * packH_ : 0, 0.0f);
	workPrev_.assign(motion ? size_t(packW_) * packH_ : 0, 0.0f);
	rotationRadius_ = std::hypot(packW_ - 0.5, packH_ - 0.5);
	rotated_ = false;

	dumpCount_ = 0;
	active_ = true;
	lastExposure_ = 0.0;
	reset();

	auto overlaps = [](const TileAxis &axis, unsigned int net) {
		if (axis.tileCount < 2)
			return std::string("-");
		unsigned int lo = net, hi = 0;
		for (unsigned int i = 0; i + 1 < axis.tileCount; i++) {
			const unsigned int o = axis.tileStart[i] + net - axis.tileStart[i + 1];
			lo = std::min(lo, o);
			hi = std::max(hi, o);
		}
		return lo == hi ? std::to_string(lo) : std::to_string(lo) + "-" + std::to_string(hi);
	};
	const char *mirror = hflip_ ? (vflip_ ? "mirrored both ways" : "mirrored horizontally")
				    : (vflip_ ? "mirrored vertically" : "not mirrored");
	LOG(RPiDenoiseBstm, Info)
		<< "Bayer " << bayerOrderName(cameraMode.bayerOrder) << " (model "
		<< bayerOrderName(config_.modelBayerOrder) << ", " << mirror << ") "
		<< width_ << "x" << height_ << " (" << packW_ << "x" << packH_
		<< " network pixels) -> " << tilesX_.tileCount << "x" << tilesY_.tileCount
		<< " tiles of " << netW_ << "x" << netH_ << ", overlaps " << overlaps(tilesX_, netW_)
		<< " x " << overlaps(tilesY_, netH_) << " (minimum " << config_.minOverlap << "), "
		<< bytes / 1024 << " KiB of tile buffers";
	if (motion)
		LOG(RPiDenoiseBstm, Info)
			<< "Motion compensation up to " << 2 * motionX_ << "x" << 2 * motionY_
			<< " raw pixels" << (rotation_ ? ", with rotation" : "");
}

void DenoiseBstm::setMode([[maybe_unused]] DenoiseMode mode)
{
}

/*
 * Restart from frame 0: the state becomes zeros, so the tiles' prev inputs
 * are packed as real zeros (for int8, its zero point, -128, not int8 0), and
 * the next blend ignores alpha. There's nothing to estimate motion against.
 */
void DenoiseBstm::reset()
{
	std::fill(prev_.begin(), prev_.end(), 0);
	std::fill(prevQ_.begin(), prevQ_.end(), int8_t(prevZp_));
	std::fill(prevFloat_.begin(), prevFloat_.end(), 0.0f);

	firstFrame_ = true;
	workPrevValid_ = false;
}

/*
 * Switch the network on or off by the analogue gain, with hysteresis. It
 * restarts from frame 0 when it comes back on, as its state is stale.
 */
void DenoiseBstm::checkGain(double analogueGain)
{
	if (!aiOn_ && analogueGain >= config_.gainThresholdOn) {
		LOG(RPiDenoiseBstm, Debug)
			<< "Analogue gain " << analogueGain << " -- switching AI denoise on";
		aiOn_ = true;
		lastExposure_ = 0.0;
		reset();
	} else if (aiOn_ && analogueGain < config_.gainThresholdOff) {
		LOG(RPiDenoiseBstm, Debug)
			<< "Analogue gain " << analogueGain << " -- switching AI denoise off";
		aiOn_ = false;
	}
}

/*
 * Restart on a large jump in exposure time x analogue gain, which the
 * network was never trained across.
 */
void DenoiseBstm::checkForReset(const DeviceStatus &deviceStatus)
{
	const double exposure = deviceStatus.exposureTime.get<std::micro>() * deviceStatus.analogueGain;
	if (exposure <= 0.0)
		return;

	const double last = std::exchange(lastExposure_, exposure);
	if (last <= 0.0)
		return;

	const double ratio = exposure / last;
	if (ratio > config_.resetRatio || ratio * config_.resetRatio < 1.0) {
		LOG(RPiDenoiseBstm, Debug)
			<< "Exposure x gain jumped from " << last << " to " << exposure
			<< " -- restarting";
		reset();
	}
}

/*
 * The inference thread: run each tile as soon as it's packed, in order,
 * pointing the interpreter's input and output tensors at its buffers first.
 * After a failure, stop until the next frame.
 */
void DenoiseBstm::npuLoop()
{
	std::unique_lock<std::mutex> lock(npuMutex_);
	for (;;) {
		npuCv_.wait(lock, [this] { return npuExit_ || (!npuError_ && done_ < packed_); });
		if (npuExit_)
			return;
		const unsigned int k = done_;
		lock.unlock();

		const Clock::time_point t0 = Clock::now();
		const int idx[4] = { inX_, inPrev_, outR_, outA_ };
		uint8_t *const buf[4] = { tileX<uint8_t>(k), tilePrev<uint8_t>(k), tileR<uint8_t>(k),
					  tileA<uint8_t>(k) };
		for (unsigned int i = 0; i < 4; i++)
			interpreter_->tensor(idx[i])->data.raw = reinterpret_cast<char *>(buf[i]);
		const bool ok = interpreter_->Invoke() == kTfLiteOk;
		const Clock::time_point t1 = Clock::now();

		lock.lock();
		npuMs_ += std::chrono::duration<double, std::milli>(t1 - t0).count();
		if (k == 0)
			npuFirstStart_ = t0;
		npuLastEnd_ = t1;
		if (ok)
			done_++;
		else
			npuError_ = true;
		npuCv_.notify_all();
	}
}

void DenoiseBstm::startFrame([[maybe_unused]] unsigned int tiles)
{
	std::lock_guard<std::mutex> lock(npuMutex_);
	packed_ = 0;
	done_ = 0;
	npuError_ = false;
	npuMs_ = 0.0;
}

void DenoiseBstm::publishPacked(unsigned int packed)
{
	{
		std::lock_guard<std::mutex> lock(npuMutex_);
		packed_ = packed;
	}
	npuCv_.notify_all();
}

/* Wait until the first tiles tiles are done; false if inference failed. */
bool DenoiseBstm::waitDone(unsigned int tiles)
{
	std::unique_lock<std::mutex> lock(npuMutex_);
	npuCv_.wait(lock, [&] { return npuError_ || done_ >= tiles; });
	return !npuError_;
}

/*
 * Pack tile k's x input: its part of the raw Bayer frame, network pixel
 * (x,y) of a tile at (ox,oy) covering rows [4(oy+y),4(oy+y)+4) and columns
 * [4(ox+x),4(ox+x)+4) of the mirrored view (see hflip_), offset (dx,dy)
 * going to channel rawToChannel(dx,dy), each sample being its square root
 * from table: quantised (xqTable_) for an int8 model, or in float
 * (xfTable_).
 */
template<typename T, bool HFlip>
void DenoiseBstm::packTileX(unsigned int k, const uint16_t *bayer16, unsigned int stridePx,
			    const T *table)
{
	const unsigned int netW = netW_, width = width_;
	const unsigned int validW = tilesX_.valid, validH = tilesY_.valid;
	const unsigned int ox = tilesX_.tileStart[k % tilesX_.tileCount];
	const unsigned int oy = tilesY_.tileStart[k / tilesX_.tileCount];
	T *in = tileX<T>(k);

	for (unsigned int y = 0; y < validH; ++y) {
		T *d = in + size_t(y) * netW * kChannels;
		const uint16_t *rows[4];
		for (unsigned int dy = 0; dy < 4; ++dy)
			rows[dy] = bayer16 + size_t(rawRow(4 * (oy + y) + dy)) * stridePx;

		for (unsigned int x = 0; x < validW; ++x) {
			T *p = d + size_t(x) * kChannels;
			const unsigned int col = blockColumn<HFlip>(width, ox + x);
			for (unsigned int dy = 0; dy < 4; ++dy) {
				const uint16_t *r = rows[dy] + col;
				for (unsigned int dx = 0; dx < 4; ++dx)
					p[rawToChannel(dx, dy)] =
						table[r[columnOffset<HFlip>(dx)] >> kRawShift];
			}
		}
	}
}

/*
 * Pack tile k's prev input: its part of the recurrent state, copied from
 * prevQ_ (already quantised) for an int8 model, or from prevFloat_ (or
 * prevFloatLast_, where estimateMotion() has swapped it), and shifted or
 * rotated by this frame's motion. After a reset that is the encoding of zero.
 */
template<typename T>
void DenoiseBstm::packTilePrev(unsigned int k)
{
	const unsigned int netW = netW_, packW = packW_;
	const unsigned int validW = tilesX_.valid, validH = tilesY_.valid;
	const unsigned int ox = tilesX_.tileStart[k % tilesX_.tileCount];
	const unsigned int oy = tilesY_.tileStart[k / tilesX_.tileCount];
	T *prevIn = tilePrev<T>(k);
	const T *state;
	if constexpr (std::is_same_v<T, int8_t>)
		state = prevQ_.data();
	else
		state = shifted_ ? prevFloatLast_.data() : prevFloat_.data();

	for (unsigned int y = 0; y < validH; ++y) {
		T *d = prevIn + size_t(y) * netW * kChannels;

		/* Unshifted, the tile's row of prev is one contiguous run of the state. */
		if (!shifted_) {
			const T *s = state + (size_t(oy + y) * packW + ox) * kChannels;
			std::copy(s, s + size_t(validW) * kChannels, d);
			continue;
		}

		for (unsigned int x = 0; x < validW; ++x) {
			alignas(16) T tmp[kChannels];
			const T *s = rotated_ ? rotatedPrev(state, packW, rotationMap_, oy + y, ox + x, tmp)
					      : shiftedPrev(state, packW, shiftMapY_, shiftMapX_, oy + y,
							    ox + x, tmp);
			std::copy(s, s + kChannels, d + size_t(x) * kChannels);
		}
	}
}

/*
 * Build the current frame's working image (see workCur_)
 * straight from the raw frame, before any tile is packed: each network
 * pixel's 16 raw samples summed. Network pixel x of the mirrored view is
 * raw block b = x counted from the left, or b = packW_ - 1 - x counted from
 * the left of the last whole block for HFlip (see blockColumn()), and the
 * sum doesn't depend on the order of the rows or columns within a block.
 */
template<bool HFlip>
void DenoiseBstm::workFrame(const uint16_t *bayer16, unsigned int stridePx)
{
	const unsigned int packW = packW_;
	const unsigned int c0 = HFlip ? width_ - 4 * packW : 0;

	for (unsigned int y = 0; y < packH_; y++) {
		const uint16_t *rows[4];
		for (unsigned int dy = 0; dy < 4; ++dy)
			rows[dy] = bayer16 + size_t(rawRow(4 * y + dy)) * stridePx + c0;
		float *out = workCur_.data() + size_t(y) * packW;

		unsigned int b = 0;
#if defined(__ARM_NEON)
		/* Four blocks at a time: pairwise sums widened to 32 bits, then pairs again. */
		for (; b + 4 <= packW; b += 4) {
			uint32x4_t lo = vdupq_n_u32(0), hi = vdupq_n_u32(0);
			for (unsigned int dy = 0; dy < 4; ++dy) {
				lo = vpadalq_u16(lo, vld1q_u16(rows[dy] + 4 * b));
				hi = vpadalq_u16(hi, vld1q_u16(rows[dy] + 4 * b + 8));
			}
			float32x4_t v = vcvtq_f32_u32(vpaddq_u32(lo, hi));
			if constexpr (HFlip) {
				v = vrev64q_f32(v);
				vst1q_f32(out + packW - 4 - b, vextq_f32(v, v, 2));
			} else {
				vst1q_f32(out + b, v);
			}
		}
#endif
		for (; b < packW; b++) {
			uint32_t sum = 0;
			for (unsigned int dy = 0; dy < 4; ++dy)
				for (unsigned int dx = 0; dx < 4; ++dx)
					sum += rows[dy][4 * b + dx];
			out[HFlip ? packW - 1 - b : b] = float(sum);
		}
	}
}

/*
 * Once the current frame's working image is complete, estimate this frame's
 * motion against the previous result's (a translation, and with rotation_
 * possibly a rotation), and set up the shift or rotation. Shifted or
 * rotated, the previous state is swapped into prevLast_ (prevFloatLast_)
 * for the blend to read, as it can't be updated in place.
 *
 * The motion is logged in the sensor's own orientation (raw pixels,
 * degrees), not mirrored: mirroring an axis negates that axis's translation,
 * and mirroring either one (but not both) negates the rotation.
 */
void DenoiseBstm::estimateMotion()
{
	shiftX_ = shiftY_ = 0;
	rotated_ = false;
	shifted_ = false;
	if (firstFrame_ || !workPrevValid_)
		return;

	const GlobalMotionResult m = estimateGlobalMotion(workCur_.data(), workPrev_.data(), packW_,
							  packH_, packW_, globalMotion_);

	static const char *kinds[] = { "none", "translation", "rotation" };
	const double sx = hflip_ ? -4.0 : 4.0, sy = vflip_ ? -4.0 : 4.0;
	std::ostringstream log;
	log << "Motion: " << kinds[int(m.kind)] << " x " << sx * m.tx << " y " << sy * m.ty;
	if (rotation_) {
		const double sTheta = hflip_ != vflip_ ? -180.0 / M_PI : 180.0 / M_PI;
		log << " degrees " << sTheta * m.theta;
	}
	log << ", costs none " << m.costNone << " translation " << m.costTranslation;
	if (rotation_) {
		log << " rotation " << m.costRotation;
		if (*m.reason)
			log << " (no rotation candidate: " << m.reason << ")";
		log << ", scale " << m.scale;
	}
	for (unsigned int h = 0; h < 2; h++) {
		/* The halves, as they lie in the raw frame. */
		const GlobalMotionHalf &half = m.half[hflip_ ? 1 - h : h];
		log << (h ? ", right" : ", left") << " x " << sx * half.dx << " y " << sy * half.dy
		    << " (peaks " << half.peakX << ", " << half.peakY;
		if (rotation_ && !half.centresOk)
			log << ", geometric centres";
		log << ")";
	}
	LOG(RPiDenoiseBstm, Debug) << log.str();

	if (m.kind != GlobalMotionResult::Kind::None)
		buildMotionMaps(m);

	if (shifted_) {
		if (isFloat_)
			std::swap(prevFloat_, prevFloatLast_);
		else
			std::swap(prev_, prevLast_);
	}
}

/*
 * Set up the warp for an estimated motion: in plane pixels (quads, two per
 * working pixel), measured from the frame's centre, output quad p reads the
 * previous result at R^T (p - t). When the rotation moves no quad by half a
 * quad or more (always, for a translation), it's a translation (rounded to
 * whole quads) instead, with shiftMapX_/shiftMapY_; otherwise rotated_ is set,
 * with rotationMap_. shifted_ is set if either moves anything.
 */
void DenoiseBstm::buildMotionMaps(const GlobalMotionResult &motion)
{
	const double tx = 2.0 * motion.tx, ty = 2.0 * motion.ty;
	if (std::abs(motion.theta) * rotationRadius_ < 0.5) {
		shiftX_ = int(std::lround(tx));
		shiftY_ = int(std::lround(ty));
		shifted_ = shiftX_ || shiftY_;
		if (shifted_) {
			buildShiftMap(shiftMapX_, packW_, shiftX_);
			buildShiftMap(shiftMapY_, packH_, shiftY_);
		}
	} else {
		buildRotationMap(rotationMap_, packW_, packH_, tx, ty, motion.theta);
		rotated_ = shifted_ = true;
	}
}

/*
 * Complete the region tile k finishes (see TileAxis) from the tiles'
 * outputs, all in float and in square-root space, for each network pixel and
 * channel:
 *
 *   s_t  = clamp(xf + r_t, 0, 1)                       for each covering tile t
 *   out  = sum_t w_t s_t                               on the first frame
 *   out  = prev + sum_t w_t alpha_t (s_t - prev)       otherwise
 *
 * with xf the current frame's square root (looked up again from the raw
 * values, which are still in the buffer), alpha_t = sigmoid(a_t), one per
 * network pixel, and w_t the tiles' cross-fade weights, which sum to 1. Every
 * covering tile is done by the time tile k is, as tiles run in raster order.
 * A pixel covered by a single tile (weight 1, and all of them when the model
 * is as large as the frame) takes exactly the arithmetic of a single
 * prev + alpha * (s - prev). out becomes the state for the next frame, and is
 * written back over the raw buffer squared, as round(out^2 * 65535). Nothing
 * is written to the tiles' buffers.
 *
 * On a shifted frame, prev is read from the previous state (swapped into
 * prevLast_ or prevFloatLast_) where this frame's motion moved it from,
 * (or rotated it from), exactly as each tile's prev input was packed. With
 * motion compensation, out also makes the working image (workPrev_) that
 * the next frame's motion is estimated against.
 *
 * int8 model: r and a are dequantised, (q - zero point) * scale, alpha comes
 * from alphaTable_, and the state is prev_ (16-bit fixed point), with
 * prevQ_ quantised from the unrounded result for the next frame's input.
 */
template<bool HFlip>
void DenoiseBstm::blendRegionInt8(unsigned int k, uint16_t *bayer16, unsigned int stridePx)
{
	const unsigned int netW = netW_, packW = packW_, width = width_, nx = tilesX_.tileCount;
	const unsigned int tx = k % nx, ty = k / nx;
	const unsigned int x0 = tilesX_.blendStart[tx], x1 = tilesX_.blendStart[tx + 1];
	const unsigned int y0 = tilesY_.blendStart[ty], y1 = tilesY_.blendStart[ty + 1];
	/*
	 * config_.demo: write back the raw buffer's right half only (but update
	 * everything else), which is the left half of the view when mirrored.
	 */
	unsigned int xBegin = 0, xEnd = packW;
	if (config_.demo && HFlip)
		xEnd = packW - packW / 2;
	else if (config_.demo)
		xBegin = packW / 2;
	const bool first = firstFrame_;
	const float *xfTable = xfTable_.data();
	const float *alphaTable = alphaTable_.data();
	uint16_t *prevBuf = prev_.data();
	const uint16_t *prevLastBuf = prevLast_.data();
	int8_t *prevQBuf = prevQ_.data();
	const bool shifted = shifted_, rotated = rotated_;
	float *workBuf = motionX_ || motionY_ ? workPrev_.data() : nullptr;
	const float rScale = rScale_;
	const int rZp = rZp_;
	const float prevInvScale = 1.0f / prevScale_;
	const int prevZp = prevZp_;

#if defined(__ARM_NEON)
	const float32x4_t vRScale = vdupq_n_f32(rScale);
	const int16x8_t vRZp = vdupq_n_s16(int16_t(rZp));
	const float32x4_t vZero = vdupq_n_f32(0.0f);
	const float32x4_t vOne = vdupq_n_f32(1.0f);
	const float32x4_t vPrevInvScale = vdupq_n_f32(prevInvScale);
	const int32x4_t vPrevZp = vdupq_n_s32(prevZp);
	const float32x4_t vRawMax = vdupq_n_f32(65535.0f);
	const float32x4_t vInvMax = vdupq_n_f32(1.0f / 65535.0f);
#endif

	for (unsigned int y = y0; y < y1; ++y) {
		const Cover cy = cover(tilesY_, y);
		uint16_t *pRow = prevBuf + size_t(y) * packW * kChannels;
		int8_t *qRow = prevQBuf + size_t(y) * packW * kChannels;
		uint16_t *rows[4];
		for (unsigned int dy = 0; dy < 4; ++dy)
			rows[dy] = bayer16 + size_t(rawRow(4 * y + dy)) * stridePx;

		for (unsigned int x = x0; x < x1; ++x) {
			const unsigned int col = blockColumn<HFlip>(width, x);
			alignas(16) float xf[kChannels];
			gatherSqrt<HFlip>(rows, col, xfTable, xf);
			uint16_t *p = pRow + size_t(x) * kChannels;
			int8_t *pq = qRow + size_t(x) * kChannels;
			alignas(16) uint16_t tmp[kChannels];
			const uint16_t *ps =
				!shifted  ? p
				: rotated ? rotatedPrev(prevLastBuf, packW, rotationMap_, y, x, tmp)
					  : shiftedPrev(prevLastBuf, packW, shiftMapY_, shiftMapX_, y, x, tmp);
			const Cover cx = cover(tilesX_, x);

			/* The covering tiles (1, 2 or 4), their weights and pixel offsets. */
			unsigned int nt = 0, tile[4];
			size_t local[4];
			float weight[4];
			for (unsigned int i = 0; i < cy.n; i++) {
				for (unsigned int j = 0; j < cx.n; j++, nt++) {
					tile[nt] = cy.tile[i] * nx + cx.tile[j];
					local[nt] = size_t(cy.local[i]) * netW + cx.local[j];
					weight[nt] = cy.weight[i] * cx.weight[j];
				}
			}

#if defined(__ARM_NEON)
			const float32x4x4_t xv = vld1q_f32_x4(xf);
			float32x4x4_t pv;
			if (!first)
				pv = loadPrev(ps, vInvMax);

			float32x4x4_t out;
			if (nt == 1) {
				const float32x4x4_t rf =
					residualMinusZp(tileR<int8_t>(tile[0]) + local[0] * kChannels, vRZp);
				const float32x4_t vAlpha =
					vdupq_n_f32(alphaTable[tileA<int8_t>(tile[0])[local[0]] + 128]);
				for (unsigned int i = 0; i < 4; ++i) {
					/* The residual's dequantising multiply is fused with the skip add. */
					float32x4_t s = vfmaq_f32(xv.val[i], rf.val[i], vRScale);
					s = vminq_f32(vmaxq_f32(s, vZero), vOne);
					out.val[i] = first ? s
							   : vfmaq_f32(pv.val[i], vAlpha, vsubq_f32(s, pv.val[i]));
				}
			} else {
				float32x4x4_t acc = { vZero, vZero, vZero, vZero };
				for (unsigned int t = 0; t < nt; t++) {
					const float32x4x4_t rf =
						residualMinusZp(tileR<int8_t>(tile[t]) + local[t] * kChannels, vRZp);
					const float alpha = alphaTable[tileA<int8_t>(tile[t])[local[t]] + 128];
					const float32x4_t vW = vdupq_n_f32(first ? weight[t] : weight[t] * alpha);
					for (unsigned int i = 0; i < 4; ++i) {
						float32x4_t s = vfmaq_f32(xv.val[i], rf.val[i], vRScale);
						s = vminq_f32(vmaxq_f32(s, vZero), vOne);
						acc.val[i] = vfmaq_f32(acc.val[i], vW,
								       first ? s : vsubq_f32(s, pv.val[i]));
					}
				}
				for (unsigned int i = 0; i < 4; ++i)
					out.val[i] = first ? acc.val[i] : vaddq_f32(pv.val[i], acc.val[i]);
			}
			storePrev(p, out, vRawMax);
			vst1q_s8(pq, quantisePrev(out, vPrevInvScale, vPrevZp));
			if (workBuf) {
				/* The next frame's working image: linear, as 16-bit raw values. */
				float32x4_t sq = vmulq_f32(out.val[0], out.val[0]);
				for (unsigned int i = 1; i < 4; ++i)
					sq = vfmaq_f32(sq, out.val[i], out.val[i]);
				workBuf[size_t(y) * packW + x] = 65535.0f * vaddvq_f32(sq);
			}
#else
			float out[kChannels];
			for (unsigned int c = 0; c < kChannels; ++c) {
				const float pc = float(ps[c]) * (1.0f / 65535.0f);
				float acc = 0.0f;
				for (unsigned int t = 0; t < nt; t++) {
					const int8_t *rq = tileR<int8_t>(tile[t]) + local[t] * kChannels;
					const float alpha = alphaTable[tileA<int8_t>(tile[t])[local[t]] + 128];
					const float s = std::clamp(std::fma(float(int(rq[c]) - rZp), rScale, xf[c]),
								   0.0f, 1.0f);
					if (nt == 1)
						acc = first ? s : std::fma(alpha, s - pc, pc);
					else
						acc = std::fma(first ? weight[t] : weight[t] * alpha,
							       first ? s : s - pc, acc);
				}
				out[c] = nt == 1 || first ? acc : pc + acc;
				p[c] = uint16_t(std::lround(out[c] * 65535.0f));
				pq[c] = int8_t(std::clamp(int(std::lround(out[c] * prevInvScale)) + prevZp,
							  -128, 127));
			}
			if (workBuf) {
				float sq = 0.0f;
				for (unsigned int c = 0; c < kChannels; ++c)
					sq += out[c] * out[c];
				workBuf[size_t(y) * packW + x] = 65535.0f * sq;
			}
#endif

			if (x >= xBegin && x < xEnd)
				writeRaw<HFlip>(rows, col, out);
		}
	}
}

/*
 * As blendRegionInt8(), for a float model: r and a need no dequantising,
 * alpha is computed directly (one exp per network pixel and tile), and the
 * state is prevFloat_.
 */
template<bool HFlip>
void DenoiseBstm::blendRegionFloat(unsigned int k, uint16_t *bayer16, unsigned int stridePx)
{
	const unsigned int netW = netW_, packW = packW_, width = width_, nx = tilesX_.tileCount;
	const unsigned int tx = k % nx, ty = k / nx;
	const unsigned int x0 = tilesX_.blendStart[tx], x1 = tilesX_.blendStart[tx + 1];
	const unsigned int y0 = tilesY_.blendStart[ty], y1 = tilesY_.blendStart[ty + 1];
	/*
	 * config_.demo: write back the raw buffer's right half only (but update
	 * everything else), which is the left half of the view when mirrored.
	 */
	unsigned int xBegin = 0, xEnd = packW;
	if (config_.demo && HFlip)
		xEnd = packW - packW / 2;
	else if (config_.demo)
		xBegin = packW / 2;
	const bool first = firstFrame_;
	const float *xfTable = xfTable_.data();
	float *prevBuf = prevFloat_.data();
	const float *prevLastBuf = prevFloatLast_.data();
	const bool shifted = shifted_, rotated = rotated_;
	float *workBuf = motionX_ || motionY_ ? workPrev_.data() : nullptr;

#if defined(__ARM_NEON)
	const float32x4_t vZero = vdupq_n_f32(0.0f);
	const float32x4_t vOne = vdupq_n_f32(1.0f);
#endif

	for (unsigned int y = y0; y < y1; ++y) {
		const Cover cy = cover(tilesY_, y);
		float *pRow = prevBuf + size_t(y) * packW * kChannels;
		uint16_t *rows[4];
		for (unsigned int dy = 0; dy < 4; ++dy)
			rows[dy] = bayer16 + size_t(rawRow(4 * y + dy)) * stridePx;

		for (unsigned int x = x0; x < x1; ++x) {
			const unsigned int col = blockColumn<HFlip>(width, x);
			alignas(16) float xf[kChannels];
			gatherSqrt<HFlip>(rows, col, xfTable, xf);
			float *p = pRow + size_t(x) * kChannels;
			alignas(16) float tmp[kChannels];
			const float *ps =
				!shifted  ? p
				: rotated ? rotatedPrev(prevLastBuf, packW, rotationMap_, y, x, tmp)
					  : shiftedPrev(prevLastBuf, packW, shiftMapY_, shiftMapX_, y, x, tmp);
			const Cover cx = cover(tilesX_, x);

			unsigned int nt = 0, tile[4];
			size_t local[4];
			float weight[4];
			for (unsigned int i = 0; i < cy.n; i++) {
				for (unsigned int j = 0; j < cx.n; j++, nt++) {
					tile[nt] = cy.tile[i] * nx + cx.tile[j];
					local[nt] = size_t(cy.local[i]) * netW + cx.local[j];
					weight[nt] = cy.weight[i] * cx.weight[j];
				}
			}

#if defined(__ARM_NEON)
			const float32x4x4_t xv = vld1q_f32_x4(xf);
			float32x4x4_t pv;
			if (!first)
				pv = vld1q_f32_x4(ps);

			float32x4x4_t out;
			if (nt == 1) {
				const float32x4x4_t rv = vld1q_f32_x4(tileR<float>(tile[0]) + local[0] * kChannels);
				const float a = tileA<float>(tile[0])[local[0]];
				const float32x4_t vAlpha = vdupq_n_f32(1.0f / (1.0f + std::exp(-a)));
				for (unsigned int i = 0; i < 4; ++i) {
					float32x4_t s = vaddq_f32(xv.val[i], rv.val[i]);
					s = vminq_f32(vmaxq_f32(s, vZero), vOne);
					out.val[i] = first ? s
							   : vfmaq_f32(pv.val[i], vAlpha, vsubq_f32(s, pv.val[i]));
				}
			} else {
				float32x4x4_t acc = { vZero, vZero, vZero, vZero };
				for (unsigned int t = 0; t < nt; t++) {
					const float32x4x4_t rv =
						vld1q_f32_x4(tileR<float>(tile[t]) + local[t] * kChannels);
					const float a = tileA<float>(tile[t])[local[t]];
					const float alpha = 1.0f / (1.0f + std::exp(-a));
					const float32x4_t vW = vdupq_n_f32(first ? weight[t] : weight[t] * alpha);
					for (unsigned int i = 0; i < 4; ++i) {
						float32x4_t s = vaddq_f32(xv.val[i], rv.val[i]);
						s = vminq_f32(vmaxq_f32(s, vZero), vOne);
						acc.val[i] = vfmaq_f32(acc.val[i], vW,
								       first ? s : vsubq_f32(s, pv.val[i]));
					}
				}
				for (unsigned int i = 0; i < 4; ++i)
					out.val[i] = first ? acc.val[i] : vaddq_f32(pv.val[i], acc.val[i]);
			}
			vst1q_f32_x4(p, out);
			if (workBuf) {
				/* The next frame's working image: linear, as 16-bit raw values. */
				float32x4_t sq = vmulq_f32(out.val[0], out.val[0]);
				for (unsigned int i = 1; i < 4; ++i)
					sq = vfmaq_f32(sq, out.val[i], out.val[i]);
				workBuf[size_t(y) * packW + x] = 65535.0f * vaddvq_f32(sq);
			}
#else
			float out[kChannels];
			for (unsigned int c = 0; c < kChannels; ++c) {
				const float pc = ps[c];
				float acc = 0.0f;
				for (unsigned int t = 0; t < nt; t++) {
					const float a = tileA<float>(tile[t])[local[t]];
					const float alpha = 1.0f / (1.0f + std::exp(-a));
					const float s = std::clamp(xf[c] + tileR<float>(tile[t])[local[t] * kChannels + c],
								   0.0f, 1.0f);
					if (nt == 1)
						acc = first ? s : std::fma(alpha, s - pc, pc);
					else
						acc = std::fma(first ? weight[t] : weight[t] * alpha,
							       first ? s : s - pc, acc);
				}
				out[c] = nt == 1 || first ? acc : pc + acc;
			}
			std::copy(out, out + kChannels, p);
			if (workBuf) {
				float sq = 0.0f;
				for (unsigned int c = 0; c < kChannels; ++c)
					sq += out[c] * out[c];
				workBuf[size_t(y) * packW + x] = 65535.0f * sq;
			}
#endif

			if (x >= xBegin && x < xEnd)
				writeRaw<HFlip>(rows, col, out);
		}
	}
}

/* Debug only: see config_.dumpDir. */
void DenoiseBstm::dumpFrame(const uint16_t *bayer16, unsigned int stridePx, const char *what)
{
	std::ostringstream name;
	name << config_.dumpDir << "/" << what << "_" << std::setw(4) << std::setfill('0') << dumpCount_
	     << ".raw";
	std::ofstream f(name.str(), std::ios::binary);
	for (unsigned int y = 0; y < height_; ++y)
		f.write(reinterpret_cast<const char *>(bayer16 + size_t(y) * stridePx),
			width_ * sizeof(uint16_t));
	if (!f)
		LOG(RPiDenoiseBstm, Error) << "Failed to write " << name.str();
}

/*
 * The tiles are pipelined: this thread packs them all in raster order,
 * handing each to the inference thread as it's ready (packing is much
 * quicker than inference, so it soon runs ahead), then blends the region
 * each tile finishes as soon as that tile is done, so that when the last
 * tile is done only its own region is left. Raw pixels are only overwritten
 * once every tile has been packed (tiles also read the raw pixels they
 * share with their neighbours), and each region reads and writes only its
 * own pixels. With a single tile this is pack, invoke, blend, as before.
 */
void DenoiseBstm::prepare(Metadata *imageMetadata)
{
	std::pair<SharedFD, std::span<uint8_t>> bayer;
	imageMetadata->get("global.bayer_buffer", bayer);

	if (!active_ || !bayer.second.data() || !width_ || !height_)
		return;

	DeviceStatus deviceStatus;
	if (imageMetadata->get("device.status", deviceStatus) == 0) {
		checkGain(deviceStatus.analogueGain);
		if (aiOn_)
			checkForReset(deviceStatus);
	}
	if (!aiOn_)
		return;

	/*
	 * Accept row padding but reject a compressed buffer: bytes-per-row
	 * >= width, not an exact size.
	 */
	const size_t rowBytes = bayer.second.size() / height_;
	const size_t minRow = size_t(width_) * sizeof(uint16_t);
	if (bayer.second.size() % height_ || rowBytes < minRow) {
		static bool warned = false;
		if (!warned) {
			warned = true;
			LOG(RPiDenoiseBstm, Error)
				<< "raw buffer " << bayer.second.size() << " bytes = " << rowBytes
				<< " B/row, need >= " << minRow << " for " << width_
				<< " px wide. Compressed raw? Skipping denoise_bstm.";
		}
		return;
	}

	const Clock::time_point tStart = Clock::now();
	dmabufSyncStart(bayer.first);
	const double syncStartMs = msSince(tStart);

	uint16_t *bayer16 = reinterpret_cast<uint16_t *>(bayer.second.data());
	const unsigned int stridePx = rowBytes / sizeof(uint16_t);
	const bool dump = !config_.dumpDir.empty() && dumpCount_ < config_.dumpFrames;

	if (dump) {
		dumpFrame(bayer16, stridePx, "in");
		std::ofstream(config_.dumpDir + "/frames.txt", dumpCount_ ? std::ios::app : std::ios::trunc)
			<< dumpCount_ << " " << (firstFrame_ ? 1 : 0) << "\n";
	}

	const unsigned int nTiles = tiles();
	startFrame(nTiles);

	auto packX = [&](unsigned int k) {
		if (isFloat_ && hflip_)
			packTileX<float, true>(k, bayer16, stridePx, xfTable_.data());
		else if (isFloat_)
			packTileX<float, false>(k, bayer16, stridePx, xfTable_.data());
		else if (hflip_)
			packTileX<int8_t, true>(k, bayer16, stridePx, xqTable_.data());
		else
			packTileX<int8_t, false>(k, bayer16, stridePx, xqTable_.data());
	};
	auto packPrev = [&](unsigned int k) {
		if (isFloat_)
			packTilePrev<float>(k);
		else
			packTilePrev<int8_t>(k);
	};

	/*
	 * With motion compensation, every tile's prev input depends on the
	 * motion, so it's estimated first, from a working image read straight
	 * from the raw frame; then tiles are packed and handed to the inference
	 * thread one by one, as before.
	 */
	double motionMs = 0.0;
	if (motionX_ || motionY_) {
		const Clock::time_point tm = Clock::now();
		if (hflip_)
			workFrame<true>(bayer16, stridePx);
		else
			workFrame<false>(bayer16, stridePx);
		estimateMotion();
		motionMs = msSince(tm);
	}

	const Clock::time_point t0 = Clock::now();
	for (unsigned int k = 0; k < nTiles; k++) {
		packX(k);
		packPrev(k);
		publishPacked(k + 1);
	}
	const double packMs = msSince(t0);

	double waitMs = 0.0, blendMs = 0.0;
	Clock::time_point tLast = Clock::now();
	bool ok = true;
	for (unsigned int k = 0; k < nTiles; k++) {
		const Clock::time_point tw = Clock::now();
		ok = waitDone(k + 1);
		waitMs += msSince(tw);
		if (!ok)
			break;

		tLast = Clock::now();
		if (isFloat_ && hflip_)
			blendRegionFloat<true>(k, bayer16, stridePx);
		else if (isFloat_)
			blendRegionFloat<false>(k, bayer16, stridePx);
		else if (hflip_)
			blendRegionInt8<true>(k, bayer16, stridePx);
		else
			blendRegionInt8<false>(k, bayer16, stridePx);
		blendMs += msSince(tLast);
	}
	/* What's left once the last tile is done: blending its own region. */
	const double tailMs = msSince(tLast);

	double npuMs;
	Clock::time_point npuFirstStart, npuLastEnd;
	{
		std::lock_guard<std::mutex> lock(npuMutex_);
		npuMs = npuMs_;
		npuFirstStart = npuFirstStart_;
		npuLastEnd = npuLastEnd_;
	}

	if (!ok) {
		LOG(RPiDenoiseBstm, Error) << "NPU inference failed";
		dmabufSyncEnd(bayer.first);
		reset();
		return;
	}
	firstFrame_ = false;
	workPrevValid_ = motionX_ || motionY_;

	if (dump) {
		dumpFrame(bayer16, stridePx, "out");
		dumpCount_++;
	}

	const Clock::time_point tEnd = Clock::now();
	dmabufSyncEnd(bayer.first);
	const double syncEndMs = msSince(tEnd);

	/*
	 * The frame's critical path: until the NPU starts the first tile, the
	 * NPU running all the tiles (with any gaps where it waits for packing),
	 * and everything after the last tile (blending its region and syncing
	 * the buffer).
	 */
	const Clock::time_point tDone = Clock::now();
	auto ms = [](Clock::time_point a, Clock::time_point b) {
		return std::chrono::duration<double, std::milli>(b - a).count();
	};
	LOG(RPiDenoiseBstm, Debug)
		<< "before NPU " << ms(tStart, npuFirstStart) << " ms, NPU span "
		<< ms(npuFirstStart, npuLastEnd) << " ms, after NPU " << ms(npuLastEnd, tDone)
		<< " ms; sync " << syncStartMs << " + " << syncEndMs << " ms, pack " << packMs
		<< " ms, motion " << motionMs << " ms, network " << npuMs << " ms (" << nTiles << " tiles), waiting " << waitMs
		<< " ms, blend " << blendMs << " ms (last tile " << tailMs << " ms), total "
		<< ms(tStart, tDone) << " ms";

	/*
	 * The classic stages' noise thresholds scale with the noise model's
	 * constant and slope. CDN's threshold is deviation * slope + constant,
	 * so scaling the noise model scales the threshold itself.
	 */
	scaleStatus<SdnStatus>(imageMetadata, "sdn.status", config_.sdnFactor,
			       [](SdnStatus &st, double f) {
				       st.noiseConstant *= f;
				       st.noiseSlope *= f;
				       st.noiseConstant2 *= f;
				       st.noiseSlope2 *= f;
			       });
	scaleStatus<TdnStatus>(imageMetadata, "tdn.status", config_.tdnFactor,
			       [](TdnStatus &st, double f) {
				       st.noiseConstant *= f;
				       st.noiseSlope *= f;
			       });
	scaleStatus<CdnStatus>(imageMetadata, "cdn.status", config_.cdnFactor,
			       [](CdnStatus &st, double f) { st.threshold *= f; });
}

/* Register algorithm with the system. */
static Algorithm *create(Controller *controller)
{
	return (Algorithm *)new DenoiseBstm(controller);
}
static RegisterAlgorithm reg(NAME, &create);
