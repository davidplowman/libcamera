/* SPDX-License-Identifier: LGPL-2.1-or-later */
/*
 * Copyright (C) 2025, Raspberry Pi plc
 *
 * camera_sensor_memory.cpp - A fake camera sensor for reading raw data from memory
 */

#include "libcamera/internal/camera_sensor_memory.h"

#include <algorithm>
#include <map>
#include <sstream>

#include <libcamera/base/log.h>
#include <libcamera/base/utils.h>

#include <libcamera/control_ids.h>
#include <libcamera/controls.h>
#include <libcamera/geometry.h>
#include <libcamera/orientation.h>
#include <libcamera/property_ids.h>
#include <libcamera/transform.h>

#include <libcamera/ipa/core_ipa_interface.h>

#include "libcamera/internal/bayer_format.h"
#include "libcamera/internal/formats.h"
#include "libcamera/internal/v4l2_subdevice.h"

namespace libcamera {

LOG_DECLARE_CATEGORY(CameraSensor)

static bool v4l2SubdeviceFormatEqual(const V4L2SubdeviceFormat &lhs, const V4L2SubdeviceFormat &rhs)
{
	return lhs.code == rhs.code && lhs.size == rhs.size && lhs.colorSpace == rhs.colorSpace;
}

CameraSensorMemory::CameraSensorMemory(const StreamConfiguration &rawInput)
	: rawInput_(rawInput), properties_(propertiesInfoMap_), controls_(controlsInfoMap_)
{
	model_ = "memory";

	std::ostringstream oss;
	oss << &rawInput;
	id_ = oss.str();

	/* The "camera" must appear to return the format the raw input wants. */
	bayerFormat_ = BayerFormat::fromPixelFormat(rawInput.pixelFormat);
	unsigned int mbusCode = bayerFormat_.toMbusCode();
	mbusCodes_ = { mbusCode };

	v4l2SubdeviceFormat_ = 	V4L2SubdeviceFormat{
		.code = mbusCode,
		.size = rawInput.size,
		.colorSpace = ColorSpace::Raw,
	};

}

CameraSensorMemory::~CameraSensorMemory() = default;

std::variant<std::unique_ptr<CameraSensor>, int>
CameraSensorMemory::match([[maybe_unused]] MediaEntity *entity)
{
	return {};
}

const std::vector<unsigned int> &CameraSensorMemory::mbusCodes() const
{
	return mbusCodes_;
}

std::vector<Size> CameraSensorMemory::sizes(unsigned int mbusCode) const
{
	if (mbusCode == mbusCodes_[0])
		return { rawInput_.size };
	else
		return {};
}

Size CameraSensorMemory::resolution() const
{
	return rawInput_.size;
}

V4L2SubdeviceFormat CameraSensorMemory::getFormat(const std::vector<unsigned int> &mbusCodes,
						  [[maybe_unused]] const Size &size,
						  const Size maxSize) const
{
	if (std::find(mbusCodes.begin(), mbusCodes.end(), mbusCodes_[0]) == mbusCodes.end())
		return {};
	
	if (maxSize.width < rawInput_.size.width || maxSize.height < rawInput_.size.height)
		return {};

	return v4l2SubdeviceFormat_;
}

int CameraSensorMemory::setFormat(V4L2SubdeviceFormat *format,
				  Transform transform)
{
	if (v4l2SubdeviceFormatEqual(*format, v4l2SubdeviceFormat_) &&
	    transform == Transform::Identity)
		return 0;

	return -EPERM;
}

int CameraSensorMemory::tryFormat(V4L2SubdeviceFormat *format) const
{
	if (v4l2SubdeviceFormatEqual(*format, v4l2SubdeviceFormat_))
		return 0;

	return -EPERM;
}

int CameraSensorMemory::applyConfiguration(const SensorConfiguration &config,
					   Transform transform,
					   V4L2SubdeviceFormat *sensorFormat)
{
	if (config.bitDepth != bayerFormat_.bitDepth ||
	    config.outputSize != rawInput_.size ||
	    config.binning.binX != 1 || config.binning.binY != 1 ||
	    config.skipping.xOddInc != 1 || config.skipping.xEvenInc != 1 ||
	    config.skipping.yOddInc != 1 || config.skipping.yEvenInc != 1 ||
	    transform != Transform::Identity)
		return -EPERM;

	if (sensorFormat)
		*sensorFormat = v4l2SubdeviceFormat_;

	return 0;
}

V4L2Subdevice::Stream CameraSensorMemory::imageStream() const
{
	return V4L2Subdevice::Stream();
}

std::optional<V4L2Subdevice::Stream> CameraSensorMemory::embeddedDataStream() const
{
	return {};
}

V4L2SubdeviceFormat CameraSensorMemory::embeddedDataFormat() const
{
	return {};
}

int CameraSensorMemory::setEmbeddedDataEnabled(bool enable)
{
	return enable ? -ENOSTR : 0;
}

const ControlList &CameraSensorMemory::properties() const
{
	return properties_;
}

int CameraSensorMemory::sensorInfo([[maybe_unused]] IPACameraSensorInfo *info) const
{
	info->model = model();

	info->bitsPerPixel = bayerFormat_.bitDepth;
	info->cfaPattern = properties::draft::RGB;

	info->activeAreaSize = rawInput_.size;
	info->analogCrop = Rectangle(rawInput_.size);
	info->outputSize = rawInput_.size;

	/*
	 * These are meaningless for us, fill with ones rather than zeros because the
	 * code will divide by some of these numbers.
	 */
	info->pixelRate = 1;
	info->minLineLength = 1;
	info-> maxLineLength = 1;
	info->minFrameLength = 1;
	info->maxFrameLength = 1;

	return 0;
}

Transform CameraSensorMemory::computeTransform(Orientation *orientation) const
{
	*orientation = Orientation::Rotate0;
	return Transform::Identity;
}

BayerFormat::Order CameraSensorMemory::bayerOrder([[maybe_unused]] Transform t) const
{
	return bayerFormat_.order;
}

const ControlInfoMap &CameraSensorMemory::controls() const
{
	return *controls_.infoMap();
}


ControlList CameraSensorMemory::getControls([[maybe_unused]] const std::vector<uint32_t> &ids)
{
	return ControlList();
}

int CameraSensorMemory::setControls([[maybe_unused]] ControlList *ctrls)
{
	return -EPERM;
}

int CameraSensorMemory::setTestPatternMode([[maybe_unused]] controls::draft::TestPatternModeEnum mode)
{
	return -EPERM;
}

const CameraSensorProperties::SensorDelays &CameraSensorMemory::sensorDelays()
{
	static constexpr CameraSensorProperties::SensorDelays defaultSensorDelays = {
		.exposureDelay = 2,
		.gainDelay = 1,
		.vblankDelay = 2,
		.hblankDelay = 2,
	};
	
	return defaultSensorDelays; /* but doesn't mean anything */
}

std::string CameraSensorMemory::logPrefix() const
{
	return "'memory'";
}

/*
 * We're not going to register this camera sensor as it doesn't match media entities
 * like other sensors. Pipeline handlers will have to call it explicitly.
 */

} /* namespace libcamera */
