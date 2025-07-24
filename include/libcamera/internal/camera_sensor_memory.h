/* SPDX-License-Identifier: LGPL-2.1-or-later */
/*
 * Copyright (C) 2025, Raspberry Pi plc
 *
 * camera_sensor_memory.h - A fake camera sensor for reading raw data from memory
 */

#pragma once

#include <optional>
#include <string>
#include <vector>

#include <libcamera/camera.h>

#include "libcamera/internal/camera_sensor.h"

namespace libcamera {

class BayerFormat;
class Camera;
class CameraLens;
class MediaEntity;
class SensorConfiguration;

struct CameraSensorProperties;

enum class Orientation;

LOG_DECLARE_CATEGORY(CameraSensor)

class CameraSensorMemory : public CameraSensor, protected Loggable
{
public:
	CameraSensorMemory(const StreamConfiguration &rawInput);
	~CameraSensorMemory();

	static std::variant<std::unique_ptr<CameraSensor>, int>
	match(MediaEntity *entity);

	const std::string &model() const override { return model_; }
	const std::string &id() const override { return id_; }

	const MediaEntity *entity() const override { return nullptr; }
	V4L2Subdevice *device() override { return nullptr; }

	CameraLens *focusLens() override { return nullptr; }

	const std::vector<unsigned int> &mbusCodes() const override;
	std::vector<Size> sizes(unsigned int mbusCode) const override;
	Size resolution() const override;

	V4L2SubdeviceFormat getFormat(const std::vector<unsigned int> &mbusCodes,
				      const Size &size,
				      const Size maxSize) const override;
	int setFormat(V4L2SubdeviceFormat *format,
		      Transform transform = Transform::Identity) override;
	int tryFormat(V4L2SubdeviceFormat *format) const override;

	int applyConfiguration(const SensorConfiguration &config,
			       Transform transform = Transform::Identity,
			       V4L2SubdeviceFormat *sensorFormat = nullptr) override;

	V4L2Subdevice::Stream imageStream() const override;
	std::optional<V4L2Subdevice::Stream> embeddedDataStream() const override;
	V4L2SubdeviceFormat embeddedDataFormat() const override;
	int setEmbeddedDataEnabled(bool enable) override;

	const ControlList &properties() const override;
	int sensorInfo(IPACameraSensorInfo *info) const override;
	Transform computeTransform(Orientation *orientation) const override;
	BayerFormat::Order bayerOrder(Transform t) const override;

	const ControlInfoMap &controls() const override;
	ControlList getControls(const std::vector<uint32_t> &ids) override;
	int setControls(ControlList *ctrls) override;

	const std::vector<controls::draft::TestPatternModeEnum> &
	testPatternModes() const override { return testPatternModes_; }
	int setTestPatternMode(controls::draft::TestPatternModeEnum mode) override;
	const CameraSensorProperties::SensorDelays &sensorDelays() override;

protected:
	std::string logPrefix() const override;

private:
	LIBCAMERA_DISABLE_COPY(CameraSensorMemory)

	StreamConfiguration rawInput_;

	std::string model_;
	std::string id_;

	BayerFormat bayerFormat_;
	std::vector<unsigned int> mbusCodes_;

	V4L2SubdeviceFormat v4l2SubdeviceFormat_;

	ControlInfoMap propertiesInfoMap_;
	ControlInfoMap controlsInfoMap_;
	ControlList properties_;
	ControlList controls_;

	std::vector<controls::draft::TestPatternModeEnum> testPatternModes_;
};

} /* namespace libcamera */
