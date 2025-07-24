/* SPDX-License-Identifier: BSD-2-Clause */
/*
 * Copyright (C) 2025, Raspberry Pi Ltd
 *
 * default camera helper for unknown sensors
 */

#include <assert.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

#include "cam_helper.h"

using namespace RPiController;

class CamHelperDefault : public CamHelper
{
public:
	CamHelperDefault();
	uint32_t gainCode(double gain) const override;
	double gain(uint32_t gainCode) const override;
};

CamHelperDefault::CamHelperDefault()
	: CamHelper({}, 0)
{
}

uint32_t CamHelperDefault::gainCode([[maybe_unused]] double gain) const
{
	return 0;
}

double CamHelperDefault::gain([[maybe_unused]] uint32_t gainCode) const
{
	return 1.0;
}

static CamHelper *create()
{
	return new CamHelperDefault();
}

static RegisterCamHelper reg("default", &create);
