// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/matcher_pass.hpp"

namespace ov::intel_cpu {

/**
 * @brief Keeps the phase argument of a Sin/Cos in FP32 when it originates from a
 * CumSum (an unbounded phase accumulator), to avoid catastrophic accuracy loss
 * under BF16/FP16 inference precision.
 *
 * NSF / source-filter vocoders (e.g. kokoro) synthesize a harmonic excitation as
 * sin(phase), where `phase` is produced by accumulating positive per-sample phase
 * increments with a CumSum. Over tens of thousands of samples `phase` grows to
 * hundreds of thousands of radians. In BF16 (8 mantissa bits) the ULP at that
 * magnitude (~2048 rad) is hundreds of sine periods, so the rounded argument makes
 * sin(phase) effectively random. Marking the CumSum->Sin sub-path (and the Sin/Cos
 * itself) with the DisablePrecisionConversion rt_info keeps it in FP32 while leaving
 * the surrounding Conv/MatMul acoustic model in BF16.
 *
 * CPU-internal pass: it is registered at the end of ConvertToCPUSpecificOpset,
 * after ConvertToPowerStatic, because the DisablePrecisionConversion marker is
 * non-copyable and would otherwise be dropped when the eltwise scaler on the phase
 * path is replaced by a PowerStatic node.
 */
class MarkSinCosPhaseToKeepInMixedPrecision : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("MarkSinCosPhaseToKeepInMixedPrecision");
    MarkSinCosPhaseToKeepInMixedPrecision();
};

}  // namespace ov::intel_cpu
