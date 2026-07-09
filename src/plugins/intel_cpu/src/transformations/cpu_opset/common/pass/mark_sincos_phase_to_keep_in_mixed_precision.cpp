// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mark_sincos_phase_to_keep_in_mixed_precision.hpp"

#include <deque>
#include <unordered_set>

#include "openvino/cc/pass/itt.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/cos.hpp"
#include "openvino/op/cum_sum.hpp"
#include "openvino/op/group_conv.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/sin.hpp"
#include "openvino/op/util/shape_of_base.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/rt_info/disable_precision_conversion.hpp"

namespace ov::intel_cpu {

namespace v0 = ov::op::v0;
namespace v1 = ov::op::v1;
namespace op_util = ov::op::util;

MarkSinCosPhaseToKeepInMixedPrecision::MarkSinCosPhaseToKeepInMixedPrecision() {
    MATCHER_SCOPE(MarkSinCosPhaseToKeepInMixedPrecision);
    auto sincos = ov::pass::pattern::wrap_type<v0::Sin, v0::Cos>();

    ov::matcher_pass_callback callback = [](ov::pass::pattern::Matcher& m) {
        const auto& node = m.get_match_root();
        if (!node)
            return false;

        auto is_boundary = [](const ov::Node* n) -> bool {
            // Stop the upstream walk (do not mark / do not cross) at graph inputs,
            // shape sub-graphs and GEMM/Convolution ops. A genuine phase Sin/Cos
            // reaches its CumSum accumulator through only cheap magnitude-preserving
            // ops (Multiply/Add/Transpose/Reshape/Interpolate/...). If a MatMul or
            // Convolution is encountered first, this Sin/Cos is a downstream consumer
            // (e.g. a resblock activation), not the phase generator, so it must not
            // be marked and the accumulator behind the GEMM must not be reached.
            return ov::is_type<v0::Constant>(n) || ov::is_type<v0::Parameter>(n) ||
                   ov::is_type<op_util::ShapeOfBase>(n) || ov::is_type<v0::MatMul>(n) ||
                   ov::is_type<v1::Convolution>(n) || ov::is_type<v1::GroupConvolution>(n) ||
                   ov::is_type<v1::ConvolutionBackpropData>(n) ||
                   ov::is_type<v1::GroupConvolutionBackpropData>(n);
        };

        // Walk upstream from the Sin/Cos argument and collect the phase segment: the
        // chain of cheap element-wise ops between the Sin/Cos and the nearest CumSum
        // phase accumulator that feeds it, including the CumSum itself. Only this
        // CumSum->Sin sub-path is precision sensitive: the phase grows unbounded
        // through the CumSum (NSF/source-filter vocoders such as kokoro reach ~3e5
        // rad), and at that magnitude the BF16/FP16 ULP is far larger than a sine
        // period, so rounding the accumulated phase turns sin()/cos() into noise.
        std::unordered_set<ov::Node*> segment;
        std::deque<ov::Node*> stack;
        bool found_cumsum = false;

        stack.push_back(node->input_value(0).get_node());
        while (!stack.empty()) {
            auto* cur = stack.back();
            stack.pop_back();
            if (!cur || segment.count(cur) || is_boundary(cur))
                continue;
            segment.insert(cur);
            if (ov::is_type<v0::CumSum>(cur)) {
                // The accumulator holds the large phase value and must stay in high
                // precision. Stop the walk here: nodes upstream of the CumSum only
                // carry small per-step increments and stay in BF16/FP16, which keeps
                // the surrounding MatMul/Convolution acoustic model in low precision.
                found_cumsum = true;
                continue;
            }
            for (const auto& in : cur->input_values()) {
                // Follow only floating-point data edges: this reaches the accumulator
                // while pruning integer shape/index sub-graphs (ShapeOf, Range, Gather
                // indices, ScatterNDUpdate, Reshape shape inputs, ...), which are never
                // converted to BF16/FP16 anyway.
                if (in.get_element_type().is_real())
                    stack.push_back(in.get_node());
            }
        }

        if (!found_cumsum)
            return false;

        // Keep the Sin/Cos itself and the CumSum->Sin segment in FP32.
        ov::disable_conversion(node, ov::element::f16);
        for (auto* n : segment) {
            ov::disable_conversion(n->shared_from_this(), ov::element::f16);
        }
        return false;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(sincos, matcher_name);
    this->register_matcher(m, callback);
}

}  // namespace ov::intel_cpu
