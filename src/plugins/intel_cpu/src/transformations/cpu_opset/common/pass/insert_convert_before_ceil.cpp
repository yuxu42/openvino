// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "insert_convert_before_ceil.hpp"

#include <memory>
#include <optional>

#include "openvino/cc/pass/itt.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/ceiling.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

namespace {

auto is_low_precision(const ov::element::Type& prc) -> bool {
    return prc == ov::element::f16 || prc == ov::element::bf16;
}

auto pick_low_precision_from_type(const ov::element::Type& prc, std::optional<ov::element::Type>& current) -> void {
    if (!is_low_precision(prc)) {
        return;
    }
    if (prc == ov::element::bf16) {
        current = ov::element::bf16;
        return;
    }
    if (!current.has_value()) {
        current = ov::element::f16;
    }
}

auto pick_low_precision_from_convert(const std::shared_ptr<const ov::Node>& node,
                                     std::optional<ov::element::Type>& current) -> void {
    if (const auto convert = ov::as_type_ptr<const ov::op::v0::Convert>(node)) {
        pick_low_precision_from_type(convert->get_input_element_type(0), current);
        pick_low_precision_from_type(convert->get_output_element_type(0), current);
    }
}

}  // namespace

ov::intel_cpu::InsertConvertBeforeCeil::InsertConvertBeforeCeil() {
    MATCHER_SCOPE(InsertConvertBeforeCeil);

    auto add_m = ov::pass::pattern::wrap_type<ov::op::v1::Add>({ov::pass::pattern::any_input(),
                                                               ov::pass::pattern::any_input()});
    auto ceil_m = ov::pass::pattern::wrap_type<ov::op::v0::Ceiling>({add_m});

    ov::matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        auto add = ov::as_type_ptr<ov::op::v1::Add>(pattern_map.at(add_m).get_node_shared_ptr());
        auto ceil = ov::as_type_ptr<ov::op::v0::Ceiling>(pattern_map.at(ceil_m).get_node_shared_ptr());
        if (!add || !ceil) {
            return false;
        }

        if (const auto convert = ov::as_type_ptr<ov::op::v0::Convert>(ceil->get_input_node_shared_ptr(0))) {
            if (is_low_precision(convert->get_output_element_type(0))) {
                return false;
            }
        }

        std::optional<ov::element::Type> low_prc;
        pick_low_precision_from_type(add->get_output_element_type(0), low_prc);
        pick_low_precision_from_type(ceil->get_output_element_type(0), low_prc);

        for (size_t i = 0; i < add->get_input_size(); ++i) {
            pick_low_precision_from_type(add->get_input_element_type(i), low_prc);
            pick_low_precision_from_convert(add->get_input_node_shared_ptr(i), low_prc);
        }
        pick_low_precision_from_convert(ceil->get_input_node_shared_ptr(0), low_prc);

        if (!low_prc.has_value()) {
            return false;
        }

        auto convert = std::make_shared<ov::op::v0::Convert>(add->output(0), low_prc.value());
        convert->set_friendly_name(add->get_friendly_name() + "/round_to_" + low_prc.value().get_type_name());
        ov::copy_runtime_info({add, ceil}, convert);
        ceil->input(0).replace_source_output(convert->output(0));
        return true;
    };

    auto matcher = std::make_shared<ov::pass::pattern::Matcher>(ceil_m, matcher_name);
    this->register_matcher(matcher, callback);
}
