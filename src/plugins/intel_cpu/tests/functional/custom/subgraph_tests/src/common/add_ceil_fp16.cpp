// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "openvino/core/type/float16.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/ceiling.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/runtime/core.hpp"
#include "utils/cpu_test_utils.hpp"

namespace ov {
namespace test {

TEST(smoke_Basic, AddCeilFp16BoundaryRegression) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();

    auto lhs = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{2});
    auto rhs = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{2});

    auto add = std::make_shared<ov::op::v1::Add>(lhs, rhs);
    auto ceil = std::make_shared<ov::op::v0::Ceiling>(add);

    auto model = std::make_shared<ov::Model>(ov::ResultVector{std::make_shared<ov::op::v0::Result>(ceil)},
                                             ov::ParameterVector{lhs, rhs});

    ov::Core core;
    auto compiled_model = core.compile_model(model, "CPU");
    auto infer_request = compiled_model.create_infer_request();

    auto lhs_tensor = ov::Tensor(ov::element::f16, ov::Shape{2});
    auto rhs_tensor = ov::Tensor(ov::element::f16, ov::Shape{2});

    auto* lhs_data = lhs_tensor.data<ov::float16>();
    auto* rhs_data = rhs_tensor.data<ov::float16>();

    lhs_data[0] = ov::float16(4.1875f);
    rhs_data[0] = ov::float16(3.81640625f);

    lhs_data[1] = ov::float16(3.67578125f);
    rhs_data[1] = ov::float16(6.328125f);

    infer_request.set_input_tensor(0, lhs_tensor);
    infer_request.set_input_tensor(1, rhs_tensor);
    infer_request.infer();

    const auto output = infer_request.get_output_tensor();
    const auto* out_data = output.data<const ov::float16>();

    ASSERT_EQ(static_cast<float>(out_data[0]), 8.0f);
    ASSERT_EQ(static_cast<float>(out_data[1]), 10.0f);
}

}  // namespace test
}  // namespace ov
