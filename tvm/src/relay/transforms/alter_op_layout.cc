/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file alter_op_layout.cc
 * \brief Alternate the layouts of operators or replace primitive operators with
          other expressions. This pass can be used for computing convolution in
          custom layouts or other general weight pre-transformation.
 */
#include <tvm/relay/analysis.h>
#include <tvm/relay/attrs/transform.h>
#include <tvm/relay/op_attr_types.h>
#include <tvm/relay/transform.h>
#include <tvm/te/operation.h>
#include <tvm/relay/expr_functor.h>

#include <functional>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "pattern_utils.h"
#include "transform_layout.h"

namespace tvm {
namespace relay {

namespace alter_op_layout {

/*!
 * \brief Container to instantiate a Node for alter op layouts.
 */
class AlterTransformMemorizerNode : public TransformMemorizerNode {
 public:
  static constexpr const char* _type_key = "relay.alter_op_layout.AlterTransformMemorizerNode";
};

/*!
 * \brief Container that provides the transformation function for alter layout..
 */
class AlterTransformMemorizer : public TransformMemorizer {
 public:
  AlterTransformMemorizer() {}
  explicit AlterTransformMemorizer(ObjectPtr<Object> n) : TransformMemorizer(n) {}

  AlterTransformMemorizerNode* operator->() {
    return static_cast<AlterTransformMemorizerNode*>(get_mutable());
  }

  std::string GetBackendFromAnnotation(std::string backend_annotation) {
    // e.g., backend_annotation: 0-tensorrt_conv2d
    std::string delimiter = "-";
    int delim_pos = backend_annotation.find(delimiter);

    // This gives us "tensorrt_conv2d"
    std::string backend = backend_annotation.substr(delim_pos+1);
    delimiter = "_";
    delim_pos = backend.find(delimiter);

    return backend.substr(0, delim_pos);
  }
  /*!
   * \brief Defines the call transformation for AlterOpLayout pass. The new layouts are defined by
   * used for different targets using a packed func.
   * \param ref_call The original call.
   * \param new_args The traversed/recursed args to the call.
   * \return The new Call after calling the packed func.
   */
  Call CallWithNewLayouts(const Call& ref_call, const std::vector<Expr>& new_args) override {
    static auto falter_layout = Op::GetAttrMap<FTVMAlterOpLayout>("FTVMAlterOpLayout");
    Op op = Downcast<Op>(ref_call->op);

    Expr new_e;
    bool modified = false;

    // Warning(@Soo): Second condition is added to prevent CuDNN op (Conv2d)
    // from being altered to Winograd conv
    // MKL (dense) op should be prevented from being altered to nn.contrib_dense_pack
    bool is_cudnn_or_cublas_or_mkl = GetBackendFromAnnotation(ref_call->backend) == "cudnn";
    is_cudnn_or_cublas_or_mkl |= GetBackendFromAnnotation(ref_call->backend) == "cublas";
    is_cudnn_or_cublas_or_mkl |= GetBackendFromAnnotation(ref_call->backend) == "mkl";

    if (falter_layout.count(op) && !is_cudnn_or_cublas_or_mkl) {
      tvm::Array<tvm::te::Tensor> tinfos;
      for (auto expr : ref_call->args) {
        auto ttype = expr->type_as<TensorTypeNode>();
        tinfos.push_back(tvm::te::placeholder(ttype->shape, ttype->dtype));
      }
      // TODO(@kevinthesun, @icemelon9): This won't work if inputs/outputs are dynamic shapes.
      //   Probably we need to disable the AlterOpLayout when compiling dynamic models.
      Expr altered_value =
          falter_layout[op](ref_call->attrs, new_args, tinfos, ref_call->checked_type());

//      std::cerr << "new_args ===================================" << std::endl;
//      for (auto & elem : new_args) std::cerr << elem << ", ";
//      std::cerr << std::endl;

//      auto print_call = tvm::runtime::Registry::Get("collage.optimizer.print_attr_args");
//      (*print_call)(ref_call);
//      std::cerr << "OP, tinfos, checked_type : " << op << " , " << ref_call->attrs <<
//                " , " << tinfos << " , " << ref_call->checked_type() << std::endl;
      if (altered_value.defined()) {
        new_e = altered_value;
//        std::cerr << "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$" << std::endl;
//        std::cerr << "OP : " << new_e.as<CallNode>()->op << std::endl;
        modified = true;
      }
    }
    if (!modified) {
      new_e = Call(ref_call->op, new_args, ref_call->attrs);
    }

    MutateBackend(new_e, ref_call->backend);
    const CallNode* new_call = new_e.as<CallNode>();
    ICHECK(new_call) << "Can only replace the original operator with another call node";
    return GetRef<Call>(new_call);
  }

  using ContainerType = AlterTransformMemorizerNode;
};

/*!
 * Limitations:
 * 1. The altered op should have the same number of arguments as the previous one.
 * 2. Do not support nested tuple arguments.
 */
Expr AlterOpLayout(const Expr& expr) {
  // Warning(@Soo): Ignore this pass if this is custom fusion pass because
  // we shouldn't alter op layout before we make op assignment decision
  // e.g., we can't offload conv2d_winograd to TensorRT

  // Update(@Soo): We can always do AlterOpLayout pass no matter whether it's custom fusion or not
//  const FunctionNode* fn_node = static_cast<const FunctionNode*>(expr.get());
//  if (fn_node->GetAttr<IntImm>(attr::kCustomFusionPass).defined()
//      && !fn_node->GetAttr<IntImm>(attr::kAllowAlterOp).defined()) {
//    std::cerr << "Pass AlterOpLayout because it's custom pass" << std::endl;
//    return expr;
//  } else {
//    std::cerr << "AlterOpLayout will be executed!" << std::endl;
//  }

  // TODO(@icemelon9): need to rerun type inference after applying an alter op.
  AlterTransformMemorizer alterMemorizer(make_object<AlterTransformMemorizerNode>());
  auto fcontext = [&](const Call& call) -> ObjectRef { return alterMemorizer; };
//  std::cerr << "[AlterOpLayout] Input expr: " << expr << std::endl;
  auto expr_out = ForwardRewrite(expr, LayoutRewriter<AlterTransformMemorizer>, fcontext);
//  std::cerr << "[AlterOpLayout] Output expr: " << expr_out << std::endl;
  return expr_out;
}

}  // namespace alter_op_layout

namespace transform {

Pass AlterOpLayout() {
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
      [=](Function f, IRModule m, PassContext pc) {
        return Downcast<Function>(relay::alter_op_layout::AlterOpLayout(f));
      };
  return CreateFunctionPass(pass_func, 3, "AlterOpLayout", {"InferType"});
}

TVM_REGISTER_GLOBAL("relay._transform.AlterOpLayout").set_body_typed(AlterOpLayout);

}  // namespace transform

}  // namespace relay
}  // namespace tvm
