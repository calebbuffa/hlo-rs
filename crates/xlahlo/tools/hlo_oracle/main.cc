#include <cstdio>
#include <iostream>
#include <iterator>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_original_value.h"
#include "xla/hlo/ir/replica_group.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/literal.h"
#include "xla/service/hlo.pb.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/hlo_verifier.h"
#include "xla/service/shape_inference.h"
#include "xla/shape_util.h"
#include "xla/xla.pb.h"

template <typename T>
int WriteAttribute(const absl::StatusOr<T>& value) {
  if (!value.ok()) {
    std::cerr << "parse: " << value.status() << "\n";
    return 2;
  }
  if (!value->SerializeToOstream(&std::cout)) {
    std::cerr << "Cannot write attribute protobuf\n";
    return 1;
  }
  return 0;
}

absl::StatusOr<std::unique_ptr<xla::HloModule>> ImportProto(
    const std::string& bytes) {
  xla::HloModuleProto proto;
  if (!proto.ParseFromString(bytes)) {
    return absl::InvalidArgumentError("Cannot decode HLO module protobuf");
  }
  auto config =
      xla::HloModule::CreateModuleConfigFromProto(proto, xla::DebugOptions{});
  if (!config.ok()) {
    return config.status();
  }
  return xla::HloModule::CreateFromProto(proto, *config);
}

absl::Status RemoveUnusedParameter(xla::HloModule* module, bool allow_entry) {
  xla::HloInstruction* target = nullptr;
  for (auto* computation : module->computations()) {
    if (computation == module->entry_computation() && !allow_entry) {
      continue;
    }
    for (auto* instruction : computation->instructions()) {
      if (instruction->opcode() == xla::HloOpcode::kParameter &&
          instruction->IsDead()) {
        target = instruction;
        break;
      }
    }
    if (target != nullptr) {
      break;
    }
  }
  if (target == nullptr) {
    return absl::InvalidArgumentError("No eligible unused parameter");
  }
  auto callers = [module](const xla::HloComputation* callee) {
    std::vector<xla::HloInstruction*> result;
    for (auto* computation : module->computations()) {
      for (auto* instruction : computation->instructions()) {
        for (auto* called : instruction->called_computations()) {
          if (called == callee) {
            result.push_back(instruction);
            break;
          }
        }
      }
    }
    return result;
  };
  auto status = target->parent()->RemoveInstructionAndUnusedOperands(
      target, std::nullopt, false, callers, allow_entry);
  if (!status.ok()) {
    return status;
  }
  if (allow_entry) {
    module->mutable_config().SetDefaultComputationLayout(
        module->entry_computation()->ComputeProgramShape());
  }
  return absl::OkStatus();
}

absl::StatusOr<xla::Shape> InferCore(const xla::HloInstruction* root) {
  if (root->operand_count() == 0) {
    return absl::InvalidArgumentError("Expected an operand for core inference");
  }
  const auto& operand = root->operand(0)->shape();
  switch (root->opcode()) {
    case xla::HloOpcode::kReshape:
      return xla::ShapeInference::InferReshapeShape(
          operand, root->shape().dimensions(), -1);
    case xla::HloOpcode::kTranspose:
      return xla::ShapeInference::InferTransposeShape(operand,
                                                      root->dimensions());
    case xla::HloOpcode::kReducePrecision:
      return xla::ShapeInference::InferReducePrecisionShape(
          operand, root->exponent_bits(), root->mantissa_bits());
    case xla::HloOpcode::kCholesky:
      return xla::ShapeInference::InferCholeskyShape(operand);
    case xla::HloOpcode::kTriangularSolve:
      if (root->operand_count() != 2) {
        return absl::InvalidArgumentError("Expected matrix and rhs operands");
      }
      return xla::ShapeInference::InferTriangularSolveShape(
          operand, root->operand(1)->shape(), root->triangular_solve_options());
    case xla::HloOpcode::kFft:
      return xla::ShapeInference::InferFftShape(operand, root->fft_type(),
                                                root->fft_length());
    case xla::HloOpcode::kGetDimensionSize:
    case xla::HloOpcode::kSetDimensionSize: {
      const auto wire = root->ToProto();
      if (wire.dimensions_size() != 1) {
        return absl::InvalidArgumentError("Expected one dimension");
      }
      if (root->opcode() == xla::HloOpcode::kGetDimensionSize) {
        return xla::ShapeInference::InferGetDimensionSizeShape(
            operand, wire.dimensions(0));
      }
      if (root->operand_count() != 2) {
        return absl::InvalidArgumentError("Expected data and size operands");
      }
      return xla::ShapeInference::InferSetDimensionSizeShape(
          operand, root->operand(1)->shape(), wire.dimensions(0));
    }
    case xla::HloOpcode::kDynamicUpdateSlice: {
      if (root->operand_count() < 2) {
        return absl::InvalidArgumentError("Expected data and update operands");
      }
      std::vector<xla::Shape> indices;
      for (int i = 2; i < root->operand_count(); ++i) {
        indices.push_back(root->operand(i)->shape());
      }
      return xla::ShapeInference::InferDynamicUpdateSliceShape(
          operand, root->operand(1)->shape(), indices);
    }
    case xla::HloOpcode::kDynamicReshape: {
      std::vector<const xla::Shape*> sizes;
      for (int i = 1; i < root->operand_count(); ++i) {
        sizes.push_back(&root->operand(i)->shape());
      }
      const auto dynamic = root->shape().dynamic_dimensions();
      return xla::ShapeInference::InferDynamicReshapeShape(
          operand, sizes, root->shape().dimensions(),
          std::vector<bool>(dynamic.begin(), dynamic.end()));
    }
    default:
      return absl::InvalidArgumentError("Unsupported core inference opcode");
  }
}

int main(int argc, char** argv) {
  if (argc != 2 ||
      (std::string(argv[1]) != "parse" && std::string(argv[1]) != "verify" &&
       std::string(argv[1]) != "verify-mixed" &&
       std::string(argv[1]) != "verify-deadlocks" &&
       std::string(argv[1]) != "infer-core" &&
       std::string(argv[1]) != "scalar-bytes" &&
       std::string(argv[1]) != "import-proto" &&
       std::string(argv[1]) != "verify-proto" &&
       std::string(argv[1]) != "remove-parameter" &&
       std::string(argv[1]) != "remove-entry-parameter" &&
       std::string(argv[1]) != "remove-unused-operands" &&
       std::string(argv[1]) != "effects" &&
       std::string(argv[1]) != "parse-original-value" &&
       std::string(argv[1]) != "parse-frontend-attributes" &&
       std::string(argv[1]) != "parse-statistics" &&
       std::string(argv[1]) != "parse-parameter-replication" &&
       std::string(argv[1]) != "parse-shape-list" &&
       std::string(argv[1]) != "parse-layout" &&
       std::string(argv[1]) != "parse-padding" &&
       std::string(argv[1]) != "parse-replica-groups" &&
       std::string(argv[1]) != "parse-device-list" &&
       std::string(argv[1]) != "parse-module-config" &&
       std::string(argv[1]) != "parse-module-options" &&
       std::string(argv[1]) != "infer-reshape")) {
    std::cerr << "Usage: hlo-oracle "
                 "parse|verify|verify-mixed|verify-deadlocks|infer-reshape|infer-core|"
                 "verify-no-groups|scalar-bytes|import-proto|verify-proto|remove-parameter|"
                 "remove-entry-parameter|remove-unused-operands|effects|"
                 "parse-original-value|parse-frontend-attributes|"
                 "parse-statistics|parse-parameter-replication|"
                 "parse-shape-list|parse-layout|parse-padding|"
                 "parse-replica-groups|parse-device-list|parse-module-config|"
                 "parse-module-options < "
                 "input > output.pb\n";
    return 1;
  }
#ifdef _WIN32
  if (_setmode(_fileno(stdin), _O_BINARY) == -1 ||
      _setmode(_fileno(stdout), _O_BINARY) == -1) {
    std::cerr << "Cannot set binary stream mode\n";
    return 1;
  }
#endif
  const std::string text{std::istreambuf_iterator<char>(std::cin),
                         std::istreambuf_iterator<char>()};
  if (std::cin.bad()) {
    std::cerr << "Cannot read HLO input\n";
    return 1;
  }
  if (std::string(argv[1]) == "parse-frontend-attributes") {
    return WriteAttribute(xla::ParseFrontendAttributes(text));
  }
  if (std::string(argv[1]) == "parse-statistics") {
    return WriteAttribute(xla::ParseStatisticsViz(text));
  }
  if (std::string(argv[1]) == "parse-padding") {
    return WriteAttribute(xla::ParsePaddingConfig(text));
  }
  if (std::string(argv[1]) == "parse-replica-groups") {
    auto value = xla::ParseReplicaGroupsOnly(text);
    if (!value.ok()) {
      std::cerr << "parse: " << value.status() << "\n";
      return 2;
    }
    xla::CollectiveDeviceListProto wire;
    for (const auto& group : *value) {
      *wire.add_replica_groups() = group;
    }
    return WriteAttribute(absl::StatusOr<xla::CollectiveDeviceListProto>(wire));
  }
  if (std::string(argv[1]) == "parse-device-list") {
    auto value = xla::ParseCollectiveDeviceListBase(text);
    if (!value.ok()) {
      std::cerr << "parse: " << value.status() << "\n";
      return 2;
    }
    xla::HloInstructionProto wire;
    switch ((*value)->version()) {
      case xla::CollectiveDeviceListVersion::kListOfLists:
        *wire.mutable_collective_device_list() =
            static_cast<const xla::CollectiveDeviceList&>(**value).ToProto();
        break;
      case xla::CollectiveDeviceListVersion::kIota:
        *wire.mutable_iota_collective_device_list() =
            static_cast<const xla::IotaReplicaGroupList&>(**value).ToProto();
        break;
      case xla::CollectiveDeviceListVersion::kMeshAxes:
        *wire.mutable_mesh_axes_replica_group_list() =
            static_cast<const xla::MeshAxesReplicaGroupList&>(**value).ToProto();
        break;
    }
    return WriteAttribute(absl::StatusOr<xla::HloInstructionProto>(wire));
  }
  if (std::string(argv[1]) == "parse-layout") {
    auto value = xla::ParseLayout(text);
    if (!value.ok()) {
      std::cerr << "parse: " << value.status() << "\n";
      return 2;
    }
    return WriteAttribute(absl::StatusOr<xla::LayoutProto>(value->ToProto()));
  }
  if (std::string(argv[1]) == "parse-shape-list") {
    auto values = xla::ParseShapeList(text);
    if (!values.ok()) {
      std::cerr << "parse: " << values.status() << "\n";
      return 2;
    }
    std::vector<xla::Shape> shapes(values->begin(), values->end());
    return WriteAttribute(absl::StatusOr<xla::ShapeProto>(
        xla::ShapeUtil::MakeTupleShape(shapes).ToProto()));
  }
  if (std::string(argv[1]) == "parse-original-value") {
    auto value = xla::ParseOriginalValue(text);
    if (!value.ok()) {
      std::cerr << "parse: " << value.status() << "\n";
      return 2;
    }
    return WriteAttribute(
        absl::StatusOr<xla::OriginalValueProto>((*value)->ToProto()));
  }
  if (std::string(argv[1]) == "parse-parameter-replication") {
    auto values = xla::ParseParameterReplication(text);
    if (!values.ok()) {
      std::cerr << "parse: " << values.status() << "\n";
      return 2;
    }
    xla::ParameterReplication wire;
    for (bool value : *values) {
      wire.add_replicated_at_leaf_buffers(value);
    }
    return WriteAttribute(absl::StatusOr<xla::ParameterReplication>(wire));
  }
  const bool binary = std::string(argv[1]) == "import-proto" ||
                      std::string(argv[1]) == "verify-proto";
  auto parsed =
      binary ? ImportProto(text) : xla::ParseAndReturnUnverifiedModule(text);
  if (!parsed.ok()) {
    std::cerr << (binary ? "import: " : "parse: ") << parsed.status() << "\n";
    return 2;
  }
  if (std::string(argv[1]) == "parse-module-config") {
    const auto& config = (*parsed)->config();
    std::cout << config.replica_count() << " " << config.num_partitions() << " "
              << (config.use_spmd_partitioning() ? "true" : "false") << "\n";
    return 0;
  }
  if (std::string(argv[1]) == "parse-module-options") {
    const auto& config = (*parsed)->config();
    std::cout << (config.alias_passthrough_params() ? "true" : "false") << "\n";
    for (bool value : config.allow_spmd_sharding_propagation_to_parameters()) {
      std::cout << (value ? "true" : "false") << " ";
    }
    std::cout << "\n";
    for (bool value : config.allow_spmd_sharding_propagation_to_output()) {
      std::cout << (value ? "true" : "false") << " ";
    }
    std::cout << "\n";
    return 0;
  }
  if (std::string(argv[1]) == "effects") {
    std::cout
        << ((*parsed)->entry_computation()->root_instruction()->HasSideEffect()
                ? "1\n"
                : "0\n");
    return 0;
  }
  if (std::string(argv[1]) == "remove-parameter" ||
      std::string(argv[1]) == "remove-entry-parameter") {
    auto status = RemoveUnusedParameter(
        parsed->get(), std::string(argv[1]) == "remove-entry-parameter");
    if (!status.ok()) {
      std::cerr << "edit: " << status << "\n";
      return 2;
    }
  }
  if (std::string(argv[1]) == "remove-unused-operands") {
    auto* computation = (*parsed)->entry_computation();
    xla::HloInstruction* target = nullptr;
    for (auto* instruction : computation->instructions()) {
      if (instruction->name() == "dead") {
        target = instruction;
        break;
      }
    }
    if (target == nullptr) {
      std::cerr << "Expected entry instruction named dead\n";
      return 2;
    }
    auto status = computation->RemoveInstructionAndUnusedOperands(target);
    if (!status.ok()) {
      std::cerr << "edit: " << status << "\n";
      return 2;
    }
  }
  if (std::string(argv[1]) == "scalar-bytes") {
    const auto* root = (*parsed)->entry_computation()->root_instruction();
    if (root->opcode() != xla::HloOpcode::kConstant ||
        !root->shape().IsArrayExcludingBuffer() ||
        !root->shape().dimensions().empty()) {
      std::cerr << "Expected a scalar constant root\n";
      return 1;
    }
    const auto& literal = root->literal();
    std::cout.write(static_cast<const char*>(literal.untyped_data()),
                    literal.size_bytes());
    if (!std::cout) {
      std::cerr << "Cannot write scalar storage bytes\n";
      return 1;
    }
    return 0;
  }
  if (std::string(argv[1]) == "infer-reshape" ||
      std::string(argv[1]) == "infer-core") {
    const auto* root = (*parsed)->entry_computation()->root_instruction();
    if (std::string(argv[1]) == "infer-reshape" &&
        (root->opcode() != xla::HloOpcode::kReshape ||
         root->operand_count() != 1)) {
      std::cerr << "Expected a reshape root\n";
      return 1;
    }
    auto shape = InferCore(root);
    if (!shape.ok()) {
      std::cerr << "infer: " << shape.status() << "\n";
      return 4;
    }
    if (!shape->ToProto().SerializeToOstream(&std::cout)) {
      std::cerr << "Cannot write shape protobuf\n";
      return 1;
    }
    return 0;
  }
  if (std::string(argv[1]) == "verify" ||
      std::string(argv[1]) == "verify-mixed" ||
      std::string(argv[1]) == "remove-parameter" ||
      std::string(argv[1]) == "remove-entry-parameter" ||
      std::string(argv[1]) == "remove-unused-operands" ||
      std::string(argv[1]) == "verify-proto" ||
      std::string(argv[1]) == "verify-deadlocks" ||
      std::string(argv[1]) == "verify-no-groups") {
    xla::HloVerifier verifier(
        xla::HloVerifierOpts{}
            .WithLayoutSensitive(false)
            .WithAllowMixedPrecision(std::string(argv[1]) == "verify-mixed")
            .WithVerifyNoCollectiveDeadlocks(std::string(argv[1]) ==
                                             "verify-deadlocks")
            .WithCheckReplicaGroups(std::string(argv[1]) !=
                                    "verify-no-groups"));
    auto verified = verifier.Run(parsed->get());
    if (!verified.ok()) {
      std::cerr << "verify: " << verified.status() << "\n";
      return 3;
    }
  }
  if (!(*parsed)->ToProto().SerializeToOstream(&std::cout)) {
    std::cerr << "Cannot write HLO protobuf\n";
    return 1;
  }
  return 0;
}
