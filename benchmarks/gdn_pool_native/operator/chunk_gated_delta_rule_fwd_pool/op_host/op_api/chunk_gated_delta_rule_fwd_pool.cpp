/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * CANN Open Software License Agreement Version 2.0.
 */
#include "chunk_gated_delta_rule_fwd_pool.h"

#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(ChunkGatedDeltaRuleFwdPool);

namespace {
const aclTensor *ConvertMetadata(const aclIntArray *values, aclOpExecutor *executor)
{
    if (values == nullptr) {
        return nullptr;
    }
    const aclTensor *tensor = executor->ConvertToTensor(values, DataType::DT_INT64);
    if (tensor != nullptr) {
        auto *mutableTensor = const_cast<aclTensor *>(tensor);
        mutableTensor->SetStorageFormat(Format::FORMAT_ND);
        mutableTensor->SetViewFormat(Format::FORMAT_ND);
        mutableTensor->SetOriginalFormat(Format::FORMAT_ND);
    }
    return tensor;
}
} // namespace

const std::array<const aclTensor *, 4> ChunkGatedDeltaRuleFwdPool(
    const aclTensor *q,
    const aclTensor *k,
    const aclTensor *v,
    const aclTensor *beta,
    const aclTensor *aStorage,
    const aclTensor *rawG,
    const aclTensor *gkOptional,
    const aclTensor *poolState,
    const aclIntArray *cuSeqlensOptional,
    const aclIntArray *chunkIndicesOptional,
    const aclTensor *sourceSlots,
    const aclTensor *destinationSlots,
    const aclTensor *hasInitial,
    bool outputFinalState,
    int64_t chunkSize,
    double scale,
    int64_t poolHeadStride,
    int64_t poolSlotStride,
    const aclTensor *oOut,
    const aclTensor *finalStateOut,
    const aclTensor *gCumsumBthOut,
    const aclTensor *aOut,
    aclOpExecutor *executor)
{
    const aclTensor *cuSeqlens = ConvertMetadata(cuSeqlensOptional, executor);
    const aclTensor *chunkIndices = ConvertMetadata(chunkIndicesOptional, executor);
    if ((cuSeqlensOptional != nullptr && cuSeqlens == nullptr) ||
        (chunkIndicesOptional != nullptr && chunkIndices == nullptr)) {
        return {nullptr, nullptr, nullptr, nullptr};
    }

    const aclError ret = ADD_TO_LAUNCHER_LIST_AICORE(
        ChunkGatedDeltaRuleFwdPool,
        OP_INPUT(q, k, v, beta, aStorage, rawG, gkOptional, poolState,
                 cuSeqlens, chunkIndices, sourceSlots, destinationSlots, hasInitial),
        OP_OUTPUT(oOut, finalStateOut, gCumsumBthOut, aOut),
        OP_ATTR(outputFinalState, chunkSize, scale, poolHeadStride, poolSlotStride));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "ADD_TO_LAUNCHER_LIST_AICORE ChunkGatedDeltaRuleFwdPool failed.");
        return {nullptr, nullptr, nullptr, nullptr};
    }
    return {oOut, finalStateOut, gCumsumBthOut, aOut};
}

} // namespace l0op
