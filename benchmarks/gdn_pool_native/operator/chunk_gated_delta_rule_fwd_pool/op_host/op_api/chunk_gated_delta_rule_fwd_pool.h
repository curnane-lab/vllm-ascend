/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * CANN Open Software License Agreement Version 2.0.
 */
#pragma once

#include "opdev/op_executor.h"

namespace l0op {
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
    aclOpExecutor *executor);
}
