#ifndef GDN_SKETCH_STEP_TILING_H
#define GDN_SKETCH_STEP_TILING_H

#include "register/tilingdata_base.h"
#include "tiling_base/error_log.h"
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(GdnSketchStepTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, numInstances);
    TILING_DATA_FIELD_DEF(uint32_t, instancesPerCore);
    TILING_DATA_FIELD_DEF(uint32_t, remainderCores);
    TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum);
    TILING_DATA_FIELD_DEF(uint32_t, batch);
    TILING_DATA_FIELD_DEF(uint32_t, hv);
    TILING_DATA_FIELD_DEF(uint32_t, h);
    TILING_DATA_FIELD_DEF(float, scale);
    TILING_DATA_FIELD_DEF(int64_t, sQkv);
    TILING_DATA_FIELD_DEF(int64_t, sA);
    TILING_DATA_FIELD_DEF(int64_t, sB);
    TILING_DATA_FIELD_DEF(int64_t, sStSlot);
    TILING_DATA_FIELD_DEF(int64_t, sStHead);
    TILING_DATA_FIELD_DEF(int64_t, sDcSlot);
    TILING_DATA_FIELD_DEF(int64_t, sDcHead);
    TILING_DATA_FIELD_DEF(int64_t, sKcSlot);
    TILING_DATA_FIELD_DEF(int64_t, sKcHead);
    TILING_DATA_FIELD_DEF(int64_t, sGcSlot);
    TILING_DATA_FIELD_DEF(int64_t, sGcHead);
    TILING_DATA_FIELD_DEF(int64_t, sSu);
    TILING_DATA_FIELD_DEF(int64_t, sSm);
    TILING_DATA_FIELD_DEF(int64_t, sSf);
END_TILING_DATA_DEF;

struct GdnSketchStepCompileInfo {
    uint32_t totalCoreNum = 0;
};

REGISTER_TILING_DATA_CLASS(GdnSketchStep, GdnSketchStepTilingData)

}  // namespace optiling

#endif  // GDN_SKETCH_STEP_TILING_H
