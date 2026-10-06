#ifndef GDN_LQ_GEMV_TILING_H
#define GDN_LQ_GEMV_TILING_H

#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "tiling_base/error_log.h"
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(GdnLqGemvTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, numInstances);
    TILING_DATA_FIELD_DEF(uint32_t, instancesPerCore);
    TILING_DATA_FIELD_DEF(uint32_t, remainderCores);
    TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum);
    TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, mmTilingData);
END_TILING_DATA_DEF;

struct GdnLqGemvCompileInfo {
    uint32_t totalCoreNum = 0;
};

REGISTER_TILING_DATA_CLASS(GdnLqGemv, GdnLqGemvTilingData)

}  // namespace optiling

#endif  // GDN_LQ_GEMV_TILING_H
