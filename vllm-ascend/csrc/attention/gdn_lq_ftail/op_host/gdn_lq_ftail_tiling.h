#ifndef GDN_LQ_FTAIL_TILING_H
#define GDN_LQ_FTAIL_TILING_H

#include "register/tilingdata_base.h"
#include "tiling_base/error_log.h"
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(GdnLqFtailTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, numInstances);
    TILING_DATA_FIELD_DEF(uint32_t, instancesPerCore);
    TILING_DATA_FIELD_DEF(uint32_t, remainderCores);
    TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum);
    TILING_DATA_FIELD_DEF(uint32_t, h);
    TILING_DATA_FIELD_DEF(int64_t, sSlot);
    TILING_DATA_FIELD_DEF(uint32_t, phase);
END_TILING_DATA_DEF;

struct GdnLqFtailCompileInfo {
    uint32_t totalCoreNum = 0;
};

REGISTER_TILING_DATA_CLASS(GdnLqFtail, GdnLqFtailTilingData)

}  // namespace optiling

#endif  // GDN_LQ_FTAIL_TILING_H
