#ifndef GDN_LQ_EPILOGUE_TILING_H
#define GDN_LQ_EPILOGUE_TILING_H

#include "register/tilingdata_base.h"
#include "tiling_base/error_log.h"
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(GdnLqEpilogueTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, numInstances);
    TILING_DATA_FIELD_DEF(uint32_t, instancesPerCore);
    TILING_DATA_FIELD_DEF(uint32_t, remainderCores);
    TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum);
    TILING_DATA_FIELD_DEF(uint32_t, h);
    TILING_DATA_FIELD_DEF(uint32_t, nk);
    TILING_DATA_FIELD_DEF(int64_t, sSlot);
    TILING_DATA_FIELD_DEF(float, scale);
END_TILING_DATA_DEF;

struct GdnLqEpilogueCompileInfo {
    uint32_t totalCoreNum = 0;
};

REGISTER_TILING_DATA_CLASS(GdnLqEpilogue, GdnLqEpilogueTilingData)

}  // namespace optiling

#endif  // GDN_LQ_EPILOGUE_TILING_H
