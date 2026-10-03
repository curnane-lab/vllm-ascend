#ifndef GDN_SKETCH_PROBE_TILING_H
#define GDN_SKETCH_PROBE_TILING_H

#include "register/tilingdata_base.h"
#include "tiling_base/error_log.h"
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(GdnSketchProbeTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, numInstances);
    TILING_DATA_FIELD_DEF(uint32_t, instancesPerCore);
    TILING_DATA_FIELD_DEF(uint32_t, remainderCores);
    TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum);
    TILING_DATA_FIELD_DEF(uint32_t, mode);
END_TILING_DATA_DEF;

struct GdnSketchProbeCompileInfo {
    uint32_t totalCoreNum = 0;
};

REGISTER_TILING_DATA_CLASS(GdnSketchProbe, GdnSketchProbeTilingData)

}  // namespace optiling

#endif  // GDN_SKETCH_PROBE_TILING_H
