#ifndef BF_XCORR_H_INCLUDE_GUARD_
#define BF_XCORR_H_INCLUDE_GUARD_

// Bifrost Includes
#include <bifrost/common.h>
#include <bifrost/memory.h>
#include <bifrost/array.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct BFxcorr_impl* BFxcorr;

BFstatus bfxCorrCreate(BFxcorr* plan);
BFstatus bfxCorrInit(BFxcorr       plan,
                      BFsize         ngrid,
                      BFbool         polmajor);
BFstatus bfxCorrSetStream(BFxcorr    plan,
                           void const* stream);
BFstatus bfxCorrExecute(BFxcorr          plan,
                         BFarray const* in,
                         BFarray const* out);
BFstatus bfxCorrDestroy(BFxcorr plan);

#ifdef __cplusplus
}
#endif

#endif // BF_XCORR_H_INCLUDE_GUARD
