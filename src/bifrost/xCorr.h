#ifndef BF_CORR_H_INCLUDE_GUARD_
#define BF_CORR_H_INCLUDE_GUARD_

// Bifrost Includes
#include <bifrost/common.h>
#include <bifrost/memory.h>
#include <bifrost/array.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct BFcorr_impl* BFcorr;

BFstatus bfCorrCreate(BFcorr* plan);
BFstatus bfCorrInit(BFcorr       plan,
                      BFsize         ngrid,
                      BFbool         polmajor);
BFstatus bfCorrSetStream(BFcorr    plan,
                           void const* stream);
BFstatus bfCorrExecute(BFcorr          plan,
                         BFarray const* in,
                         BFarray const* out);
BFstatus bfCorrDestroy(BFcorr plan);

#ifdef __cplusplus
}
#endif

#endif // BF_CORR_H_INCLUDE_GUARD
