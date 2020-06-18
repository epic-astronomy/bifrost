#ifndef BF_ACORR_H_INCLUDE_GUARD_
#define BF_ACORR_H_INCLUDE_GUARD_

// Bifrost Includes
#include <bifrost/common.h>
#include <bifrost/memory.h>
#include <bifrost/array.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct BFaCorr_impl* BFacorr;

BFstatus bfaCorrCreate(BFacorr* plan);
BFstatus bfaCorrInit(BFacorr       plan,
                      BFarray const* positions,
                      BFbool         polmajor);
BFstatus bfaCorrSetStream(BFacorr    plan,
                           void const* stream);
BFstatus bfaCorrExecute(BFacorr          plan,
                         BFarray const* in,
                         BFarray const* out);
BFstatus bfaCorrDestroy(BFacorr plan);

#ifdef __cplusplus
}
#endif

#endif // BF_ACORR_H_INCLUDE_GUARD
