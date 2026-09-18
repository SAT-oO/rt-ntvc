#ifndef RTNV_DECODE_H
#define RTNV_DECODE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct Engine RtnvEngine;

RtnvEngine *rtnv_engine_load(const char *weight_path, const float *global_freq, int vocab);
void rtnv_engine_free(RtnvEngine *eng);

int rtnv_encode(
    RtnvEngine *eng,
    const int32_t *tokens,
    int n_frames,
    int s,
    uint8_t *out,
    size_t out_cap,
    size_t *out_len,
    int adaptive);

/* Returns n_frames on success, <0 on error. n_frames=0 means read from bitstream. */
int rtnv_decode(
    RtnvEngine *eng,
    const uint8_t *bits,
    size_t bits_len,
    int n_frames,
    int32_t *tokens_out);

size_t rtnv_hot_allocs(void);

#ifdef __cplusplus
}
#endif

#endif
