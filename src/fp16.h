#pragma once

#ifdef __ARM_FP16_FORMAT_IEEE

typedef __fp16 fp16;

#define fp16_read(p) (*(p))

#define fp16_write(p, v) *(p) = (v)

#else

#include <x86intrin.h>

typedef uint16_t fp16;

#define fp16_read(p) _cvtsh_ss(*(p))

#define fp16_write(p, v) *(p) = _cvtss_sh(v)

#endif
