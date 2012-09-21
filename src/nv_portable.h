/*
 * This file is part of libnv.
 *
 * Copyright (C) 2008-2012 nagadomi@nurs.or.jp
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License,
 * or any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef NV_PORTABLE_H
#define NV_PORTABLE_H

#ifdef __cplusplus
extern "C" {
#endif

void nv_initialize(void);

#ifdef _MSC_VER
#  define NV_MSVC  1
#  define NV_GCC   0
#  define NV_MINGW 0	
#elif __GNUC__
#  define NV_MSVC 0
#  define NV_GCC  1
#  if (defined(__MINGW32__) || defined(__MINGW64__))
#    define NV_MINGW 1
#  else
#    define NV_MINGW 0
#  endif
#else
#  error "unknown compiler"
#endif

#if (defined(_WIN32) || defined(_WIN64))
#  define NV_WINDOWS 1
#  define NV_POSIX   0
#  include <windows.h>
#  include <malloc.h>
#else //  (defined(__linux__) || defined(__FreeBSD__))
#  define NV_WINDOWS 0	
#  define NV_POSIX   1
#endif

/* stdc */
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdarg.h>
#include <math.h>
#include <float.h>
#include <ctype.h>
#include <assert.h>
#include <errno.h>
#include <time.h>

/* C99 stdint */
#if NV_MSVC
#  include "inttypes.h"
#else
#  include <inttypes.h>
#endif

/* OpenMP */
#if (NV_MSVC && defined(_OPENMP))
#  ifdef _DEBUG
#    undef _DEBUG
#    include <omp.h>
#    define _DEBUG
#  else
#    include <omp.h>
#  endif
#elif (defined(_OPENMP))
#  include <omp.h>	
#endif
	
/* assert & backtrace */
#if NV_ENABLE_GLIBC_BACKTRACE
#  include <execinfo.h>
#  define NV_BACKTRACE											\
	{															\
		void *trace_[512];										\
		int n_ = backtrace(trace_, sizeof(trace_) / sizeof(trace_[0])); \
		backtrace_symbols_fd(trace_, n_, 2);							\
		fflush(stderr);													\
	}
#else
#  define NV_BACKTRACE	{ fprintf(stderr, "%s(%d)\n", __FILE__, __LINE__); fflush(stderr); }
#endif

#ifdef NDEBUG
#  define NV_ASSERT(x) 
#else
#  define NV_ASSERT(x) \
	if (!(x)) { \
		NV_BACKTRACE;							\
		assert(x); \
	} 
#endif

/* memory align */
#if NV_MSVC
#  define NV_ALIGNED(typed, variable, align_size) __declspec(align(align_size)) typed variable
#else	// gcc
#  define NV_ALIGNED(typed, variable, align_size) typed variable __attribute__((aligned(align_size)))
#endif
	
#if NV_WINDOWS
int nv_aligned_malloc(void **memptr, size_t alignment, size_t size);
#  if NV_MINGW  
#    define nv_aligned_realloc(memptr, alignment, size) __mingw_aligned_realloc(memptr, size, alignment)
#    define nv_aligned_free(memptr) __mingw_aligned_free(memptr)
#  else
#    define nv_aligned_realloc(memptr, alignment, size) _aligned_realloc(memptr, size, alignment)
#    define nv_aligned_free(memptr) _aligned_free(memptr)
#  endif  
#else	
#  define nv_aligned_malloc(memptr, alignment, size) posix_memalign(memptr, alignment, size)
#  define nv_aligned_realloc(memptr, alignment, size) nv_realloc(memptr, size)
#  define nv_aligned_free(memptr) nv_free(memptr)
#endif

/* inline */
#if NV_MSVC
#define NV_INLINE __inline
#else
#define NV_INLINE inline
#endif

char *nv_strerror_r(int errnum, char *buf, size_t buflen);
struct tm *nv_gmtime_r(const time_t *clock, struct tm *result);
struct tm *nv_localtime_r(const time_t *clock, struct tm *result);
time_t nv_mkgmtime(struct tm *tm);

int nv_vsnprintf(char *str, size_t size, const char *format, va_list ap);
void nv_enable_sse2_math(void);

void nv_setenv(const char *name, const char *value);
void nv_unsetenv(const char *name);
const char *nv_getenv(const char *name);

#if NV_MSVC
#  define nv_snprintf         _snprintf
#  define nv_strtoll          _strtoi64
#  define nv_strcasecmp(a, b) _stricmp(a, b)
#else
#  define nv_snprintf         snprintf
#  define nv_strtoll          strtoll
#  define nv_strcasecmp(a, b) strcasecmp(a, b)
#endif

/* isnan */
#if NV_GCC	
#  define NV_ISNAN(x) __builtin_isnan(x)
#elif NV_MSVC
#  define NV_ISNAN(x) _isnan(x)
#else
#  define NV_ISNAN(x) 0
#endif

#if NV_GCC	
#  define NV_RESTRICT __restrict__
#else
#  define NV_RESTRICT
#endif

#if NV_WINDOWS
#  ifdef NV_INTERNAL
#    define NV_DECLARE_DATA __declspec(dllexport)
#  else
#    define NV_DECLARE_DATA __declspec(dllimport)
#  endif
#else
#    define NV_DECLARE_DATA 
#endif

int nv_cuda_enabled(void);
void nv_cuda_set(int onoff);
	
#ifdef __cplusplus
}
#endif

/* SIMD */
#if NV_ENABLE_SSE
#  include <xmmintrin.h>
#endif
#if NV_ENABLE_SSE2
#  include <emmintrin.h>
#endif
#if NV_ENABLE_SSE3
#  include <pmmintrin.h>
#  if NV_MSVC
#    include <intrin.h>
#  endif
#endif
#if NV_ENABLE_SSE4_1
#  include <smmintrin.h>
#endif
#if NV_ENABLE_SSE4_2
#  include <smmintrin.h>
#endif
#if NV_ENABLE_AVX
#  include <immintrin.h>
#endif
#if NV_ENABLE_SSE4_2
#  include <nmmintrin.h>
#endif
#if (NV_ENABLE_SSE4_2 || NV_ENABLE_POPCNT)
#  if NV_MSVC  
#    include <intrin.h>
#    define NV_POPCNT_U32(x) _mm_popcnt_u32(x)
#    ifdef _M_X64
#      define NV_POPCNT_U64(x) _mm_popcnt_u64(x)
#    else
#      define NV_POPCNT_U64(x) (uint64_t)(_mm_popcnt_u32((uint32_t)((x) & 0xFFFFFFFFULL)) + _mm_popcnt_u32((uint32_t)((x) >> 32ULL)))
#    endif
#  elif NV_GCC
#    define NV_POPCNT_U64(x) __builtin_popcountll(x)
#    define NV_POPCNT_U32(x) __builtin_popcount(x)
#  else
#    define NV_POPCNT_U64(x) nv_popcnt_u64(x)
#    define NV_POPCNT_U32(x) nv_popcnt_u32(x)
#  endif
#else
#  define NV_POPCNT_U64(x) nv_popcnt_u64(x)
#  define NV_POPCNT_U32(x) nv_popcnt_u32(x)
#endif

#if NV_ENABLE_SSE
#  define NV_PREFETCH_NTA(p) _mm_prefetch(p, _MM_HINT_NTA)
#  define NV_PREFETCH_T0(p)	 _mm_prefetch(p, _MM_HINT_T0)
#  define NV_PREFETCH_T2(p)	 _mm_prefetch(p, _MM_HINT_T2)
#else
#  define NV_PREFETCH_NTA(p)
#  define NV_PREFETCH_T0(p)
#  define NV_PREFETCH_T2(p)
#endif


#endif
