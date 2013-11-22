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

#ifndef NV_CORE_UTIL_H
#define NV_CORE_UTIL_H
#include "nv_config.h"
#ifdef __cplusplus
extern "C" {
#endif
#include <math.h>

#define NV_PI        3.141593f     /** 円周率 */
#define NV_PI_DIV2   1.570796f     /* NV_PI / 2.0 */
#define NV_PI2_INV   1.591549E-01f /* 1.0f / (NV_PI * 2.0f) */

#define NV_SQRT2     1.414214f     /** √2   */
#define NV_SQRT2_INV 7.071068E-01f /** 1/√2 */

typedef struct {
	int y;
	int x;
	int r;
	float theta;
} nv_circle_t;

typedef struct {
	int y;
	int x;
	int height;
	int width;
} nv_rect_t;

typedef struct {
	int y;
	int x;
} nv_point_t;

typedef struct {
	int height;
	int width;
} nv_image_size_t;

typedef struct {
	float v[4];
} nv_color_t;

typedef struct {
	int   i;
	float f;
} nv_int_float_t;

uint32_t nv_popcnt_u32(uint32_t x);
uint64_t nv_popcnt_u64(uint64_t x);
int nv_omp_thread_id(void);
int nv_omp_procs(void);
void nv_omp_set_procs(int n);
float nv_log2(float v);
float nv_sign(float x);

#define NV_SIGMOID(a) (1.0f / (1.0f + expf(-(a))))
#define NV_SIGN(x) (((x) > 0.0f) ? 1.0f:-1.0f)
	
#define NV_SWAP(type, a, b) \
{ \
	type nv_swap_temp = (a); \
	(a) = (b); \
	(b) = nv_swap_temp; \
}

#define NV_FLOOR_INT(x) ((int)(x))
#define NV_FLOOR(x) ((float)NV_FLOOR_INT(x))
#define NV_ROUND_INT(x) (NV_FLOOR_INT((0.5f + (x))))
#define NV_MAX(a, b) ((a) > (b) ? (a):(b))
#define NV_MIN(a, b) ((a) < (b) ? (a):(b))

#define NV_VALID_SSE(mat) ((((mat)->step & 0x3) == 0) && (((size_t)((mat)->v) & 0xf) == 0))
#define NV_VALID_AVX(mat) ((((mat)->step & 0x3) == 0) && (((size_t)((mat)->v) & 0x1f) == 0))	
	
#ifdef __cplusplus
}
#endif


#endif
