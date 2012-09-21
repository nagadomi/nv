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

#ifndef NV_IP_SHAPECONTEXT_H
#define NV_IP_SHAPECONTEXT_H

#include "nv_core.h"

#ifdef __cplusplus
extern "C" {
#endif

#define NV_SC_LOG_R_BIN 3
#define NV_SC_THETA_BIN 8

typedef struct {
	int n;
	nv_matrix_t *tan_angle;
	nv_matrix_t *coodinate;
	nv_matrix_t *radius;
	nv_matrix_t *sctx;
} nv_shapecontext_t;

nv_shapecontext_t *nv_shapecontext_alloc(int n);
void nv_shapecontext_free(nv_shapecontext_t **sctx);

float nv_shapecontext(const nv_shapecontext_t *img1,
					  const nv_shapecontext_t *img2);

void nv_shapecontext_feature(nv_shapecontext_t *sctx,
							const nv_matrix_t *img,
							float r);

float nv_shapecontext_distance(const nv_shapecontext_t *sctx1,
							   const nv_shapecontext_t *sctx2);


#ifdef __cplusplus
}
#endif

#endif

