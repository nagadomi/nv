/*
 * This file is part of libnv.
 *
 * Copyright (C) 2012 nagadomi@nurs.or.jp
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

#ifndef NV_IP_BGSEG_H
#define NV_IP_BGSEG_H
#include "nv_core.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
	int rows;
	int cols;
	int frame_rows;
	int frame_cols;
	int init_1st;
	int init_2nd;
	int init_1st_finished;
	int init_2nd_finished;
	float zeta;
	float bg_v;
	float fg_v;
	int size;
	nv_matrix_t *av;
	nv_matrix_t *sgm;
} nv_bgseg_t;

#define NV_BGSEG_ZETA 10.0f
#define NV_BGSEG_SIZE 320
#define NV_BGSEG_BG_V (1.0f / 20.0f)
#define NV_BGSEG_FG_V (1.0f / 80.0f)

nv_bgseg_t *nv_bgseg_alloc(int rows, int cols,
						   float zeta, float bg_v, float fg_v, int size);
void nv_bgseg_free(nv_bgseg_t **bg);

void nv_bgseg_init_1st_update(nv_bgseg_t *bg, const nv_matrix_t *frame);
void nv_bgseg_init_1st_finish(nv_bgseg_t *bg);
void nv_bgseg_init_2nd_update(nv_bgseg_t *bg, const nv_matrix_t *frame);
void nv_bgseg_init_2nd_finish(nv_bgseg_t *bg);	
void nv_bgseg_update(nv_bgseg_t *bg, nv_matrix_t *mask, const nv_matrix_t *frame);

#ifdef __cplusplus
}
#endif

#endif

