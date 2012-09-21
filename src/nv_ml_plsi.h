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

#ifndef NV_PLSI_H
#define NV_PLSI_H
#include "nv_core.h"

typedef struct {
	int k;
	int d;
	int w;
	nv_matrix_t *dz;
	nv_matrix_t *wz;
	nv_matrix_t *z;
} nv_plsi_t;

nv_plsi_t *nv_plsi_alloc(int d, int w, int k);
void nv_plsi_free(nv_plsi_t **p);
void nv_plsi_init(nv_plsi_t *p);
void nv_plsi(nv_plsi_t *p, const nv_matrix_t *data, int step);
void nv_plsi_emstep(nv_plsi_t *p, const nv_matrix_t *data, float tem_beta);

#endif
