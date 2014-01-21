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

#ifndef NV_NUM_EIGEN_H
#define NV_NUM_EIGEN_H
#include "nv_core.h"
#ifdef __cplusplus
extern "C" {
#endif

int nv_eigen(nv_matrix_t *eigen_vec, 
			 nv_matrix_t *eigen_val,
			 const nv_matrix_t *mat,
			 int n,
			 int max_epoch);
int nv_eigen_sym(nv_matrix_t *eigen_vec, 
				 nv_matrix_t *eigen_val,
				 const nv_matrix_t *mat,
				 int max_epoch);

#ifdef __cplusplus
}
#endif
#endif
