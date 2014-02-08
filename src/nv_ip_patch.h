/*
 * This file is part of libnv.
 *
 * Copyright (C) 2014 nagadomi@nurs.or.jp
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
#ifndef NV_IP_PATCH_H
#define NV_IP_PATCH_H

#ifdef __cplusplus
extern "C" {
#endif
#include "nv_core.h"

nv_matrix_t *nv_patch_matrix_alloc_ex(const nv_matrix_t *src,
									  int patch_size, int grid_rows, int grid_cols);
nv_matrix_t *nv_patch_matrix_alloc(const nv_matrix_t *src, int patch_size);
void nv_patch_extract(nv_matrix_t *patches, const nv_matrix_t *src, int patch_size);

#ifdef __cplusplus
}
#endif

#endif

