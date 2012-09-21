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

#ifndef NV_UTIL_2D_INDEX_H
#define NV_UTIL_2D_INDEX_H
#ifdef __cplusplus
extern "C" {
#endif

typedef struct nv_2d_index nv_2d_index_t;

nv_2d_index_t *nv_2d_index_alloc(void);

typedef int (*nv_2d_index_func_t)(int y, int x, void *data, void *userdata);

void nv_2d_index_insert(nv_2d_index_t *index, int y, int x, void *data);

int nv_2d_index_each(const nv_2d_index_t *index, nv_2d_index_func_t func, void *userdata);
int nv_2d_index_rect_each(const nv_2d_index_t *index, nv_2d_index_func_t func, 
						  int sy, int sx, int h, int w, void *userdata);
int nv_2d_index_circle_each(const nv_2d_index_t *index, nv_2d_index_func_t func,
							int y, int x, int r, void *userdata);
void nv_2d_index_free(nv_2d_index_t **index);
#ifdef __cplusplus
}
#endif


#endif
