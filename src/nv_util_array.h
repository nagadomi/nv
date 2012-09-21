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

#ifndef NV_UTIL_ARRAY_H
#define NV_UTIL_ARRAY_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct nv_array nv_array_t;

nv_array_t *nv_array_alloc(int n);
void nv_array_free(nv_array_t **ary);
void nv_array_set(nv_array_t *ary, int index, void *data);
void *nv_array_get(const nv_array_t *ary, int index);
void nv_array_push(nv_array_t *ary, void *data);
int nv_array_count(const nv_array_t *ary);

typedef int (*nv_array_foreach_func_t)(void *data, void *userdata);
int nv_array_foreach(nv_array_t *ary, nv_array_foreach_func_t func, void *userdata);

#ifdef __cplusplus
}
#endif

#endif
