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

#ifndef NV_CORE_ALLOC_H
#define NV_CORE_ALLOC_H

#ifdef __cplusplus
extern "C" {
#endif

typedef void (*nv_outofmemory_t)(void);

nv_outofmemory_t nv_set_outofmemory_func(nv_outofmemory_t func);
nv_outofmemory_t nv_get_outofmemory_func(void);

void *nv_malloc(size_t s);
void *nv_realloc(void *mem, size_t s);
void nv_free(void *mem);

#define nv_alloc_type(type, n) (type *)nv_malloc(sizeof(type) * (n))

#ifdef __cplusplus
}
#endif

#endif
