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

#include "nv_core.h"
#if NV_WITH_EIIO
#  include "eiio.h"
#endif

static nv_outofmemory_t nv_outofmemory_func = NULL;

nv_outofmemory_t
nv_set_outofmemory_func(nv_outofmemory_t func)
{
	nv_outofmemory_t old_func = nv_outofmemory_func;
	nv_outofmemory_func = func;
	
#if NV_WITH_EIIO	
	eiio_set_outofmemory_func(func);
#endif
	return old_func;
}

nv_outofmemory_t
nv_get_outofmemory_func(void)
{
	return nv_outofmemory_func;
}

static void *
nv_outofmemory(void)
{
	if (nv_outofmemory_func) {
		(*nv_outofmemory_func)();
	} else {
		fprintf(stderr, "nv_alloc: out of memory\n");
		abort();
	}
	
	return NULL;
}

void *
nv_malloc(size_t s)
{
	void *mem = malloc(s);
	
	if (mem == NULL) {
		return nv_outofmemory();
	}

	return mem;
}

void *
nv_realloc(void *mem, size_t s)
{
	void *newmem = realloc(mem, s);
	
	if (mem == NULL) {
		return nv_outofmemory();
	}
	
	return newmem;
}

void
nv_free(void *mem)
{
	free(mem);
}
