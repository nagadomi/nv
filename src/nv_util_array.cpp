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

#include <vector>
#include "nv_core.h"

struct nv_array {
	std::vector<void *> vec;
};


nv_array_t *
nv_array_alloc(int n)
{
	nv_array_t *ary = new nv_array_t;
	ary->vec.reserve(n);
	return ary;
}

void 
nv_array_free(nv_array_t **ary)
{
	if (ary && *ary) {
		delete *ary;
		*ary = NULL;
	}
}

void 
nv_array_set(nv_array_t *ary, int index, void *data)
{
	ary->vec[index] = data;
}

void *
nv_array_get(const nv_array_t *ary, int index)
{
	return ary->vec[index];
}

void 
nv_array_push(nv_array_t *ary, void *data)
{
	ary->vec.push_back(data);
}

int
nv_array_count(const nv_array_t *ary)
{
	return ary->vec.size();
}

int 
nv_array_foreach(const nv_array_t *ary, nv_array_foreach_func_t func, void *userdata)
{
	std::vector<void *>::const_iterator it;
	int ret;
	for (it = ary->vec.begin(); it != ary->vec.end(); it++) {
		if ((ret = func(*it, userdata)) != 0) {
			return ret;
		}
	}
	return 0;
}
