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
#include <map>

typedef struct nv_x 
{
	void *data;
} nv_x_t;

typedef struct nv_y 
{
	std::multimap<int, nv_x_t> x;
} nv_y_t;

typedef std::map<int, nv_y_t> nv_geo_t;

struct nv_2d_index {
	nv_geo_t geo;
};

nv_2d_index_t *
nv_2d_index_alloc(void)
{
	return new nv_2d_index_t;
}

void 
nv_2d_index_free(nv_2d_index_t **index)
{
	if (index && *index) {
		delete *index;
		*index = NULL;
	}
}

void
nv_2d_index_insert(nv_2d_index_t *index, int y, int x, void *data)
{
	nv_geo_t::iterator it = index->geo.find(y);
	if (it == index->geo.end()) {
		nv_y_t yv;
		nv_x_t xv;
		xv.data = data;
		yv.x.insert(std::make_pair(x, xv));
		index->geo.insert(std::make_pair(y, yv));
	} else {
		nv_x_t xv;
		xv.data = data;
		it->second.x.insert(std::make_pair(x, xv));
	}
}

int 
nv_2d_index_each(const nv_2d_index_t *index, nv_2d_index_func_t func, void *userdata)
{
	nv_geo_t::const_iterator syitr = index->geo.begin();
	nv_geo_t::const_iterator eyitr = index->geo.end();
	nv_geo_t::const_iterator itr;
	int ret = 0;

	for (itr = syitr; itr != eyitr; ++itr) {
		std::multimap<int, nv_x_t>::const_iterator sxitr = itr->second.x.begin();
		std::multimap<int, nv_x_t>::const_iterator exitr = itr->second.x.end();
		for (std::multimap<int, nv_x_t>::const_iterator jtr = sxitr; jtr != exitr; ++jtr) {
			ret = func(itr->first, jtr->first, jtr->second.data, userdata);
			if (ret != 0) {
				return ret;
			}
		}
	}

	return 0;
}

int 
nv_2d_index_rect_each(const nv_2d_index_t *index, nv_2d_index_func_t func, 
					  int sy, int sx, int h, int w, void *userdata)
{
	nv_geo_t::const_iterator syitr = index->geo.lower_bound(sy);
	nv_geo_t::const_iterator eyitr = index->geo.upper_bound(sy + h - 1);
	for (nv_geo_t::const_iterator itr = syitr; itr != eyitr; ++itr) {
		std::multimap<int, nv_x_t>::const_iterator sxitr = itr->second.x.lower_bound(sx);
		std::multimap<int, nv_x_t>::const_iterator exitr = itr->second.x.upper_bound(sx + w - 1);

		for (std::multimap<int, nv_x_t>::const_iterator jtr = sxitr; jtr != exitr; ++jtr) {
			int ret = func(itr->first, jtr->first, jtr->second.data, userdata);
			if (ret != 0) {
				return ret;
			}
		}
	}

	return 0;
}

typedef struct {
	float y, x, r;
	nv_2d_index_func_t func;
	void *userdata;
} nv_2d_circle_args_t;

static int
nv_2d_circle_wrapper(int y, int x, void *data, void *userdata)
{
	nv_2d_circle_args_t *args = (nv_2d_circle_args_t *)userdata;
	int ret = 0;
	float fx = (float)x;
	float fy = (float)y;

	if (sqrtf((fy - args->y) * (fy - args->y) + (fx - args->x) * (fx - args->x)) < args->r) {
		ret = args->func(y, x, data, args->userdata);
	}

	return ret;
}

int 
nv_2d_index_circle_each(const nv_2d_index_t *index, nv_2d_index_func_t func,
						int y, int x, int r, void *userdata)
{
	nv_2d_circle_args_t args;
	args.y = (float)y;
	args.x = (float)x;
	args.r = (float)r;
	args.userdata = userdata;
	args.func = func;

	return nv_2d_index_rect_each(index, 
		nv_2d_circle_wrapper, 
		y - r, x - r, r * 2, r * 2,
		&args);
}
