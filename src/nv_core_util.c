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

#include <time.h>
#include "nv_core.h"
#include "nv_core_matrix.h"
#include "nv_core_util.h"

static const uint32_t bits_in_8bit[256] = {
	0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4,
	1,2,2,3,2,3,3,4,2,3,3,4,3,4,4,5,
	1,2,2,3,2,3,3,4,2,3,3,4,3,4,4,5,
	2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,
	1,2,2,3,2,3,3,4,2,3,3,4,3,4,4,5,
	2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,
	2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,
	3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,
	1,2,2,3,2,3,3,4,2,3,3,4,3,4,4,5,
	2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,
	2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,
	3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,
	2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,
	3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,
	3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,
	4,5,5,6,5,6,6,7,5,6,6,7,6,7,7,8
};

uint32_t
nv_popcnt_u32(uint32_t x)
{
	uint32_t n = 0;
	
	n = 
		bits_in_8bit[x & 0xff]
	+   bits_in_8bit[(x >> 8) & 0xff]
	+   bits_in_8bit[(x >> 16) & 0xff]
	+   bits_in_8bit[(x >> 24) & 0xff];

	return n;
}

uint64_t
nv_popcnt_u64(uint64_t x)
{
	uint64_t n = 0;

	n = 
		bits_in_8bit[x & 0xff]
	+   bits_in_8bit[(x >> 8) & 0xff]
	+   bits_in_8bit[(x >> 16) & 0xff]
	+   bits_in_8bit[(x >> 24) & 0xff]
	+   bits_in_8bit[(x >> 32) & 0xff]
	+   bits_in_8bit[(x >> 40) & 0xff]
	+   bits_in_8bit[(x >> 48) & 0xff]
	+   bits_in_8bit[(x >> 56) & 0xff];

	return n;
}

float 
nv_sign(float x)
{
	return (x >= 0.0f) ? 1.0f:-1.0f;
}
static int g_omp_procs = 0;

int 
nv_omp_thread_id(void)
{
#ifdef _OPENMP
	return omp_get_thread_num();
#else
	return 0;
#endif
}

void
nv_omp_set_procs(int n)
{
	if (n >= 0) {
		g_omp_procs = n;
#ifdef _OPENMP		
		omp_set_num_threads(g_omp_procs);
#endif		
	}
}

int 
nv_omp_procs(void)
{
#ifdef _OPENMP
	if (g_omp_procs) {
		return g_omp_procs;
	} else {
		int procs = 0;
		const char *env_limit = nv_getenv("OMP_THREAD_LIMIT");
		if (env_limit != NULL) {
			procs = atoi(env_limit);
			if (procs <= 0) {
				const char *env_procs = nv_getenv("OMP_NUM_THREADS");
				if (env_procs != NULL) {
					procs = atoi(env_procs);
					if (procs <= 0) {
						procs = omp_get_num_procs();
					}
				} else {
					procs = omp_get_num_procs();
				}
			}
		} else {
			const char *env_procs = nv_getenv("OMP_NUM_THREADS");
			if (env_procs != NULL) {
				procs = atoi(env_procs);
				if (procs <= 0) {
					procs = omp_get_num_procs();
				}
			} else {
				procs = omp_get_num_procs();
			}
		}
		nv_omp_set_procs(procs);
		
		return g_omp_procs;
	}
#else
	return 1;
#endif
}

float nv_log2(float v)
{
	return logf(v)/logf(2.0f);
}

