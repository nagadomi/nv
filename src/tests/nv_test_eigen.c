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

#undef NDEBUG
#include "nv_core.h"
#include "nv_num.h"
#include "nv_test.h"

void
nv_test_eigen(void)
{
	nv_matrix_t *a = nv_matrix_alloc(3, 3);
	nv_matrix_t *vec = nv_matrix_alloc(3, 3);
	nv_matrix_t *val = nv_matrix_alloc(1, 3);

	nv_matrix_zero(vec);
	nv_matrix_zero(val);
	NV_MAT_V(a, 0, 0) = 1.0f;
	NV_MAT_V(a, 0, 1) = 0.5f;
	NV_MAT_V(a, 0, 2) = 0.3f;
	NV_MAT_V(a, 1, 0) = 0.5f;
	NV_MAT_V(a, 1, 1) = 1.0f;
	NV_MAT_V(a, 1, 2) = 0.6f;
	NV_MAT_V(a, 2, 0) = 0.3f;
	NV_MAT_V(a, 2, 1) = 0.6f;
	NV_MAT_V(a, 2, 2) = 1.0f;
	
	NV_TEST_NAME;

	nv_eigen(vec, val, a, 3, 50);
	
	nv_matrix_print(stdout, val);
	nv_matrix_print(stdout, vec);
	
	NV_ASSERT((fabsf(NV_MAT_V(val, 0, 0)) - fabsf(1.944f)) < 0.001f);
	NV_ASSERT((fabsf(NV_MAT_V(val, 1, 0)) - fabsf(0.707f)) < 0.001f);
	NV_ASSERT((fabsf(NV_MAT_V(val, 2, 0)) - fabsf(0.349f)) < 0.001f);
	
	nv_matrix_free(&val);
	nv_matrix_free(&vec);
	nv_matrix_free(&a);
}

