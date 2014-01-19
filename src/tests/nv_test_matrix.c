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

#undef NDEBUG
#include "nv_core.h"
#include "nv_num.h"
#include "nv_test.h"


void
nv_test_matrix_mul(void)
{
	static const float src[3][3] = {
		{ 1.0f, 2.0f, 3.0f },
		{ 4.0f, 5.0f, 6.0f },
		{ 7.0f, 8.0f, 9.0f }
	};
	static const float src2[3] = { 1.0f, 2.0f, 3.0f };
	
	nv_matrix_t *a = nv_matrix_alloc(3, 3);
	nv_matrix_t *y = nv_matrix_alloc(3, 3);
	nv_matrix_t *d = nv_matrix_alloc(3, 1);
	
	int i, j;
	
	NV_TEST_NAME;
	
	for (j = 0; j < a->m; ++j) {
		for (i = 0; i < a->n; ++i) {
			NV_MAT_V(a, j, i) = src[j][i];
		}
	}
	for (i = 0; i < d->n; ++i) {
		NV_MAT_V(d, 0, i) = src2[i];
	}
	printf("A = \n");
	nv_matrix_print(stdout, a);
	
	printf("A * A = \n");
	nv_matrix_mul(y, a, NV_MAT_NOTR, a, NV_MAT_NOTR);
	nv_matrix_print(stdout, y);
	
	printf("A' * A' = \n");
	nv_matrix_mul(y, a, NV_MAT_TR, a, NV_MAT_TR);
	nv_matrix_print(stdout, y);
	
	printf("A * A' = \n");
	nv_matrix_mul(y, a, NV_MAT_NOTR, a, NV_MAT_TR);
	nv_matrix_print(stdout, y);
	
	printf("A' * A = \n");
	nv_matrix_mul(y, a, NV_MAT_TR, a, NV_MAT_NOTR);
	nv_matrix_print(stdout, y);

	nv_matrix_free(&a);
	nv_matrix_free(&y);
	nv_matrix_free(&d);	
}

void
nv_test_matrix(void)
{
	nv_test_matrix_mul();
}
