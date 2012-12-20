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
#include "nv_io.h"
#include "nv_test.h"

void nv_test_serialize(void)
{
	nv_matrix_t *mat = nv_matrix_alloc(3, 11), *mat2;
	int i, j;
	char *s;

	NV_TEST_NAME;
	
	for (j = 0; j < mat->m; ++j) {
		for (i = 0; i < mat->n; ++i) {
			NV_MAT_V(mat, j, i) = nv_rand();
		}
	}
	s = nv_serialize_matrix(mat);
	
	NV_ASSERT(s != NULL);
	
	printf("%s\n", s);
	
	mat2 = nv_deserialize_matrix(s);
	NV_ASSERT(mat2 != NULL);
	
	for (j = 0; j < mat->m; ++j) {
		for (i = 0; i < mat->n; ++i) {
			NV_ASSERT(fabsf(NV_MAT_V(mat, j, i) - NV_MAT_V(mat2, j, i)) < FLT_EPSILON);
		}
	}

	nv_matrix_free(&mat);
	nv_matrix_free(&mat2);
	nv_free(s);

	fflush(stdout);
}
