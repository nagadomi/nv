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

static void
nv_test_io_mat(void)
{
	nv_matrix_t *mat = nv_matrix_alloc(72, 10000);
	nv_matrix_t *mat2, *mat3;
	int i, j;
	long t;

	nv_matrix_zero(mat);
	
	NV_TEST_NAME;
	
	for (j = 0; j < mat->m; ++j) {
		for (i = 0; i < mat->n; ++i) {
			NV_MAT_V(mat, j, i) = nv_rand() * nv_rand_index(10000);
		}
	}
	
	t = nv_clock();
	nv_save_matrix_text("test.mat", mat);
	printf("save text %ldms\n", nv_clock() - t);
	t = nv_clock();
	mat2 = nv_load_matrix_text("test.mat");
	printf("load text %ldms\n", nv_clock() - t);

	t = nv_clock();
	nv_save_matrix_bin("test.matb", mat);
	printf("save bin %ldms\n", nv_clock() - t);
	t = nv_clock();
	mat3 = nv_load_matrix_bin("test.matb");
	printf("load bin %ldms\n", nv_clock() - t);
	
	for (j = 0; j < mat->m; ++j) {
		for (i = 0; i < mat->n; ++i) {
			//printf("raw-text: %f == %f\n", NV_MAT_V(mat, j, i), NV_MAT_V(mat2, j, i));
			//printf("raw-bin:  %E == %E\n", NV_MAT_V(mat, j, i), NV_MAT_V(mat3, j, i));
			NV_ASSERT(NV_TEST_EQ(NV_MAT_V(mat, j, i), NV_MAT_V(mat2, j, i)));
			NV_ASSERT(NV_TEST_EQ(NV_MAT_V(mat, j, i), NV_MAT_V(mat3, j, i)));
		}
	}
	nv_matrix_free(&mat);
	nv_matrix_free(&mat2);
	nv_matrix_free(&mat3);

	fflush(stdout);
}

static void
nv_test_io_mat_array(void)
{
	nv_matrix_t *array[3];
	nv_matrix_t *ret1[3], *ret2[3];
	nv_matrix_t *mat = nv_matrix_alloc(72, 1024);
	int i, j, k, n;
	
	nv_matrix_zero(mat);
	
	NV_TEST_NAME;
	
	for (j = 0; j < mat->m; ++j) {
		for (i = 0; i < mat->n; ++i) {
			NV_MAT_V(mat, j, i) = nv_rand() * nv_rand_index(10000);
		}
	}
	
	array[0] = mat;
	array[1] = mat;
	array[2] = mat;
	NV_ASSERT(nv_save_matrix_array_text("test.mat", array, 3) == 0);
	n = 3;
	NV_ASSERT(nv_load_matrix_array_text("test.mat", ret1, &n) == 0);
	NV_ASSERT(n == 3);
	
	NV_ASSERT(nv_save_matrix_array_bin("test.matb", array, 3) == 0);
	n = 3;
	NV_ASSERT(nv_load_matrix_array_bin("test.matb", ret2, &n) == 0);
	NV_ASSERT(n == 3);
	
	for (k = 0; k < 3; ++k) {
		for (j = 0; j < mat->m; ++j) {
			for (i = 0; i < mat->n; ++i) {
				NV_ASSERT(NV_TEST_EQ(NV_MAT_V(mat, j, i), NV_MAT_V(ret1[k], j, i)));
				NV_ASSERT(NV_TEST_EQ(NV_MAT_V(mat, j, i), NV_MAT_V(ret2[k], j, i)));
			}
		}
	}
	for (i = 0; i < 3; ++i) {
		nv_matrix_free(&ret1[i]);
		nv_matrix_free(&ret2[i]);
	}
	
	array[0] = mat;
	array[1] = mat;
	NV_ASSERT(nv_save_matrix_array_text("test.mat", array, 2) == 0);
	n = 2;
	NV_ASSERT(nv_load_matrix_array_text("test.mat", ret1, &n) == 0);
	NV_ASSERT(n == 2);
	
	NV_ASSERT(nv_save_matrix_array_bin("test.matb", array, 2) == 0);
	n = 2;
	NV_ASSERT(nv_load_matrix_array_bin("test.matb", ret2, &n) == 0);
	NV_ASSERT(n == 2);

	for (k = 0; k < 2; ++k) {
		for (j = 0; j < mat->m; ++j) {
			for (i = 0; i < mat->n; ++i) {
				NV_ASSERT(NV_TEST_EQ(NV_MAT_V(mat, j, i), NV_MAT_V(ret1[k], j, i)));
				NV_ASSERT(NV_TEST_EQ(NV_MAT_V(mat, j, i), NV_MAT_V(ret2[k], j, i)));
			}
		}
	}
	for (i = 0; i < 2; ++i) {
		nv_matrix_free(&ret1[i]);
		nv_matrix_free(&ret2[i]);
	}
	
	n = 4;
	NV_ASSERT(nv_load_matrix_array_text("test.mat", ret1, &n) == 0);
	NV_ASSERT(n == 2);
	n = 4;
	NV_ASSERT(nv_load_matrix_array_bin("test.matb", ret2, &n) == 0);
	NV_ASSERT(n == 2);
	for (k = 0; k < 2; ++k) {
		for (j = 0; j < mat->m; ++j) {
			for (i = 0; i < mat->n; ++i) {
				NV_ASSERT(NV_TEST_EQ(NV_MAT_V(mat, j, i), NV_MAT_V(ret1[k], j, i)));
				NV_ASSERT(NV_TEST_EQ(NV_MAT_V(mat, j, i), NV_MAT_V(ret2[k], j, i)));
			}
		}
	}
	for (i = 0; i < 2; ++i) {
		nv_matrix_free(&ret1[i]);
		nv_matrix_free(&ret2[i]);
	}
	
	nv_matrix_free(&mat);
	fflush(stdout);
}

void
nv_test_io(void)
{
	nv_test_io_mat();
	nv_test_io_mat_array();
}
