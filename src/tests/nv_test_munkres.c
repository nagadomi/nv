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
nv_test_munkres(void)
{
	// 1
	nv_matrix_t *cost = nv_matrix_alloc(4, 4);
	nv_matrix_t *task = nv_matrix_alloc(4, 1);
	int j, i;
	float min_cost;

	NV_TEST_NAME;
	
	for (j = 0; j < cost->m; ++j) {
		for (i = 0; i < cost->n; ++i) {
			NV_MAT_V(cost, j, i) = (float)((j + 1) * (i + 1));
		}
	}
	nv_matrix_print(stdout, cost);

	min_cost = nv_munkres(task, cost);
	printf("task: ");
	for (i = 0; i < task->n; ++i) {
		printf("%d ", (int)NV_MAT_V(task, 0, i));
		NV_ASSERT((int)NV_MAT_V(task, 0, i) == 3 - i);
	}
	printf("\nmin_cost: %f\n", min_cost);
	NV_ASSERT(min_cost == 20.0f);
	
	nv_matrix_free(&cost);
	nv_matrix_free(&task);

	// 2
	cost = nv_matrix_alloc(3, 4);
	task = nv_matrix_alloc(3, 1);

	for (j = 0; j < cost->m; ++j) {
		for (i = 0; i < cost->n; ++i) {
			NV_MAT_V(cost, j, i) = (float)((j + 1) * (i + 1));
		}
	}
	nv_matrix_print(stdout, cost);
	min_cost = nv_munkres(task, cost);
	printf("task: ");
	for (i = 0; i < task->n; ++i) {
		printf("%d ", (int)NV_MAT_V(task, 0, i));
		NV_ASSERT((int)NV_MAT_V(task, 0, i) == 2 - i);
	}
	printf("\nmin_cost: %f\n", min_cost);
	NV_ASSERT(min_cost == 10.0f);
	nv_matrix_free(&cost);
	nv_matrix_free(&task);

	// 3
	cost = nv_matrix_alloc(5, 3);
	task = nv_matrix_alloc(5, 1);

	for (j = 0; j < cost->m; ++j) {
		for (i = 0; i < cost->n; ++i) {
			NV_MAT_V(cost, j, i) = (float)((4 - j) * (5 - i));
		}
	}
	nv_matrix_print(stdout, cost);
	min_cost = nv_munkres(task, cost);
	printf("task: ");
	for (i = 0; i < task->n; ++i) {
		printf("%d ", (int)NV_MAT_V(task, 0, i));
	}
	printf("\nmin_cost: %f\n", min_cost);
	NV_ASSERT(min_cost == 16.0f);
	nv_matrix_free(&cost);
	nv_matrix_free(&task);

	fflush(stdout);
}
