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
#include "nv_ip.h"
#include "nv_num.h"

typedef struct {
	int x;
	int y;
} nv_ccv_point_t;

static NV_INLINE void
nv_ccv_fill(nv_matrix_t *labels,
			nv_matrix_t *id_image,
			int y, int x, float label
	)
{
	float p = NV_MAT3D_V(id_image, y, x, 0);
	nv_ccv_point_t *stack = nv_alloc_type(nv_ccv_point_t, id_image->rows * id_image->cols);
	int index = 0;
	
	stack[index].y = y;
	stack[index].x = x;
	while (index >= 0) {
		y = stack[index].y;
		x = stack[index].x;

		assert(id_image->rows * id_image->cols > index);

		if (y > 0) {
			if (NV_MAT3D_V(labels, y - 1, x, 0) < 0.0f 
				&& p == NV_MAT3D_V(id_image, y - 1, x, 0)) 
			{
				NV_MAT3D_V(labels, y - 1, x, 0) = label;
				stack[index].y = y - 1;
				stack[index].x = x;
				++index;
			}
		}
		if (x > 0) {
			if (y > 0) {
				if (NV_MAT3D_V(labels, y - 1, x - 1, 0) < 0.0f 
					&& p == NV_MAT3D_V(id_image, y - 1, x - 1, 0)) 
				{
					NV_MAT3D_V(labels, y - 1, x - 1, 0) = label;
					stack[index].y = y - 1;
					stack[index].x = x - 1;
					++index;
				}
				if (NV_MAT3D_V(labels, y, x - 1, 0) < 0.0f 
					&& p == NV_MAT3D_V(id_image, y, x - 1, 0)) 
				{
					NV_MAT3D_V(labels, y, x - 1, 0) = label;
					stack[index].y = y;
					stack[index].x = x - 1;
					++index;
				}
				if (y < id_image->rows - 1) {
					if (NV_MAT3D_V(labels, y + 1, x - 1, 0) < 0.0f 
						&& p == NV_MAT3D_V(id_image, y + 1, x - 1, 0)) 
					{
						NV_MAT3D_V(labels, y + 1, x - 1, 0) = label;
						stack[index].y = y + 1;
						stack[index].x = x - 1;
						++index;
					}
				}
			}
			if (y < id_image->rows - 1) {
				if (NV_MAT3D_V(labels, y + 1, x, 0) < 0.0f 
					&& p == NV_MAT3D_V(id_image, y + 1, x, 0)) 
				{
					NV_MAT3D_V(labels, y + 1, x, 0) = label;
					stack[index].y = y + 1;
					stack[index].x = x;
					++index;
				}
			}
			if (x < id_image->cols - 1) {
				if (y > 0) {
					if (NV_MAT3D_V(labels, y - 1, x + 1, 0) < 0.0f 
						&& p == NV_MAT3D_V(id_image, y - 1, x + 1, 0)) 
					{
						NV_MAT3D_V(labels, y - 1, x + 1, 0) = label;
						stack[index].y = y - 1;
						stack[index].x = x + 1;
						++index;
					}
				}
				if (NV_MAT3D_V(labels, y, x + 1, 0) < 0.0f 
					&& p == NV_MAT3D_V(id_image, y, x + 1, 0)) 
				{
					NV_MAT3D_V(labels, y, x + 1, 0) = label;
					stack[index].y = y;
					stack[index].x = x + 1;
					++index;
				}
				if (y < id_image->rows - 1) {
					if (NV_MAT3D_V(labels, y + 1, x + 1, 0) < 0.0f 
						&& p == NV_MAT3D_V(id_image, y + 1, x + 1, 0)) 
					{
						NV_MAT3D_V(labels, y + 1, x + 1, 0) = label;
						stack[index].y = y + 1;
						stack[index].x = x + 1;
						++index;
					}
				}
			}
		}
		--index;
	}
	nv_free(stack);
}


void nv_ccv_nn(nv_matrix_t *ccv,
			   int ccv_j,
			   const nv_matrix_t *image,
			   const nv_matrix_t *centroids,
			   int threshold
	)
{
	nv_matrix_t *id_image = nv_matrix3d_alloc(1, image->rows, image->cols);	
	nv_matrix_t *labels = nv_matrix3d_alloc(1, image->rows, image->cols);
	int *count, *color_index;
	int j, y, x;
	int label = 0;
	int threads = nv_omp_procs();

	assert(ccv->n == NV_CCV_DIM(centroids->m));

#ifdef _OPENMP
#pragma omp parallel for num_threads(threads)
#endif
	for (j = 0; j < image->m; ++j) {
		NV_MAT_V(id_image, j, 0) = (float)nv_nn(centroids, image, j);
	}

	nv_matrix_fill(labels, -1.0f);
	for (y = 0; y < id_image->rows; ++y) {
		for (x = 0; x < id_image->cols; ++x) {
			if (NV_MAT3D_V(labels, y, x, 0) < 0.0f) {
				NV_MAT3D_V(labels, y, x, 0) = (float)label;
				nv_ccv_fill(labels, id_image, y, x, (float)label);
				++label;
			}
		}
	}

	count = nv_alloc_type(int, label);
	color_index = nv_alloc_type(int, label);
	
	memset(count, 0, sizeof(int) * label);
	memset(color_index, 0xff, sizeof(int) * label);
	for (j = 0; j < labels->m; ++j) {
		int l = (int)NV_MAT_V(labels, j, 0);
		++count[l];
		color_index[l] = j;
	}
	
	nv_vector_zero(ccv, ccv_j);
	for (j = 0; j < label; ++j) {
		if (color_index[j] >= 0) {
			int c = (int)NV_MAT_V(id_image, color_index[j], 0);
			if (count[j] > threshold) {
				NV_MAT_V(ccv, ccv_j, centroids->m + c) += (float)count[j];
			} else {
				NV_MAT_V(ccv, ccv_j, c) += (float)count[j];
			}
		}
	}

	nv_free(color_index);
	nv_free(count);
	nv_matrix_free(&labels);
	nv_matrix_free(&id_image);
}
