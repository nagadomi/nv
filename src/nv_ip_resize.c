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

#include "nv_ip.h"

static void 
nv_resize_biliner(nv_matrix_t *dest, const nv_matrix_t *src)
{
	const float col_scale = (float)src->cols / dest->cols;
	const float row_scale = (float)src->rows / dest->rows;
	int y, x;
	int *x1s, *x2s;
	float *x1ws, *x2ws;
	int threads = nv_omp_procs();
	
	NV_ASSERT(dest->n == src->n);
	
	if (dest->cols == src->cols && dest->rows == src->rows) {
		nv_matrix_copy_all(dest, src);
		return;
	}
	
	x1s = nv_alloc_type(int, dest->cols);
	x2s = nv_alloc_type(int, dest->cols);
	x1ws = nv_alloc_type(float, dest->cols);
	x2ws = nv_alloc_type(float, dest->cols);
	
	for (x = 0; x < dest->cols; ++x) {
		float scale_x = col_scale * x;
		x1s[x] = (int)NV_FLOOR(scale_x);
		x2s[x] = NV_MIN(x1s[x] + 1, src->cols - 1);
		x2ws[x] = scale_x - (float)x1s[x];
		x1ws[x] = 1.0f - x2ws[x];
	}
	
	if (dest->n == 3) {
#ifdef _OPENMP
#pragma omp parallel for private(x) num_threads(threads)
#endif
		for (y = 0; y < dest->rows; ++y) {
			float scale_y = row_scale * y;
			int y1 = (int)NV_FLOOR(scale_y);
			int y2 = NV_MIN(y1 + 1, src->rows - 1);
			float y2_w = scale_y - (float)y1;
			float y1_w = 1.0f - y2_w;
			
#if 0/* NV_ENABLE_SSE2 // nv_matrix_allocでnが3だとアライメントされてないかもしれないので注意 */
			const __m128 y1w = _mm_set1_ps(y1_w);
			const __m128 y2w = _mm_set1_ps(y2_w);
			for (x = 0; x < dest->cols; ++x) {
				NV_ALIGNED(float, mm[4], 16);
				__m128 x1y1 = _mm_loadu_ps(&NV_MAT3D_V(src, y1, x1s[x], 0));
				__m128 x2y1 = _mm_loadu_ps(&NV_MAT3D_V(src, y1, x2s[x], 0));
				__m128 x1y2 = _mm_loadu_ps(&NV_MAT3D_V(src, y2, x1s[x], 0));
				__m128 x2y2 = _mm_loadu_ps(&NV_MAT3D_V(src, y2, x2s[x], 0));
				const __m128 x1w = _mm_set1_ps(x1ws[x]);
				const __m128 x2w = _mm_set1_ps(x2ws[x]);

				x1y1 = _mm_mul_ps(x1w, x1y1);
				x2y1 = _mm_mul_ps(x2w, x2y1);
				x1y1 = _mm_add_ps(x1y1, x2y1);
				
				x1y2 = _mm_mul_ps(x1w, x1y2);
				x2y2 = _mm_mul_ps(x2w, x2y2);
				x1y2 = _mm_add_ps(x1y2, x2y2);

				x1y1 = _mm_mul_ps(x1y1, y1w);
				x1y2 = _mm_mul_ps(x1y2, y2w);
				x1y1 = _mm_add_ps(x1y1, x1y2);
				
				_mm_store_ps(mm, x1y1);
				memmove(&NV_MAT3D_V(dest, y, x, 0), mm, sizeof(float) * 3);
			}
#else
			for (x = 0; x < dest->cols; ++x) {
				NV_MAT3D_V(dest, y, x, 0) = 
					y1_w * (x1ws[x] * NV_MAT3D_V(src, y1, x1s[x], 0)
							+ x2ws[x] * NV_MAT3D_V(src, y1, x2s[x], 0))
					+ y2_w * (x1ws[x] * NV_MAT3D_V(src, y2, x1s[x], 0)
							  + x2ws[x] * NV_MAT3D_V(src, y2, x2s[x], 0));
				NV_MAT3D_V(dest, y, x, 1) = 
					y1_w * (x1ws[x] * NV_MAT3D_V(src, y1, x1s[x], 1)
							+ x2ws[x] * NV_MAT3D_V(src, y1, x2s[x], 1))
					+ y2_w * (x1ws[x] * NV_MAT3D_V(src, y2, x1s[x], 1)
							  + x2ws[x] * NV_MAT3D_V(src, y2, x2s[x], 1));
				NV_MAT3D_V(dest, y, x, 2) = 
					y1_w * (x1ws[x] * NV_MAT3D_V(src, y1, x1s[x], 2)
							+ x2ws[x] * NV_MAT3D_V(src, y1, x2s[x], 2))
					+ y2_w * (x1ws[x] * NV_MAT3D_V(src, y2, x1s[x], 2)
							  + x2ws[x] * NV_MAT3D_V(src, y2, x2s[x], 2));
			}
#endif			
		}
	} else if (dest->n == 1) {
#ifdef _OPENMP
#pragma omp parallel for private(x) num_threads(threads)
#endif
		for (y = 0; y < dest->rows; ++y) {
			float scale_y = row_scale * y;
			int y1 = (int)NV_FLOOR(scale_y);
			int y2 = NV_MIN(y1 + 1, src->rows - 1);
			float y2_w = scale_y - (float)y1;
			float y1_w = 1.0f - y2_w;
			
			for (x = 0; x < dest->cols; ++x) {
				NV_MAT3D_V(dest, y, x, 0) = 
					y1_w * (x1ws[x] * NV_MAT3D_V(src, y1, x1s[x], 0)
							+ x2ws[x] * NV_MAT3D_V(src, y1, x2s[x], 0))
					+ y2_w * (x1ws[x] * NV_MAT3D_V(src, y2, x1s[x], 0)
							  + x2ws[x] * NV_MAT3D_V(src, y2, x2s[x], 0));
			}
		}
	} else {
#ifdef _OPENMP
#pragma omp parallel for private(x) num_threads(threads)
#endif
		for (y = 0; y < dest->rows; ++y) {
			float scale_y = row_scale * y;
			int y1 = (int)NV_FLOOR(scale_y);
			int y2 = NV_MIN(y1 + 1, src->rows - 1);
			float y2_w = scale_y - (float)y1;
			float y1_w = 1.0f - y2_w;
			
			for (x = 0; x < dest->cols; ++x) {
				int i;
				for (i = 0; i < dest->n; ++i) {
					NV_MAT3D_V(dest, y, x, i) = 
						y1_w * (x1ws[x] * NV_MAT3D_V(src, y1, x1s[x], i)
								+ x2ws[x] * NV_MAT3D_V(src, y1, x2s[x], i))
						+ y2_w * (x1ws[x] * NV_MAT3D_V(src, y2, x1s[x], i)
								  + x2ws[x] * NV_MAT3D_V(src, y2, x2s[x], i));
				}
			}
		}
	}
	
	nv_free(x1s);
	nv_free(x2s);
	nv_free(x1ws);
	nv_free(x2ws);
}

void 
nv_resize(nv_matrix_t *dest, const nv_matrix_t *src)
{
	nv_resize_biliner(dest, src);
}

nv_matrix_t *
nv_crop(const nv_matrix_t *src, nv_rect_t rect)
{
	nv_matrix_t *roi = nv_matrix3d_alloc(src->n, rect.height, rect.width);
	int y, copy_len;
	
	nv_matrix_zero(roi);
	
	if (src->cols <= rect.x) {
		return roi;
	}
	
	if (rect.x + rect.width < src->cols) {
		copy_len = rect.width * src->step * sizeof(float);
	} else {
		copy_len = (rect.width - ((rect.x + rect.width) - src->cols))
			* src->step * sizeof(float);
	}
	for (y = 0; y < rect.height && y + rect.y < src->rows; ++y) {
		memmove(&NV_MAT3D_V(roi, y, 0, 0),
			   &NV_MAT3D_V(src, rect.y + y, rect.x, 0),
			   copy_len);
	}
	return roi;
}

void
nv_crop_resize(nv_matrix_t *dest, const nv_matrix_t *src, nv_rect_t rect)
{
	nv_matrix_t *roi = nv_crop(src, rect);
	nv_resize(dest, roi);
}
