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
#include "nv_core.h"
#include "nv_num.h"
#include "nv_ip_pooling.h"

void
nv_max_pooling(nv_matrix_t *output,
			   const nv_matrix_t *conv,
			   int pooling_size,
			   int stride)
{
	int y;

	NV_ASSERT(output->n == conv->n);
	NV_ASSERT(output->rows == conv->rows / stride);
	NV_ASSERT(output->cols == conv->cols / stride);
	
	nv_matrix_zero(output);
/*
  + + + + +
  + + + + +
  + + + + +
  + + + + +
  + + + + +  
 */


#ifdef _OPENMP
#pragma omp parallel for
#endif	
	for (y = 0; y < output->rows; ++y) {
		int x;
		for (x = 0; x < output->cols; ++x) {
			int h;
			for (h = 0; h < pooling_size; ++h) {
				int w;
				const int yh = y * stride + h;
				if (!(yh >= 0 && yh < conv->rows)) {
					continue;
				}
				for (w = 0; w < pooling_size; ++w) {
					const int xw = x * stride + w;
					if (!(xw >= 0 && xw < conv->cols)) {
						continue;
					}
					nv_vector_max(output, NV_MAT_M(output, y, x),
								  output, NV_MAT_M(output, y, x),
								  conv, NV_MAT_M(conv, yh, xw));
				}
			}
		}
	}
}
void
nv_average_pooling(nv_matrix_t *output,
				   const nv_matrix_t *conv,
				   int pooling_size,
				   int stride)
{
	int y;
	NV_ASSERT(output->n == conv->n);
	NV_ASSERT(output->rows == conv->rows / stride);
	NV_ASSERT(output->cols == conv->cols / stride);
	
	nv_matrix_zero(output);
#ifdef _OPENMP
#pragma omp parallel for
#endif	
	for (y = 0; y < output->rows; ++y) {
		int x;
		for (x = 0; x < output->cols; ++x) {
			int h;
			int count = 0;
			for (h = 0; h < pooling_size; ++h) {
				int w;
				const int yh = y * stride + h;
				if (!(yh >= 0 && yh < conv->rows)) {
					continue;
				}
				for (w = 0; w < pooling_size; ++w) {
					const int xw = x * stride + w;
					if (!(xw >= 0 && xw < conv->cols)) {
						continue;
					}
					nv_vector_add(output, NV_MAT_M(output, y, x),
								  output, NV_MAT_M(output, y, x),
								  conv, NV_MAT_M(conv, yh, xw));
					++count;
				}
			}
			if (count > 0) {
				nv_vector_muls(output, NV_MAT_M(output, y, x),
							   output, NV_MAT_M(output, y, x),
							   1.0f / (float)count);
			}
		}
	}
}
