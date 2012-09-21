/*
 * This file is part of libnv.
 *
 * Copyright (C) 2012 nagadomi@nurs.or.jp
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

#ifndef NV_CUDA_UTIL_H
#define NV_CUDA_UTIL_H
#ifdef __cplusplus
extern "C" {
#endif

#define NV_CUDA_THRESHOLD 256
#define NV_CUDA_THREAD_MAX 192
#define NV_CUDA_PIN_MALLOC 0 // use cudaMallocHost

int nv_cuda_init(void);
int nv_cuda_available(void);
int nv_cuda_block(int n);
int nv_cuda_thread(int n);

int nv_cuda_optz_block();
int nv_gpu_optz_thread();

nv_matrix_t *nv_cuda_matrix_dup(const nv_matrix_t *hostmat);
nv_matrix_t *nv_cuda_matrix_clone(const nv_matrix_t *hostmat);
nv_matrix_t *nv_cuda_matrix_alloc(int n, int m);
nv_matrix_t *nv_cuda_matrix3d_alloc(int n, int rows, int cols);

nv_matrix_t *nv_cuda_matrix_clone_zero(const nv_matrix_t *hostmat);
nv_matrix_t *nv_cuda_matrix_alloc_zero(int n, int m);
nv_matrix_t *nv_cuda_matrix3d_alloc_zero(int n, int rows, int cols);

void nv_cuda_matrix_zero(nv_matrix_t *mat);
void nv_cuda_matrix_free(nv_matrix_t *devmat);

#ifdef __cplusplus
}
#endif

#endif



