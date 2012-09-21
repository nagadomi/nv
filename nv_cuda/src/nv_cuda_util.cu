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

#include <cutil_inline.h>
//#include <cutil.h>
//#include <cublas.h>
#include "nv_core.h"
#include "nv_cuda_util.h"

static int nv_cuda_is_available = 0;
static int nv_cuda_sm_count = 0;
static int nv_cuda_thread_max = 0;

int 
nv_cuda_init(void)
{
#if __DEVICE_EMULATION__
	nv_cuda_is_available = 1;
	nv_cuda_sm_count = 16;
	nv_cuda_thread_max = NV_CUDA_THREAD_MAX;
	return 0;
#else
	int count = 0;
	int i = 0;
	cudaDeviceProp prop;
    
	if (cudaGetDeviceCount(&count) != cudaSuccess) {
		return -1;
	}
	if(count == 0) {
		return -1;
	}
	
	for(i = 0; i < count; i++) {
		if(cudaGetDeviceProperties(&prop, i) == cudaSuccess) {
			if(prop.major >= 1) {
				nv_cuda_sm_count = prop.multiProcessorCount;
				nv_cuda_thread_max = prop.maxThreadsPerBlock;

				// max
				if (nv_cuda_thread_max > NV_CUDA_THREAD_MAX) {
					nv_cuda_thread_max = NV_CUDA_THREAD_MAX;
				}

				break;
			}
		}
	}
	if(i == count) {
		return -1;
	}
	if (cudaSetDevice(i) != cudaSuccess) {
		return -1;
	}

	nv_cuda_is_available = 1;

	return 0;
#endif
}


int 
nv_cuda_available(void)
{
	return nv_cuda_is_available;
}


int 
nv_cuda_block(int n)
{
	if (n < nv_cuda_sm_count) {
		return 1;
	}
	if (n < nv_cuda_sm_count * 32) {
		return n / 32 + (n % 32 != 0 ? 1:0);
	}
	if (n < nv_cuda_sm_count * nv_cuda_thread_max) {
		return nv_cuda_sm_count;
	}
	return n / nv_cuda_thread_max + (n % nv_cuda_thread_max != 0 ? 1:0);
}

int 
nv_cuda_thread(int n)
{
	if (n < nv_cuda_sm_count) {
		return n;
	}
	if (n < nv_cuda_sm_count * 32) {
		return 32;
	}

	if (n < nv_cuda_sm_count * nv_cuda_thread_max) {
		return n / nv_cuda_sm_count 
			+ (32 - (nv_cuda_sm_count >= 32 ? 0:n % nv_cuda_sm_count));
	}
	return nv_cuda_thread_max;
}

int 
nv_cuda_optz_block()
{
	return nv_cuda_sm_count;
}

int 
nv_cuda_optz_thread()
{
	return nv_cuda_thread_max > 32 ? 32: nv_cuda_thread_max;
}

nv_matrix_t *
nv_cuda_matrix_dup(const nv_matrix_t *mat)
{
	nv_matrix_t *dev_mat = NULL;
	nv_matrix_t *dup = nv_matrix_clone(mat);

	CUDA_SAFE_CALL(cudaMalloc((void **)&dev_mat, sizeof(nv_matrix_t)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&dup->v, (size_t)(mat->list * mat->list_step * sizeof(float))));
	CUDA_SAFE_CALL(cudaMemcpy(dup->v, mat->v, (size_t)(mat->list * mat->list_step * sizeof(float)), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(dev_mat, dup, sizeof(nv_matrix_t), cudaMemcpyHostToDevice));
	nv_matrix_free(&dup);

	return dev_mat;
}

nv_matrix_t *
nv_cuda_matrix_clone(const nv_matrix_t *mat)
{
	return nv_cuda_matrix3d_alloc(mat->n, mat->rows, mat->cols);
}


nv_matrix_t *
nv_cuda_matrix_alloc(int n, int m)
{
	nv_matrix_t *mat = nv_matrix_alloc(n, m);
	nv_matrix_t *dev_mat;

	CUDA_SAFE_CALL(cudaMalloc((void **)&dev_mat, sizeof(nv_matrix_t)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&mat->v, (size_t)(mat->list * mat->list_step * sizeof(float))));
	CUDA_SAFE_CALL(cudaMemcpy(dev_mat, mat, sizeof(nv_matrix_t), cudaMemcpyHostToDevice));

	nv_matrix_free(&mat);

	return dev_mat;
}

nv_matrix_t *
nv_cuda_matrix_clone_zero(const nv_matrix_t *mat)
{
	return nv_cuda_matrix3d_alloc_zero(mat->n, mat->rows, mat->cols);
}

nv_matrix_t *
nv_cuda_matrix_alloc_zero(int n, int m)
{
	nv_matrix_t *mat = nv_matrix_alloc(n, m);
	nv_matrix_t *dev_mat;

	CUDA_SAFE_CALL(cudaMalloc((void **)&dev_mat, sizeof(nv_matrix_t)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&mat->v, (size_t)(mat->list * mat->list_step * sizeof(float))));
	CUDA_SAFE_CALL(cudaMemset(mat->v, 0, (size_t)(mat->list * mat->list_step * sizeof(float))));
	CUDA_SAFE_CALL(cudaMemcpy(dev_mat, mat, sizeof(nv_matrix_t), cudaMemcpyHostToDevice));

	nv_matrix_free(&mat);

	return dev_mat;
}

void
nv_cuda_matrix_zero(nv_matrix_t *mat)
{
	nv_matrix_t host_mat;

	CUDA_SAFE_CALL(cudaMemcpy(&host_mat, mat, sizeof(nv_matrix_t), cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaMemset(host_mat.v, 0, (size_t)(host_mat.list * host_mat.list_step * sizeof(float))));
}

nv_matrix_t *
nv_cuda_matrix3d_alloc_zero(int n, int rows, int cols)
{
	nv_matrix_t *mat = nv_matrix3d_alloc(n, rows, cols);
	nv_matrix_t *dev_mat;

	CUDA_SAFE_CALL(cudaMalloc((void **)&dev_mat, sizeof(nv_matrix_t)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&mat->v, (size_t)(mat->list * mat->list_step * sizeof(float))));
	CUDA_SAFE_CALL(cudaMemset(mat->v, 0, (size_t)(mat->list * mat->list_step * sizeof(float))));
	CUDA_SAFE_CALL(cudaMemcpy(dev_mat, mat, sizeof(nv_matrix_t), cudaMemcpyHostToDevice));

	nv_matrix_free(&mat);

	return dev_mat;
}

nv_matrix_t *
nv_cuda_matrix3d_alloc(int n, int rows, int cols)
{
	nv_matrix_t *mat = nv_matrix3d_alloc(n, rows, cols);
	nv_matrix_t *dev_mat;

	CUDA_SAFE_CALL(cudaMalloc((void **)&dev_mat, sizeof(nv_matrix_t)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&mat->v, (size_t)(mat->list * mat->list_step * sizeof(float))));
	CUDA_SAFE_CALL(cudaMemcpy(dev_mat, mat, sizeof(nv_matrix_t), cudaMemcpyHostToDevice));

	nv_matrix_free(&mat);

	return dev_mat;
}

void 
nv_cuda_matrix_free(nv_matrix_t *dev_mat)
{
	nv_matrix_t mat;

	CUDA_SAFE_CALL(cudaMemcpy(&mat, dev_mat, sizeof(nv_matrix_t), cudaMemcpyDeviceToHost));
	cudaFree(mat.v);
	cudaFree(dev_mat);
}
