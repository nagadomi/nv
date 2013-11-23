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
#include "nv_num.h"
#include <vector>
#include <queue>

int 
nv_nn_ex(const nv_matrix_t *mat,
		 const nv_matrix_t *vec, int vec_j,
		 nv_knn_func_t func)
{
	int i, mi;
	int threads = nv_omp_procs();
	int *min_index = nv_alloc_type(int, threads);
	float *min_dist = nv_alloc_type(float, threads);

	for (i = 0; i < threads; ++i) {
		min_index[i] = -1;
		min_dist[i] = FLT_MAX;
	}
	
	NV_ASSERT(mat->n == vec->n);
	
#ifdef _OPENMP
#pragma omp parallel for num_threads(threads) 
#endif	
	for (i = 0; i < mat->m; ++i) {
		int thread_idx = nv_omp_thread_id();
		float dist = (*func)(mat, i, vec, vec_j);
		if (dist < min_dist[thread_idx]) {
			min_dist[thread_idx] = dist;
			min_index[thread_idx] = i;
		}
	}
	for (i = 1; i < threads; ++i) {
		if (min_dist[i] < min_dist[0]) {
			min_dist[0] = min_dist[i];
			min_index[0] = min_index[i];
		}
	}
	mi = min_index[0];
	
	nv_free(min_index);
	nv_free(min_dist);
	
	return mi;
}

int 
nv_nn(const nv_matrix_t *mat,
	  const nv_matrix_t *vec, int vec_j)
{
	int i;
	int min_index = -1;
	float min_dist = FLT_MAX;
	
	NV_ASSERT(mat->n == vec->n);
	
	for (i = 0; i < mat->m; ++i) {
		float dist = nv_euclidean2(mat, i, vec, vec_j);
		if (dist < min_dist) {
			min_dist = dist;
			min_index = i;
		}
	}
	
	return min_index;
}

nv_int_float_t
nv_nn_dist(const nv_matrix_t *mat,
		   const nv_matrix_t *vec, int vec_j)
{
	int i;
	nv_int_float_t min_v;
	
	min_v.i = -1;
	min_v.f = FLT_MAX;
	
	NV_ASSERT(mat->n == vec->n);
	
	for (i = 0; i < mat->m; ++i) {
		float dist = nv_euclidean2(mat, i, vec, vec_j);
		if (dist < min_v.f) {
			min_v.f = dist;
			min_v.i = i;
		}
	}
	return min_v;
}

static inline bool
operator<(const nv_knn_result_t &d1,
		  const nv_knn_result_t &d2)
{
	if (d1.dist < d2.dist) {
		return true;
	}
	return false;
}

typedef std::priority_queue<nv_knn_result_t, std::vector<nv_knn_result_t>,
							std::less<std::vector<nv_knn_result_t>::value_type> > nv_knn_topn_t;

int 
nv_knn_ex(nv_knn_result_t *results, int n,
		  const nv_matrix_t *mat,
		  const nv_matrix_t *vec,
		  int vec_j,
		  nv_knn_func_t func)
{
	int i, imax;
	int threads = nv_omp_procs();
	nv_knn_topn_t topn;
	std::vector<float> dist_max(threads);
	std::vector<nv_knn_topn_t> topn_temp(threads);
	
#ifdef _OPENMP
#pragma omp parallel for num_threads(threads)
#endif
	for (i = 0; i < mat->m; ++i) {
		int thread_idx = nv_omp_thread_id();
		nv_knn_result_t new_node;
		
		new_node.dist = func(mat, i, vec, vec_j);
		new_node.index = i;
		
		if (topn_temp[thread_idx].size() < (unsigned int)n) {
			topn_temp[thread_idx].push(new_node);
			dist_max[thread_idx] = topn_temp[thread_idx].top().dist;
		} else if (new_node.dist < dist_max[thread_idx]) {
			topn_temp[thread_idx].pop();
			topn_temp[thread_idx].push(new_node);
			dist_max[thread_idx] = topn_temp[thread_idx].top().dist;
		}
	}
	for (i = 0; i < threads; ++i) {
		while (!topn_temp[i].empty()) {
			topn.push(topn_temp[i].top());
			topn_temp[i].pop();
		}
	}
	while (topn.size() > (unsigned int)n) {
		topn.pop();
	}
	imax = NV_MIN(n, mat->m);
	for (i = imax - 1; i >= 0; --i) {
		results[i] = topn.top();
		topn.pop();
	}
	
	return imax;
}

int 
nv_knn(nv_knn_result_t *results, int n,
	   const nv_matrix_t *mat,
	   const nv_matrix_t *vec,
	   int vec_j)
{
	int i, imax;
	nv_knn_topn_t topn;
	float dist_max = FLT_MAX;
	
	for (i = 0; i < mat->m; ++i) {
		nv_knn_result_t new_node;
		
		new_node.dist = nv_euclidean2(mat, i, vec, vec_j);
		new_node.index = i;
		
		if (topn.size() < (unsigned int)n) {
			topn.push(new_node);
			dist_max = topn.top().dist;
		} else if (new_node.dist < dist_max) {
			topn.pop();
			topn.push(new_node);
			dist_max = topn.top().dist;
		}
	}
	
	imax = NV_MIN(n, mat->m);
	for (i = imax - 1; i >= 0; --i) {
		results[i] = topn.top();
		topn.pop();
	}
	
	return imax;
}
