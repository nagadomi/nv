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

/* TODO: MinGWで結果がおかしいのでバグってるかも・・・  */

#include "nv_core.h"
#include "nv_num.h"
#include "nv_ml_plsi.h"


nv_plsi_t *
nv_plsi_alloc(int d, int w, int k)
{
	nv_plsi_t *p = (nv_plsi_t *)nv_malloc(sizeof(nv_plsi_t));

	p->k = k;
	p->d = d;
	p->w = w;

	p->z = nv_matrix_alloc(p->k, 1);
	p->dz = nv_matrix_alloc(p->k, p->d);
	p->wz = nv_matrix_alloc(p->k, p->w);

	return p;
}

void
nv_plsi_free(nv_plsi_t **p)
{
	if (*p) {
		nv_matrix_free(&(*p)->z);
		nv_matrix_free(&(*p)->dz);
		nv_matrix_free(&(*p)->wz);
		nv_free(*p);
		*p = NULL;
	}
}

void
nv_plsi_init(nv_plsi_t *p)
{
	int d, w;
	float sum;

	nv_vector_rand(p->z, 0, 0.0f, 1.0f);
	sum = nv_vector_sum(p->z, 0);
	if (sum != 0.0f) {
		nv_vector_muls(p->z, 0, p->z, 0, 1.0f / sum);
	}

	for (d = 0; d < p->d; ++d) {
		nv_vector_rand(p->dz, d, 0.0f, 1.0f);
		sum = nv_vector_sum(p->dz, d);
		if (sum != 0.0f) {
			nv_vector_muls(p->dz, 0, p->dz, 0, 1.0f / sum);
		}
	}

	for (w = 0; w < p->w; ++w) {
		nv_vector_rand(p->wz, w, 0.0f, 1.0f);
		sum = nv_vector_sum(p->wz, w);
		if (sum != 0.0f) {
			nv_vector_muls(p->wz, 0, p->wz, 0, 1.0f / sum);
		}
	}
}

void
nv_plsi_emstep(nv_plsi_t *p, const nv_matrix_t *data, float tem_beta)
{
	int i, z, w, d;
	float zsum;
	float zfactor = 0.0f;
	nv_matrix_t *zfq= nv_matrix_alloc(p->k, 1);
	nv_matrix_t *zdw_sum = nv_matrix_alloc(p->w, p->d);
#ifdef _OPENMP
	int threads = nv_omp_procs();
#else
	int threads = 1;
#endif
	nv_matrix_t **zdw_val = (nv_matrix_t **)nv_malloc(sizeof(nv_matrix_t *) * threads);

	for (i = 0; i < threads; ++i) {
		zdw_val[i] = nv_matrix_alloc(p->w, p->d);
	}

	/* sum */
#ifdef _OPENMP
#pragma omp parallel for private(w, z)
#endif
	for (d = 0; d < p->d; ++d) {
		for (w = 0; w < p->w; ++w) {
			float sum = 0.0f;
			for (z = 0; z < p->k; ++z) {
				sum += NV_MAT_V(p->dz, d, z) * (powf(NV_MAT_V(p->wz, w, z) * NV_MAT_V(p->z, 0, z), tem_beta));
			}
			if (sum != 0.0f) {
				NV_MAT_V(zdw_sum, d, w) = 1.0f / sum;
			} else {
				NV_MAT_V(zdw_sum, d, w) = 0.0f;
			}
		}
	}

	zsum = 0.0f;
#ifdef _OPENMP
#pragma omp parallel for private(d, w) reduction(+:zsum)
#endif
	for (z = 0; z < p->k; ++z) {
		float sum = 0.0f;
		float factor = 0.0f;
#ifdef _OPENMP
		int thread_idx = nv_omp_thread_id();
#else
		int thread_idx = 0;
#endif

		/* zdw val */
		for (d = 0; d < p->d; ++d) {
			for (w = 0; w < p->w; ++w) {
				/* (t-1) dz,wz,z */
				NV_MAT_V(zdw_val[thread_idx], d, w) = NV_MAT_V(p->dz, d, z) 
					* (powf(NV_MAT_V(p->wz, w, z) * NV_MAT_V(p->z, 0, z), tem_beta)) * NV_MAT_V(zdw_sum, d, w);
			}
		}
		for (d = 0; d < p->d; ++d) {
			for (w = 0; w < p->w; ++w) {
				sum += NV_MAT_V(data, d, w) * NV_MAT_V(zdw_val[thread_idx], d, w);
			}
		}
		if (sum != 0.0f) {
			factor = 1.0f / sum;
		}

		for (d = 0; d < p->d; ++d) {
			NV_MAT_V(p->dz, d, z) = 0.0f;
		}
		for (w = 0; w < p->w; ++w) {
			NV_MAT_V(p->wz, w, z) = 0.0f;
		}

		/* p(d,z), p(w,z) */
		for (d = 0; d < p->d; ++d) {
			for (w = 0; w < p->w; ++w) {
				float ndwf = NV_MAT_V(data, d, w) * NV_MAT_V(zdw_val[thread_idx], d, w) * factor;
				NV_MAT_V(p->dz, d, z) += ndwf;
				NV_MAT_V(p->wz, w, z) += ndwf;
			}
		}
		zsum += sum;
		NV_MAT_V(zfq, 0, z) = sum;
	}

	/* p(z) */
	if (zsum != 0.0f) {
		zfactor = 1.0f / zsum;
	}
	for (z = 0; z < p->k; ++z) {
		NV_MAT_V(p->z, 0, z) = NV_MAT_V(zfq, 0, z) * zfactor;
	}
	for (i = 0; i < threads; ++i) {
		nv_matrix_free(&zdw_val[i]);
	}
	nv_free(zdw_val);
	nv_matrix_free(&zfq);
	nv_matrix_free(&zdw_sum);
}

void
nv_plsi(nv_plsi_t *p, const nv_matrix_t *data, int step)
{
	int i;

	for (i = 0; i < step; ++i) {
		nv_plsi_emstep(p, data, (1000.0f / (1000.0f + i)));
	}
}
