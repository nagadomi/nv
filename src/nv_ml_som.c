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
#include "nv_ml.h"

/* 自己組織化マップ */

#define NV_SOM_LR 0.1f

static int nv_som_progress_flag = 0;

void
nv_som_progress(int onoff)
{
	nv_som_progress_flag = onoff;
}

void 
nv_som_init(nv_matrix_t *som, const nv_matrix_t *data)
{
	int n;

	for (n = 0; n < data->n; ++n) {
		int m;
		float nmax = FLT_MIN;
		float nmin = FLT_MAX;

		for (m = 0; m < data->m; ++m) {
			if (nmax < NV_MAT_V(data, m, n)) {
				nmax = NV_MAT_V(data, m, n);
			}
			if (nmin > NV_MAT_V(data, m, n)) {
				nmin = NV_MAT_V(data, m, n);
			}
		}
		for (m = 0; m < som->m; ++m) {
			NV_MAT_V(som, m, n) = nmin + (nmax - nmin) * nv_rand();
		}
	}
}

void 
nv_som_train(nv_matrix_t *som, const nv_matrix_t *data, int max_epoch)
{
	nv_som_train_ex(som, data, 0, max_epoch, max_epoch);
}

void 
nv_som_train_ex(nv_matrix_t *som, const nv_matrix_t *data,
				int start_epoch, int end_epoch, int max_epoch)
{
	int t, m, i;
	long tm;
	float w = (NV_SOM_LR / data->m) * (400.0f / max_epoch);
	/*float w = (NV_SOM_LR / data->m) * 0.001f; */
	float deno = 1.0f / ((float)max_epoch / logf(0.6f * NV_MAX(som->cols, som->rows)));
	int threads;
	nv_matrix_t **d;
#ifdef _OPENMP
	threads = nv_omp_procs();
#else
	threads = 1;
#endif

	d = (nv_matrix_t **)nv_malloc(sizeof(nv_matrix_t *) * threads);
	for (i = 0; i < threads; ++i) {
		d[i] = nv_matrix3d_alloc(som->n, som->rows, som->cols);
	}

	for (t = start_epoch; t < end_epoch; ++t) {
		int radius = (int)(0.6f * NV_MAX(som->cols, som->rows) * expf(-t * deno));
		float radius2 = (float)radius * (float)radius;
		int l;
		
		tm = nv_clock();
		
		for (l = 0; l < 5; ++l) {
			for (i = 0; i < threads; ++i) {
				nv_matrix_zero(d[i]);
			}

#ifdef _OPENMP
#pragma omp parallel for num_threads(threads)
#endif
			for (m = 0; m < data->m; ++m) {
				int row;
				int bmu = nv_nn(som, data, m);
				int bmcol = NV_MAT_COL(som, bmu);
				int bmrow = NV_MAT_ROW(som, bmu);
				int srow = NV_MAX(0, bmrow - radius);
				int erow = NV_MIN(som->rows, bmrow + radius);
				int scol = NV_MAX(0, bmcol - radius);
				int ecol = NV_MIN(som->cols, bmcol + radius);

#ifdef _OPENMP
				int thread_idx = nv_omp_thread_id();
#else
				int thread_idx = 0;
#endif

				for (row = srow; row < erow; ++row) {
					int col;

					for (col = scol; col < ecol; ++col) {
						float dist2 = (float)((row - bmrow) * (row - bmrow) + (col - bmcol) * (col - bmcol));
						if (dist2 < radius2) {
							int n;
							float a = expf(-(dist2 / (2.0f * radius2)));
							for (n = 0; n < som->n; ++n) {
								NV_MAT3D_V(d[thread_idx], row, col, n) += a * (NV_MAT_V(data, m, n) - NV_MAT3D_V(som, row, col, n));
							}
						}
					}
				}
			}
			for (i = 0; i < threads; ++i) {
				for (m = 0; m < d[i]->m; ++m) {
					int n;
					for (n = 0; n < d[i]->n; ++n) {
						NV_MAT_V(som, m, n) += w * expf(-(t / max_epoch)) * NV_MAT_V(d[i], m, n);
					}
				}
			}
		}
		if (nv_som_progress_flag) {
			printf("%d: som train: radius: %d, %ldms\n",
				t, radius, nv_clock() - tm);
		}
	}

	for (i = 0; i < threads; ++i) {
		nv_matrix_free(&d[i]);
	}
	nv_free(d);
}

void 
nv_som_train_ex2(nv_matrix_t *som, const nv_matrix_t *data,
				 int start_epoch, int end_epoch, int max_epoch)
{
	int t;
	/* float w = (NV_SOM_LR / data->m) * (1000.0f / max_epoch); */
	float w = (NV_SOM_LR / data->m);
	float deno = 1.0f / ((float)max_epoch / logf(0.5f * som->cols));
	int threads;
#ifdef _OPENMP
	threads = nv_omp_procs();
#else
	threads = 1;
#endif

	for (t = start_epoch; t < end_epoch; ++t) {
		int radius = (int)((0.5f * NV_MAX(som->cols, som->rows) * expf(-t * deno)) * 2.0f);
		float radius2 = (float)radius * (float)radius;
		int k, l;
		long tm = nv_clock();

		for (l = 0; l < 5; ++l) {
#ifdef _OPENMP
#pragma omp parallel for num_threads(threads)
#endif
			for (k = 0; k < data->m; ++k) {
				int m = nv_rand_index(data->m);
				int row;
				int bmu = nv_nn(som, data, m);
				int bmcol = NV_MAT_COL(som, bmu);
				int bmrow = NV_MAT_ROW(som, bmu);
				int srow = NV_MAX(0, bmrow - radius);
				int erow = NV_MIN(som->rows, bmrow + radius);
				int scol = NV_MAX(0, bmcol - radius);
				int ecol = NV_MIN(som->cols, bmcol + radius);

				for (row = srow; row < erow; ++row) {
					int col;

					for (col = scol; col < ecol; ++col) {
						float dist2 = (float)((row - bmrow) * (row - bmrow) + (col - bmcol) * (col - bmcol));
						if (dist2 < radius2) {
							int n;
							float a = expf(-(dist2 / (2.0f * radius2)));
							for (n = 0; n < som->n; ++n) {
								NV_MAT3D_V(som, row, col, n) += (w * expf(-(t / max_epoch)) * a) * (NV_MAT_V(data, m, n) - NV_MAT3D_V(som, row, col, n));
							}
						}
					}
				}
			}
		}
		if (nv_som_progress_flag) {
			printf("%d: som train: radius: %d, %ldms\n",
				t, radius, nv_clock() - tm);
		}
	}
}
