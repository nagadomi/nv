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
#include "nv_io.h"
#include "nv_num.h"

void
nv_save_cov_fp(FILE *fp, const nv_cov_t *cov)
{
	fprintf(fp, "%d %d\n", cov->n, cov->data_m);
	nv_save_matrix_fp(fp, cov->u);
	nv_save_matrix_fp(fp, cov->cov);
	nv_save_matrix_fp(fp, cov->eigen_vec);
	nv_save_matrix_fp(fp, cov->eigen_val);
}

nv_cov_t *
nv_load_cov_fp(FILE *fp)
{
	int n, fn, data_m;
	nv_cov_t *cov = (nv_cov_t *)nv_malloc(sizeof(nv_cov_t));

	fn = fscanf(fp, "%d %d", &n, &data_m);
	if (fn != 2) {
		nv_cov_free(&cov);
		return NULL;
	}

	cov->n = n;
	cov->data_m = data_m;
	cov->u = nv_load_matrix_fp(fp);
	cov->cov = nv_load_matrix_fp(fp);
	cov->eigen_vec = nv_load_matrix_fp(fp);
	cov->eigen_val = nv_load_matrix_fp(fp);

	NV_ASSERT(cov->u->n == n);
	NV_ASSERT(cov->u->m == 1);
	NV_ASSERT(cov->cov->n == n);
	NV_ASSERT(cov->cov->m == n);
	NV_ASSERT(cov->eigen_vec->n == n);
	NV_ASSERT(cov->eigen_vec->m == n);
	NV_ASSERT(cov->eigen_val->n == 1);
	NV_ASSERT(cov->eigen_val->m == n);

	return cov;
}

void nv_save_cov(const char *filename, const nv_cov_t *cov)
{
	FILE *fp = fopen(filename, "w");

	if (fp == NULL) {
		perror(filename);
		NV_ASSERT(0);
		return;
	}
	nv_save_cov_fp(fp, cov);

	fclose(fp);
}

nv_cov_t *nv_load_cov(const char *filename)
{
	FILE *fp = fopen(filename, "r");
	nv_cov_t *cov;
	
	if (fp == NULL) {
		perror(filename);
		return NULL;
	}
	cov = nv_load_cov_fp(fp);
	fclose(fp);
	
	return cov;
}
