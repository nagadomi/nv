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

#ifndef NV_ML_AROW_H
#define NV_ML_AROW_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct 
{
	int n;
	int k;
	nv_matrix_t *w;
	nv_matrix_t *bias;
} nv_arow_t;

nv_arow_t *nv_arow_alloc(int n, int k);
void nv_arow_free(nv_arow_t **arow);
	
void nv_arow_init(nv_arow_t *arow);
void nv_arow_train(nv_arow_t *arow,
				   const nv_matrix_t *data, const nv_matrix_t *label,
				   float r,
				   int max_epoch);
int nv_arow_predict_label(const nv_arow_t *arow, const nv_matrix_t *vec, int j);
void nv_arow_dump_c(FILE *out,
					const nv_arow_t *arow, const char *name, int static_variable);

#ifdef __cplusplus
}
#endif
	
#endif

