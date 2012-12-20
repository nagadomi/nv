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

#ifndef NV_ML_PA
#define NV_ML_PA

typedef struct 
{
	int n;
	int k;
	nv_matrix_t *w;
} nv_pa_t;

void nv_pa_progress(int onoff);
nv_pa_t *nv_pa_alloc(int n, int k);
void nv_pa_free(nv_pa_t **pa);
	
void nv_pa_init(nv_pa_t *pa);
void nv_pa_train(nv_pa_t *pa,
				   const nv_matrix_t *data, const nv_matrix_t *label,
				   float r,
				   int max_epoch);
int nv_pa_predict_label(const nv_pa_t *pa, const nv_matrix_t *vec, int j);
void nv_pa_dump_c(FILE *out,
					const nv_pa_t *pa, const char *name, int static_variable);


#endif
