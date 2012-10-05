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

#ifndef NV_NORMAL_BAYES_H
#define NV_NORMAL_BAYES_H

#include "nv_core.h"
#include "nv_ml.h"
#include "nv_num.h"


#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
	int k;
	int n;
	nv_cov_t **kcov;
	nv_matrix_t *pk;
} nv_nb_t;

nv_nb_t *nv_nb_alloc(int n, int k);
void nv_nb_train(nv_nb_t *nb, const nv_matrix_t *data, int k);
void nv_nb_train_all(nv_nb_t *nb,
					 int k,
					 const nv_matrix_t *data,
					 const nv_matrix_t *labels);
void nv_nb_train_finish(nv_nb_t *nb);
int nv_nb_predict_label(const nv_nb_t *nb, const nv_matrix_t *x, int xm, int npca);
float nv_nb_predict(const nv_nb_t *nb, const nv_matrix_t *x, int xm, int pca, int k);

void nv_nb_free(nv_nb_t **nb);


#ifdef __cplusplus
}
#endif

#endif
