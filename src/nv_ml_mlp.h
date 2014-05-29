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

#ifndef NV_ML_MLP_H
#define NV_ML_MLP_H

#include "nv_core.h"

#ifdef __cplusplus
extern "C" {
#endif

/* 2-layer-nn */
	
typedef struct
{
	int input;
	int hidden;
	int output;
	float dropout;
	float noise;
	nv_matrix_t *input_w;
	nv_matrix_t *hidden_w;
	nv_matrix_t *input_bias;
	nv_matrix_t *hidden_bias;
 } nv_mlp_t;

void nv_mlp_progress(int onoff);

nv_mlp_t *nv_mlp_alloc(int input, int hidden, int k);
void nv_mlp_free(nv_mlp_t **mlp);

void nv_mlp_init(nv_mlp_t *mlp, const nv_matrix_t *data);
void nv_mlp_init_rand(nv_mlp_t *mlp, const nv_matrix_t *data);
void nv_mlp_gaussian_init(nv_mlp_t *mlp, float var, int height, int width, int zdim);

void nv_mlp_dropout(nv_mlp_t *mlp, float dropout);
void nv_mlp_noise(nv_mlp_t *mlp, float noise);
void nv_mlp_make_t(nv_matrix_t *t, const nv_matrix_t *label);
float nv_mlp_train_ex(nv_mlp_t *mlp,
					  const nv_matrix_t *data, const nv_matrix_t *label,
					  float ir, float hr,
					 int start_epoch, int end_epoch, int max_epoch);
float nv_mlp_train_lex(nv_mlp_t *mlp,
					   const nv_matrix_t *data,
					   const nv_matrix_t *label,
					   const nv_matrix_t *t,
					   float ir, float hr,
					   int start_epoch, int end_epoch, int max_epoch);
float nv_mlp_train(nv_mlp_t *mlp, const nv_matrix_t *data, const nv_matrix_t *label, int epoch);

int nv_mlp_predict_label(const nv_mlp_t *mlp, const nv_matrix_t *x, int xm);
float nv_mlp_predict(const nv_mlp_t *mlp, const nv_matrix_t *x, int xm, int cls);
void nv_mlp_predict_vector(const nv_mlp_t *mlp,
						   nv_matrix_t *p, int p_j,
						   const nv_matrix_t *x, int x_j);

float nv_mlp_bagging_predict(const nv_mlp_t **mlp, int nmlp, 
							 const nv_matrix_t *x, int xm, int cls);

void nv_mlp_train_regression(nv_mlp_t *mlp, const nv_matrix_t *data,
							 const nv_matrix_t *t, float ir, float hr, int start_epoch, int max_epoch);
void nv_mlp_regression(const nv_mlp_t *mlp, const nv_matrix_t *x, int xm, nv_matrix_t *out, int om);

void nv_mlp_hidden_vector(const nv_mlp_t *mlp, 
						  const nv_matrix_t *x, int xm, nv_matrix_t *out, int om);

void nv_mlp_dump_c(FILE *out, const nv_mlp_t *mlp, const char *name, int static_variable);

#ifdef __cplusplus
}
#endif

#endif
