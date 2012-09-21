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

#ifndef _NV_ML_MLP3_H
#define _NV_ML_MLP3_H

/* 3-layer-nn */
#ifdef __cplusplus
extern "C" {
#endif

typedef struct
{
	int input;
	int hidden1;
	int hidden2;
	int output;
	nv_matrix_t *input_w;
	nv_matrix_t *hidden1_w;
	nv_matrix_t *hidden2_w;
	nv_matrix_t *input_bias;
	nv_matrix_t *hidden1_bias;
	nv_matrix_t *hidden2_bias;
 } nv_mlp3_t;

nv_mlp3_t *nv_mlp3_alloc(int input, int hidden1, int hidden2, int k);
void nv_mlp3_free(nv_mlp3_t **mlp);
void nv_mlp3_init(nv_mlp3_t *mlp);
float nv_mlp3_train_lex(nv_mlp3_t *mlp,
					 const nv_matrix_t *data,
					 const nv_matrix_t *label,
					 const nv_matrix_t *t,
					 float ir, float hr1, float hr2, 
					 int start_epoch, int end_epoch, int max_epoch);

float nv_mlp3_train_ex(nv_mlp3_t *mlp,
					  const nv_matrix_t *data,
					  const nv_matrix_t *label,
					  float ir, float hr1, float hr2,
					  int start_epoch, int end_epoch,
					  int max_epoch);


int nv_mlp3_predict_label(const nv_mlp3_t *mlp, const nv_matrix_t *x, int xm);
float nv_mlp3_predict(const nv_mlp3_t *mlp, const nv_matrix_t *x, int xm, int cls);
void nv_mlp3_hidden_vector(const nv_mlp3_t *mlp, 
						  const nv_matrix_t *x, int xm, nv_matrix_t *out, int om);


#ifdef __cplusplus
}
#endif


#endif
