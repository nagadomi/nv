/*
 * This file is part of libnv.
 *
 * Copyright (C) 2014 nagadomi@nurs.or.jp
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

#ifndef NV_ML_DAE_H
#define NV_ML_DAE_H

#include "nv_core.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
	NV_DAE_SIGMOID,
	NV_DAE_LINEAR
} nv_dae_type_t;

typedef struct
{
	int input;
	int hidden;
	float noise;
	float sparsity;
	nv_matrix_t *input_w;
	nv_matrix_t *input_bias;
	nv_matrix_t *hidden_bias;
} nv_dae_t;

void nv_dae_progress(int onoff);

nv_dae_t *nv_dae_alloc(int input, int hidden);
void nv_dae_free(nv_dae_t **dae);
void nv_dae_init(nv_dae_t *dae, const nv_matrix_t *data);
void nv_dae_noise(nv_dae_t *dae, float noise);
void nv_dae_sparsity(nv_dae_t *dae, float sparsity_param);
float nv_dae_train(nv_dae_t *dae,
				   const nv_matrix_t *data,
				   float lr,
				   int start_epoch, int end_epoch, int max_epoch);
float nv_dae_train_linear(nv_dae_t *dae,
						  const nv_matrix_t *data,
						  float lr,
						  int start_epoch, int end_epoch, int max_epoch);

float nv_dae_train_ex(nv_dae_t *dae,
					  nv_dae_type_t type,
					  const nv_matrix_t *data,
					  float lr, 
					  int start_epoch, int end_epoch, int max_epoch);


void nv_dae_encode(const nv_dae_t *dae,
				   nv_matrix_t *y,
				   int y_j,
				   const nv_matrix_t *x,
				   int x_j);
void nv_dae_conv2d(const nv_dae_t *dae,
				   nv_matrix_t *output,
				   const nv_matrix_t *patches);

#ifdef __cplusplus
}
#endif

#endif
