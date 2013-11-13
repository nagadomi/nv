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

#ifndef NV_ML_LR_H
#define NV_ML_LR_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct 
{
	int n;
	int k;
	nv_matrix_t *w;
} nv_lr_t;

typedef enum {
	NV_LR_REG_NONE = 0,
	NV_LR_REG_L1,
	NV_LR_REG_L2,
} nv_lr_regulaization_e;

typedef struct 
{
	float grad_w;
	nv_lr_regulaization_e reg_type;
	float reg_w;
	int max_epoch;
	int auto_balance;
} nv_lr_param_t;

void nv_lr_progress(int onoff);
nv_lr_t *nv_lr_alloc(int n, int k);
void nv_lr_free(nv_lr_t **lr);
void nv_lr_init(nv_lr_t *lr, const nv_matrix_t *data);

#define NV_LR_PARAM(max_epoch, grad_w, reg_type, reg_w, auto_balance)			\
	nv_lr_param_create((max_epoch), (grad_w), (reg_type), (reg_w), (auto_balance))
nv_lr_param_t nv_lr_param_create(int max_epoch,
								 float grad_w,
								 nv_lr_regulaization_e reg_type, 
								 float reg_w,
								 int auto_balance);
void nv_lr_train(nv_lr_t *lr,
				 const nv_matrix_t *data, const nv_matrix_t *label,
				 nv_lr_param_t param);

float nv_lr_predict(const nv_lr_t *lr, const nv_matrix_t *x, int xm, int cls);
int nv_lr_predict_label(const nv_lr_t *lr, const nv_matrix_t *x, int xm);
void nv_lr_predict_vector(const nv_lr_t *lr, nv_matrix_t *y, int yj, 
		     const nv_matrix_t *data, int dj);
nv_int_float_t
nv_lr_predict_label_and_probability(const nv_lr_t *lr,
									const nv_matrix_t *x, int xj);
	

void nv_lr_dump_c(FILE *out, const nv_lr_t *lr, const char *name, int static_variable);

#ifdef __cplusplus
}
#endif

#endif
