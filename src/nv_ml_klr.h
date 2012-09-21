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

#ifndef NV_ML_KLR_H
#define NV_ML_KLR_H
#include "nv_core.h"
#ifdef __cplusplus
extern "C" {
#endif

/* k-means + logistic regression clustering */

void nv_klr_progress(int onoff);

void
nv_klr_train(nv_lr_t *lr,
			 nv_matrix_t *count,
			 nv_matrix_t *labels,
			 const nv_matrix_t *data,
			 const nv_lr_param_t param,
			 const int max_epoch);

void 
nv_klr_init(nv_lr_t *irls,
			nv_matrix_t *count,
			nv_matrix_t *labels,
			const nv_matrix_t *data,
			const nv_lr_param_t param);

int 
nv_klr_em(nv_lr_t *irls,
		  nv_matrix_t *count,
		  nv_matrix_t *labels,
		  const nv_matrix_t *data,
		  const nv_lr_param_t param,
		  const int max_epoch);

#ifdef __cplusplus
}
#endif

#endif
