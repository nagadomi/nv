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

#ifndef NV_ML_SOM_H
#define NV_ML_SOM_H

/* 自己組織化マップ */

#ifdef __cplusplus
extern "C" {
#endif

#include "nv_core.h"

void nv_som_progress(int onoff);
void nv_som_init(nv_matrix_t *som, const nv_matrix_t *data);
void nv_som_train(nv_matrix_t *som, const nv_matrix_t *data, int max_epoch);
void nv_som_train_ex(nv_matrix_t *som, const nv_matrix_t *data,
					 int start_epoch, int end_epoch, int max_epoch);

void 
nv_som_train_ex2(nv_matrix_t *som, const nv_matrix_t *data,
				 int start_epoch, int end_epoch, int max_epoch);

#ifdef __cplusplus
}
#endif

#endif
