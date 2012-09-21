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

#ifndef NV_ML_LBGU_H
#define NV_ML_LBGU_H

#ifdef __cplusplus
extern "C" {
#endif

#include "nv_core.h"
#include "nv_ml_kmeans.h"

/* 進行状態の表示(print) ON/OFF, default OFF */
void nv_lbgu_progress(int onoff);

/* LBG-U, 初期化+実行 */
int nv_lbgu(nv_matrix_t *means,  // k
			nv_matrix_t *count,  // k
			nv_matrix_t *labels, // data->m
			const nv_matrix_t *data,
			const int k,
			const int kmeans_max_epoch,
			const int max_epoch);

/* 初期化, データの扱い方はk-meansと同じなので初期化も同じ関数が使える */

/* 実行 */
int nv_lbgu_em(nv_matrix_t *means,  /* k */
			   nv_matrix_t *count,  /* k */
			   nv_matrix_t *labels, /* data->m */
			   const nv_matrix_t *data,
			   const int k,
			   const int kmeans_max_epoch,
			   const int max_epoch);

#ifdef __cplusplus
}
#endif



#endif
