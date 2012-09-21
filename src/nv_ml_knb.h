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

#ifndef NV_KNB_H
#define NV_KNB_H

#ifdef __cplusplus
extern "C" {
#endif

/* Normal/Naive Bayes clustering */

void 
nv_knb_progress(int onoff);

void 
nv_knb_init(nv_nb_t *nb,         /* k */
			nv_matrix_t *count,  /* k */
			nv_matrix_t *labels, /* data->m */
			const nv_matrix_t *data);

int 
nv_knb_em(nv_nb_t *nb,         /* k */
		  nv_matrix_t *count,  /* k */
		  nv_matrix_t *labels, /* data->m */
		  const nv_matrix_t *data,
		  const int npca,      /*  使用する固有ベクトルの数. data->n * 0.6くらいが妥当 */
		  const int max_epoch);


#ifdef __cplusplus
}
#endif

#endif
