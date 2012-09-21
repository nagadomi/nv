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

#ifndef NV_ML_XMEANS
#define NV_ML_XMEANS

#ifdef __cplusplus
extern "C" {
#endif

int nv_xmeans(nv_matrix_t *means,  /* max_k */
			  nv_matrix_t *count,  /* max_k */
			  nv_matrix_t *labels, /* data->m */
			  const nv_matrix_t *data,
			  const int max_k, /* 2n */
			  const int max_epoch); /* k-means NV_MAX epoch */


#ifdef __cplusplus
}
#endif


#endif
