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

#ifndef NV_ML_UTIL_H
#define NV_ML_UTIL_H

#ifdef __cplusplus
extern "C" {
#endif

void nv_dataset(nv_matrix_t *data,
				nv_matrix_t *labels,
				nv_matrix_t *train_data,
				nv_matrix_t *train_labels,
				nv_matrix_t *test_data,
				nv_matrix_t *test_labels);

float nv_purity(int cluster_k,
				int correct_k,
				const nv_matrix_t *clustering_labels,
				const nv_matrix_t *correct_labels);


#ifdef __cplusplus
}
#endif

#endif
