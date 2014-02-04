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

#ifndef NV_CORE_RAND_H
#define NV_CORE_RAND_H

#ifdef __cplusplus
extern "C" {
#endif

void nv_rand_init(void);
void nv_srand_time(void);
void nv_srand(unsigned int seed);
float nv_rand(void);      /* [0 - 1] */
int nv_rand_index(int n); /* [0 - n) */

float nv_gaussian_rand(float average, float variance);

void nv_shuffle_index(int *a, int start, int end);
void nv_vector_shuffle(nv_matrix_t *mat);
void nv_vector_shuffle_pair(nv_matrix_t *mat1, nv_matrix_t *mat2);


#ifdef __cplusplus
}
#endif

#endif
