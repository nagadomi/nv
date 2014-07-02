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

#ifndef NV_IP_POOLING_H
#define NV_IP_POOLING_H
#ifdef __cplusplus
extern "C" {
#endif

void
nv_max_pooling(nv_matrix_t *output,
			   const nv_matrix_t *conv,
			   int pooling_size,
			   int stride);
void
nv_average_pooling(nv_matrix_t *output,
				   const nv_matrix_t *conv,
				   int pooling_size,
				   int stride);

#ifdef __cplusplus
}
#endif
#endif
