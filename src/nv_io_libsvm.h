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

#ifndef NV_IO_LIBSVM_H
#define NV_IO_LIBSVM_H

#ifdef __cplusplus
extern "C" {
#endif

#include "nv_core.h"

int nv_save_libsvm_fp(FILE *fp, const nv_matrix_t *data, const nv_matrix_t *labels);
int nv_save_libsvm(const char *file, const nv_matrix_t *data, const nv_matrix_t *labels);

#ifdef __cplusplus
}
#endif

#endif
