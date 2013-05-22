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

#ifndef NV_IO_MATRIX_H
#define NV_IO_MATRIX_H

#include "nv_core.h"
#ifdef __cplusplus
extern "C" {
#endif

nv_matrix_t *nv_load_matrix_fp(FILE *fp);
void nv_save_matrix_fp(FILE *fp, const nv_matrix_t *mat);

nv_matrix_t *nv_load_matrix_bin_fp(FILE *fp);
void nv_save_matrix_bin_fp(FILE *fp, const nv_matrix_t *mat);

nv_matrix_t *nv_load_matrix(const char *filename);
int nv_save_matrix(const char *filename, const nv_matrix_t *mat);

int nv_load_matrix_array_text(const char *filename, nv_matrix_t **array, int *len);
int nv_save_matrix_array_text(const char *filename, nv_matrix_t **array, int len);
int nv_load_matrix_array_bin(const char *filename, nv_matrix_t **array, int *len);
int nv_save_matrix_array_bin(const char *filename, nv_matrix_t **array, int len);

#define nv_load_matrix_text(filename) nv_load_matrix(filename)
#define nv_save_matrix_text(filename, mat) nv_save_matrix(filename, mat)

char *nv_serialize_matrix(const nv_matrix_t *mat);
nv_matrix_t *nv_deserialize_matrix(const char *s);

nv_matrix_t *nv_load_matrix_bin(const char *filename);
int nv_save_matrix_bin(const char *filename, const nv_matrix_t *mat);

#ifdef __cplusplus
}
#endif
#endif
