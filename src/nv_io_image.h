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

#ifndef NV_IO_IMAGE_H
#define NV_IO_IMAGE_H

#include "nv_core.h"
#if NV_WITH_EIIO

#ifdef __cplusplus
extern "C" {
#endif

nv_matrix_t *nv_load_image(const char *filename);
nv_matrix_t *nv_decode_image(const void *blob, size_t len);

#define nv_load(filename) nv_load_image(filename)

#if NV_WINDOWS
nv_matrix_t *nv_decode_dib(HBITMAP hDIB);
#endif

#ifdef __cplusplus
}
#endif

#endif
#endif
