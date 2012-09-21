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

#ifndef _NV_IO_STREAM_H
#define _NV_IO_STREAM_H

#include "nv_core.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
	NV_STREAM_WRITE,
	NV_STREAM_READ,
} nv_stream_mode_e;

typedef struct {
	FILE *fp;
	nv_stream_mode_e mode;
	int init_writer;
	nv_matrix_t header;
} nv_stream_t;


nv_stream_t *nv_stream_open(const char *file, nv_stream_mode_e mode);
void nv_stream_close(nv_stream_t **stream);

int nv_stream_read(nv_stream_t *steram, nv_matrix_t *mat, int vec_m);
int nv_stream_write(nv_stream_t *steram, const nv_matrix_t *mat, int vec_m);

#ifdef __cplusplus
}
#endif


#endif

