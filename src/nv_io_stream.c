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

#include "nv_core.h"
#include "nv_io_stream.h"

nv_stream_t *
nv_stream_open(const char *file, nv_stream_mode_e mode)
{
	const char *fmode = mode == NV_STREAM_WRITE ? "w":"r";
	FILE *fp = fopen(file, fmode);
	nv_stream_t *stream;
	int n;
	if (fp == NULL) {
		return NULL;
	}

	stream = (nv_stream_t *)nv_malloc(sizeof(nv_stream_t));
	memset(stream, 0, sizeof(nv_stream_t));

	if (mode == NV_STREAM_READ) {
		n = fscanf(fp, "%d %d %d %d %d ",
			   &stream->header.list,
			   &stream->header.m,
			   &stream->header.n,
			   &stream->header.rows,
			   &stream->header.cols);
		if (n != 5) {
			fclose(fp);
			nv_free(stream);
			return NULL;
		}
		
	} else if (mode == NV_STREAM_WRITE) {

	} else {
		NV_ASSERT(0);
	}
	stream->fp = fp;
	stream->mode = mode;

	return stream;
}

void 
nv_stream_close(nv_stream_t **stream)
{
	if (stream && *stream) {
		fclose((*stream)->fp);
		nv_free(*stream);
		*stream = NULL;
	}
}

int nv_stream_read(nv_stream_t *stream, nv_matrix_t *mat, int vec_m)
{
	int n, ret = 0;

	NV_ASSERT(mat->n == stream->header.n);
	NV_ASSERT(stream->mode == NV_STREAM_READ);

	for (n = 0; n < stream->header.n; ++n) {
		ret = fscanf(stream->fp, "%E ", &NV_MAT_V(mat, vec_m, n));
		if (ret <= 0) {
			ret = 0;
			break;
		}
	}

	return ret;
}

int 
nv_stream_write(nv_stream_t *stream, const nv_matrix_t *mat, int vec_m)
{
	int n;

	NV_ASSERT(stream->mode == NV_STREAM_WRITE);

	if (!stream->init_writer) {
		/* 初期化 */
		stream->header.list = 1;
		stream->header.m = -1;
		stream->header.n = mat->n;
		stream->header.cols = -1;
		stream->header.rows = -1;

		fprintf(stream->fp, "%d %d %d %d %d\n", 
			stream->header.list,
			stream->header.m,
			stream->header.n,
			stream->header.rows,
			stream->header.cols);
		/* 初期化済み */
		stream->init_writer = 1;
	}

	for (n = 0; n < mat->n; ++n) {
		if (n != 0) {
			fprintf(stream->fp, " ");
		}
		fprintf(stream->fp, "%E", NV_MAT_V(mat, vec_m, n));
	}
	fprintf(stream->fp, "\n");

	return 1;
}
