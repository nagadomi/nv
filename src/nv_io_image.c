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

#if NV_WITH_EIIO
#include "nv_ip.h"
#include "nv_io.h"
#include "eiio.h"

static nv_matrix_t *
nv_conv_eiio2nv(const eiio_image_t *src)
{
	int x, y;
	nv_matrix_t *dest = nv_matrix3d_alloc(3, src->height, src->width);

	for (y = 0; y < dest->rows; ++y) {
		for (x = 0; x < dest->cols; ++x) {
			NV_MAT3D_V(dest, y, x, NV_CH_B) = (float)eiio_get_pixel(src, x, y, EIIO_CHANNEL_B);
			NV_MAT3D_V(dest, y, x, NV_CH_G) = (float)eiio_get_pixel(src, x, y, EIIO_CHANNEL_G);
			NV_MAT3D_V(dest, y, x, NV_CH_R) = (float)eiio_get_pixel(src, x, y, EIIO_CHANNEL_R);
		}
	}
	
	return dest;
}

nv_matrix_t *
nv_load_image(const char *filename)
{
	eiio_image_t *image = eiio_read_file(filename);
	nv_matrix_t *nvmat;

	if (image == NULL) {
		return NULL;
	}
	nvmat = nv_conv_eiio2nv(image);

	eiio_image_free(&image);

	return nvmat;
}

nv_matrix_t *
nv_decode_image(const void *blob, size_t len)
{
	eiio_image_t *image = eiio_read_blob(blob, len);
	nv_matrix_t *nvmat;

	if (image == NULL) {
		return NULL;
	}
	nvmat = nv_conv_eiio2nv(image);
	eiio_image_free(&image);

	return nvmat;
}

#if NV_WINDOWS
nv_matrix_t *nv_decode_dib(HBITMAP hDIB)
{
	eiio_image_t *image = eiio_read_bitmap_handle(hDIB);
	nv_matrix_t *nvmat;

	if (image == NULL) {
		return NULL;
	}
	nvmat = nv_conv_eiio2nv(image);
	eiio_image_free(&image);

	return nvmat;
}
#endif

#endif
