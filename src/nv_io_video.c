/*
 * This file is part of libnv.
 *
 * Copyright (C) 2012 nagadomi@nurs.or.jp
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

/* 書いてから一度も実行してない */

#include "nv_core.h"
#if NV_ENABLE_VIDEO
#include "nv_io.h"
#include "nv_ip.h"
#include "eiio.h"

struct nv_video {
	eiio_video_t *ctx;
};


static nv_matrix_t *
nv_conv_eiio2nv(const eiio_image_t *src)
{
	int x, y;
	nv_matrix_t *dest = nv_matrix3d_alloc(3, src->height, src->width);
	
	for (y = 0; y < src->height; ++y) {
		for (x = 0; x < src->width; ++x) {
			NV_MAT3D_V(dest, y, x, NV_CH_B) = (float)eiio_get_pixel(src, x, y, EIIO_CHANNEL_B);
			NV_MAT3D_V(dest, y, x, NV_CH_G) = (float)eiio_get_pixel(src, x, y, EIIO_CHANNEL_G);
			NV_MAT3D_V(dest, y, x, NV_CH_R) = (float)eiio_get_pixel(src, x, y, EIIO_CHANNEL_R);
		}
	}
	return dest;
}

nv_video_t *nv_video_open(const char *file)
{
	nv_video_t *video = nv_alloc_type(nv_video_t, 1);
	
	video->ctx = eiio_video_open(file);
	if (video->ctx == NULL) {
		nv_free(video);
		return NULL;
	}
	return video;
}

nv_matrix_t *
nv_video_grab(nv_video_t *video)
{
	eiio_image_t *frame = eiio_video_next(video->ctx);
	if (frame) {
		nv_matrix_t *mat = nv_conv_eiio2nv(frame);
		eiio_image_free(&frame);
		
		return mat;
	}
	return NULL;
}

int
nv_video_set_size(nv_video_t *video, int height, int width)
{
	return 0;
}

void
nv_video_close(nv_video_t **video)
{
	if (video && *video) {
		eiio_video_close(&(*video)->ctx);
		nv_free(*video);
		*video = NULL;
	}
}

#endif
