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

#ifndef NV_IO_VIDEO_H
#define NV_IO_VIDEO_H
#include "nv_core.h"
#if NV_ENABLE_VIDEO

#ifdef __cplusplus
extern "C" {
#endif

typedef struct nv_video nv_video_t;

nv_video_t *nv_video_open(const char *file);
nv_matrix_t *nv_video_grab(nv_video_t *video);
int nv_video_set_size(nv_video_t *video, int height, int width);
void nv_video_close(nv_video_t **video);	

#ifdef __cplusplus
}
#endif

#endif
#endif
