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

#ifndef NV_IO_H
#define NV_IO_H

#include "nv_core.h"
#if NV_WITH_OPENCV
#include "nv_io_ipl.h"
#endif
#include "nv_io_matrix.h"
#include "nv_io_image.h"
#include "nv_io_stream.h"
#include "nv_io_mlp.h"
#include "nv_io_dae.h"
#include "nv_io_cov.h"
#include "nv_io_nb.h"
#include "nv_io_lr.h"
#include "nv_io_video.h"
#include "nv_io_kmeans_tree.h"
#include "nv_io_klr_tree.h"
#include "nv_io_libsvm.h"

#endif
