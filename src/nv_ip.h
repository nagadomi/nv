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

#ifndef NV_IP_H
#define NV_IP_H

#include "nv_core.h"

#ifdef __cplusplus
extern "C" {
#endif

/* color channels */
#define NV_CH_B 0
#define NV_CH_G 1
#define NV_CH_R 2
#define NV_CH_MONO 0

#include "nv_ip_gray.h"
#include "nv_ip_integral.h"
#include "nv_ip_laplacian.h"
#include "nv_ip_gaussian.h"
#include "nv_ip_euclidean_color.h"
#include "nv_ip_hsv.h"
#include "nv_ip_ccv.h"
#include "nv_ip_star_integral.h"
#include "nv_ip_star_integral_static.h"
#include "nv_ip_keypoint.h"
#include "nv_ip_resize.h"
#include "nv_ip_flip.h"
#include "nv_ip_bgseg.h"
#include "nv_ip_morphology.h"
#include "nv_ip_patch.h"
#include "nv_ip_pooling.h"

#ifdef __cplusplus
}
#endif

#endif
