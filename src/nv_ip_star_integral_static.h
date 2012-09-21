/*
 * This file is part of libnv.
 *
 * Copyright (C) 2010-2012 nagadomi@nurs.or.jp
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

#ifndef NV_IP_STAR_INTEGRAL_STATIC_H
#define NV_IP_STAR_INTEGRAL_STATIC_H
#include "nv_core.h"
#include "nv_ip_integral.h"


#ifdef __cplusplus
extern "C" {
#endif

/* star integralの面積(重なっている部分も含む)のグローバルテーブル */

#define NV_STAR_INTEGRAL_AREA_STATIC_MAX 2400

extern NV_DECLARE_DATA float nv_star_integral_area_static[NV_STAR_INTEGRAL_AREA_STATIC_MAX];
extern NV_DECLARE_DATA float nv_star_integral_area_inv_static[NV_STAR_INTEGRAL_AREA_STATIC_MAX];

#if NV_ENABLE_STRICT
float nv_star_integral_area_strict(int r);
float nv_star_integral_area_inv_strict(int r);
	
#define NV_STAR_INTEGRAL_AREA(r) nv_star_integral_area_strict((int)(r))
#define NV_STAR_INTEGRAL_AREA_INV(r) nv_star_integral_area_inv_strict((int)(r))
#else
#define NV_STAR_INTEGRAL_AREA(r) (nv_star_integral_area_static[(int)(r)])
#define NV_STAR_INTEGRAL_AREA_INV(r) (nv_star_integral_area_inv_static[(int)(r)])
#endif

#ifdef __cplusplus
}
#endif

#endif
