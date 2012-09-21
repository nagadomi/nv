/*
 * This file is part of libnv.
 *
 * Copyright (C) 2011 nagadomi@nurs.or.jp
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

#include "nv_config.h"
#if NV_ENABLE_SSE2

#ifdef __cplusplus
extern "C" {
#endif

__m128 nv_log_ps(__m128 x);
__m128 nv_exp_ps(__m128 x);
__m128 nv_sin_ps(__m128 x);
__m128 nv_cos_ps(__m128 x);
void nv_sincos_ps(__m128 x, __m128 *s, __m128 *c);

#ifdef __cplusplus
}
#endif
	
#endif
