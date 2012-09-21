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
#ifndef NV_CONFIG_H
#define NV_CONFIG_H

#if (defined(NV_BUILD_LIB) && HAVE_CONFIG_H)
#  include "config.h"
#endif

#ifdef _MSC_VER
#  include "nv_config_msvc.h"
#else
#  include "nv_config_auto.h"
#endif

#ifdef __SSE__
#  define NV_ENABLE_SSE   1
#else
#  define NV_ENABLE_SSE   0
#endif
#ifdef __SSE2__
#  define NV_ENABLE_SSE2   1
#else
#  define NV_ENABLE_SSE2   0
#endif
#ifdef __SSE3__
#  define NV_ENABLE_SSE3   1
#else
#  define NV_ENABLE_SSE3   0
#endif
#ifdef __SSE4_1__
#  define NV_ENABLE_SSE4_1   1
#else
#  define NV_ENABLE_SSE4_1   0
#endif
#ifdef __SSE4_2__
#  define NV_ENABLE_SSE4_2   1
#else
#  define NV_ENABLE_SSE4_2   0
#endif
#ifdef __POPCNT__
#  define NV_ENABLE_POPCNT   1
#else
#  define NV_ENABLE_POPCNT   0
#endif
#ifdef __AVX__
#  define NV_ENABLE_AVX      1
#else
#  define NV_ENABLE_AVX      0
#endif

#ifdef NV_INTERNAL
#  include "nv_internal.h"
#endif

#endif
