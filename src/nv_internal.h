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

#ifndef NV_INTERNAL_H
#define NV_INTERNAL_H

#include "nv_config.h"
#include "nv_portable.h"

#if NV_MSVC
#  ifdef _DEBUG
#    define _CRTDBG_MAP_ALLOC
#    include <crtdbg.h>
#  endif
#endif


#if NV_ENABLE_AVX
#else
#  if NV_ENABLE_AVX_EMU
#    define NV_ENABLE_AVX_EMU 1
#    undef  NV_ENABLE_AVX
#    define NV_ENABLE_AVX     1
#    include "avxintrin_emu.h"
#  endif
#endif

#endif
