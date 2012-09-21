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

#ifndef NV_CORE_ATOMIC_H
#define NV_CORE_ATOMIC_H
#include "nv_core.h"
#ifdef __cplusplus
extern "C" {
#endif

#ifdef  _MSC_VER
#include <windows.h>
typedef LONG nv_atomic_int_t;
#else
typedef uint32_t nv_atomic_int_t;
#endif

void nv_atomic_incl(nv_atomic_int_t *value);

#ifdef __cplusplus
}
#endif


#endif
