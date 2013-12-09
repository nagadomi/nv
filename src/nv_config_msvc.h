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
#ifndef NV_CONFIG_MSVC_H
#define NV_CONFIG_MSVC_H

#pragma warning(disable:4819)

#define NV_ENABLE_VIDEO  0
#define NV_WITH_OPENCV   0
#define NV_WITH_EIIO     1
#define NV_ENABLE_OPENSSL 0
#define NV_ENABLE_GLIBC_BACKTRACE 0

#ifdef NDEBUG
#  define NV_ENABLE_STRICT        0
#else
#  define NV_ENABLE_STRICT        1
#endif

#endif
