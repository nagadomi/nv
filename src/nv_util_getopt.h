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

#ifndef NV_UTIL_GETOPT_H
#define NV_UTIL_GETOPT_H
#include "nv_core.h"

#ifdef __cplusplus
extern "C" {
#endif

extern NV_DECLARE_DATA int nv_getopt_optind;
extern NV_DECLARE_DATA int nv_getopt_optopt;
extern NV_DECLARE_DATA int nv_getopt_opterr;
extern NV_DECLARE_DATA char *nv_getopt_optarg;

int
nv_getopt(int argc, char **argv, const char *opts);

#ifdef __cplusplus
}
#endif

#endif

