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

#ifndef NV_IO_COV_H
#define NV_IO_COV_H

#ifdef __cplusplus
extern "C" {
#endif

#include "nv_num.h"

void nv_save_cov(const char *filename, const nv_cov_t *cov);
nv_cov_t *nv_load_cov(const char *filename);
void nv_save_cov_fp(FILE *fp, const nv_cov_t *cov);
nv_cov_t *nv_load_cov_fp(FILE *fp);

#ifdef __cplusplus
}
#endif

#endif
