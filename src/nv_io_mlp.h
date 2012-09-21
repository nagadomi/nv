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

#ifndef NV_IO_MLP_H
#define NV_IO_MLP_H

#include "nv_core.h"
#include "nv_ml_mlp.h"
#ifdef __cplusplus
extern "C" {
#endif

nv_mlp_t *nv_load_mlp_text(const char *filename);
int nv_save_mlp_text(const char *filename, const nv_mlp_t *mlp);

#define nv_load_mlp(filename) nv_load_mlp_text(filename)
#define nv_save_mlp(filename, mlp) nv_save_mlp_text(filename, mlp)

#ifdef __cplusplus
}
#endif

#endif
