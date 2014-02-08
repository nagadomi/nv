/*
 * This file is part of libnv.
 *
 * Copyright (C) 2014 nagadomi@nurs.or.jp
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

#ifndef NV_IO_DAE_H
#define NV_IO_DAE_H

#include "nv_core.h"
#include "nv_ml_dae.h"
#ifdef __cplusplus
extern "C" {
#endif

nv_dae_t *nv_load_dae_text(const char *filename);
int nv_save_dae_text(const char *filename, const nv_dae_t *dae);

#define nv_load_dae(filename) nv_load_dae_text(filename)
#define nv_save_dae(filename, dae) nv_save_dae_text(filename, dae)

#ifdef __cplusplus
}
#endif

#endif
