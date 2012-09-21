/*
 * This file is part of libnv.
 *
 * Copyright (C) 2012 nagadomi@nurs.or.jp
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

#ifndef NV_IO_KMEANS_TREE_H
#define NV_IO_KMEANS_TREE_H

#ifdef __cplusplus
extern "C" {
#endif

#include "nv_ml.h"

int nv_save_kmeans_tree_text(const char *filename, const nv_kmeans_tree_t *tree);
nv_kmeans_tree_t *nv_load_kmeans_tree_text(const char *filename);

int nv_save_kmeans_tree_bin(const char *filename, const nv_kmeans_tree_t *tree);
nv_kmeans_tree_t *nv_load_kmeans_tree_bin(const char *filename);

#define nv_save_kmeans_tree(filename, tree) nv_save_kmeans_tree_text(filename, tree)
#define nv_load_kmeans_tree(filename, tree) nv_save_kmeans_tree_text(filename, tree)

#ifdef __cplusplus
}
#endif

#endif

