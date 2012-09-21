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

#ifndef NV_UTIL_MAP_H
#define NV_UTIL_MAP_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct nv_imap nv_imap_t;

nv_imap_t *nv_imap_alloc(void);
void nv_imap_free(nv_imap_t **map);
void nv_imap_insert(nv_imap_t *map, int key, void *data);
void *nv_imap_find(nv_imap_t *map, int key);
void nv_imap_remove(nv_imap_t *map, int key);
typedef int (*nv_imap_foreach_func_t)(int key, void *data);
int nv_imap_foreach(nv_imap_t *map, nv_imap_foreach_func_t func);
const int *nv_imap_keys(nv_imap_t *map, int *num_keys);

typedef struct nv_smap nv_smap_t;

nv_smap_t *nv_smap_alloc(void);
void nv_smap_free(nv_smap_t **map);
void nv_smap_insert(nv_smap_t *map, const char *key, const void *data);
void *nv_smap_find(nv_smap_t *map, const char *key);
void nv_smap_remove(nv_smap_t *map, const char *key);
typedef int (*nv_smap_foreach_func_t)(const char *key, void *data);
int nv_smap_foreach(nv_smap_t *map, nv_smap_foreach_func_t func);
const char * const *nv_smap_keys(nv_smap_t *map, int *num_keys);

/* for string data only */
void nv_smap_string_insert(nv_smap_t *map, const char *key, const char *string_data);
void nv_smap_string_free(nv_smap_t *map);

#define NV_SMAP_FIND_STRING(map, key) ((const char *)nv_smap_find(map, key))
#define NV_SMAP_FIND_STRING_TO_INT(map, key) (strtol(NV_SMAP_FIND_STRING(map, key), NULL, 10))
#define NV_SMAP_FIND_STRING_TO_FLOAT(map, key) ((float)strtod(NV_SMAP_FIND_STRING(map, key), NULL))

#ifdef __cplusplus
}
#endif

#endif
