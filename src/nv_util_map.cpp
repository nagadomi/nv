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

#include <map>
#include <string>
#include "nv_core.h"

/* int map */

struct nv_imap {
	std::map<int, void *> map;
	int *keys;
};

nv_imap_t *
nv_imap_alloc(void)
{
	nv_imap_t *map = new nv_imap_t;
	map->keys = NULL;
	
	return map;
}

void 
nv_imap_free(nv_imap_t **imap)
{
	if (imap && *imap) {
		if ((*imap)->keys) {
			nv_free((*imap)->keys);
			(*imap)->keys = NULL;
		}
		delete *imap;
		*imap = NULL;
	}
}

void 
nv_imap_insert(nv_imap_t *imap, int key, void *data)
{
	imap->map.insert(std::make_pair(key, data));
}

void *
nv_imap_find(nv_imap_t *imap, int key)
{
	std::map<int, void *>::iterator it = imap->map.find(key);
	if (it != imap->map.end()) {
		return it->second;
	}
	return NULL;
}

void 
nv_imap_remove(nv_imap_t *imap, int key)
{
	imap->map.erase(key);
}

int 
nv_imap_foreach(nv_imap_t *imap, nv_imap_foreach_func_t func)
{
	std::map<int, void *>::iterator it;
	int ret;
	for (it = imap->map.begin(); it != imap->map.end(); it++) {
		if ((ret = func(it->first, it->second)) != 0) {
			return ret;
		}
	}
	return 0;
}

const int *nv_imap_keys(nv_imap_t *imap, int *num_keys)
{
	std::map<int, void *>::iterator it;
	int i;
	
	*num_keys = imap->map.size();
	if (imap->keys) {
		nv_free(imap->keys);
	}
	imap->keys = nv_alloc_type(int, *num_keys);
	
	for (i = 0, it = imap->map.begin(); it != imap->map.end(); it++, ++i) {
		imap->keys[i] = it->first;
	}
	
	return imap->keys;
}

/* string map */

struct nv_smap {
	std::map<std::string, void *> map;
	char **keys;
};

nv_smap_t *
nv_smap_alloc(void)
{
	nv_smap_t *map = new nv_smap_t;
	map->keys = NULL;
	
	return map;
}

void 
nv_smap_free(nv_smap_t **smap)
{
	if (smap && *smap) {
		int i;
		if ((*smap)->keys) {
			for (i = 0; (*smap)->keys[i] != NULL; ++i) {
				nv_free((*smap)->keys[i]);
			}
			nv_free((*smap)->keys);
			(*smap)->keys = NULL;
		}
		delete *smap;
		*smap = NULL;
	}
}

void 
nv_smap_insert(nv_smap_t *smap, const char *key, void *data)
{
	smap->map.insert(std::make_pair(key, data));
}

void *
nv_smap_find(nv_smap_t *smap, const char *key)
{
	std::map<std::string, void *>::iterator it = smap->map.find(key);
	if (it != smap->map.end()) {
		return it->second;
	}
	return NULL;
}

void 
nv_smap_remove(nv_smap_t *smap, const char *key)
{
	smap->map.erase(key);
}

int 
nv_smap_foreach(nv_smap_t *smap, nv_smap_foreach_func_t func)
{
	std::map<std::string, void *>::iterator it;
	int ret;
	for (it = smap->map.begin(); it != smap->map.end(); it++) {
		if ((ret = func(it->first.c_str(), it->second)) != 0) {
			return ret;
		}
	}
	return 0;
}

void
nv_smap_string_insert(nv_smap_t *smap, const char *key, const char *string_data)
{
	char *new_string = nv_alloc_type(char, strlen(string_data) + 1);
	void *old_data = nv_smap_find(smap, key);
	
	if (old_data != NULL) {
		nv_free(old_data);
	}
	strcpy(new_string, string_data);
	smap->map.insert(std::make_pair(key, new_string));
}

static int
nv_smap_string_free_func(const char *key, void *data)
{
	nv_free(data);
	return 0;
}

void
nv_smap_string_free(nv_smap_t *map)
{
	nv_smap_foreach(map, nv_smap_string_free_func);
}

const char * const *
nv_smap_keys(nv_smap_t *map, int *num_keys)
{
	std::map<std::string, void *>::iterator it;
	int i;
	
	*num_keys = map->map.size();
	if (map->keys) {
		for (i = 0; map->keys[i] != NULL; ++i) {
			nv_free(map->keys[i]);
		}
		nv_free(map->keys);
	}
	map->keys = nv_alloc_type(char *, *num_keys + 1);
	for (i = 0, it = map->map.begin(); it != map->map.end(); it++, ++i) {
		const char *key = it->first.c_str();
		map->keys[i] = nv_alloc_type(char, strlen(key) + 1);
		strcpy(map->keys[i], key);
	}
	map->keys[i] = NULL;
	
	return map->keys;
}
