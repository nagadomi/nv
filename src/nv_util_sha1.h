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

#ifndef NV_UTIL_SHA1_H
#define NV_UTIL_SHA1_H

#ifdef __cplusplus
extern "C" {
#endif

#define NV_SHA1_BINARY_LEN 20
#define NV_SHA1_HEXSTR_LEN (40 + 1)

void nv_sha1(void *sha1_bin, const void *data, size_t data_len);
int nv_sha1_file(void *sha1_bin, const char *filename);

void nv_sha1_hexstr(char *sha1_hexstr, const void *data, size_t data_len);
int nv_sha1_hexstr_file(char *sha1_hexstr, const char *filename);

#ifdef __cplusplus
}
#endif

#endif
