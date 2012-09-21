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

#include "nv_core.h"
#if NV_ENABLE_OPENSSL
#  include <openssl/sha.h>
#else
#  include "sha.h"
#endif

#define NV_SHA1_BUFSIZ 1048576

static void 
nv_sha1_bin2hexstr(char *hexstr, const char *bin, size_t bin_len)
{
	size_t i;
	
	for (i = 0; i < bin_len; ++i) {
		sprintf(&hexstr[i * 2], "%02x", ((int)bin[i] & 0xff));
	}
	hexstr[bin_len * 2] = '\0';
}

void 
nv_sha1(void *sha1_bin, const void *data, size_t data_len)
{
#if NV_ENABLE_OPENSSL
	SHA1(data, data_len, sha1_bin);
#else
	SHA_CTX ctx;
	
	sha1_init(&ctx);
	sha1_update(&ctx, (sha1_byte *)data, data_len);
	sha1_final((sha1_byte *)sha1_bin, &ctx);
#endif	
}

int 
nv_sha1_file(void *sha1_bin, const char *filename)
{
#if NV_ENABLE_OPENSSL
	SHA_CTX ctx;
	FILE *fp;
	char *buff = NULL;
	size_t len;
	
	fp = fopen(filename, "rb");
	if (fp == NULL) {
		return -1;
	}
	
	SHA1_Init(&ctx);
	buff =  nv_alloc_type(char, NV_SHA1_BUFSIZ);
	while ((len = fread(buff, sizeof(char), NV_SHA1_BUFSIZ, fp)) > 0) {
		SHA1_Update(&ctx, buff, len);
	}
	fclose(fp);
	SHA1_Final(sha1_bin, &ctx);
	nv_free(buff);
	
	return 0;
#else
	SHA_CTX ctx;
	FILE *fp;
	char *buff = NULL;
	size_t len;
	
	fp = fopen(filename, "rb");
	if (fp == NULL) {
		return -1;
	}
	
	sha1_init(&ctx);
	buff =  nv_alloc_type(char, NV_SHA1_BUFSIZ);
	while ((len = fread(buff, sizeof(char), NV_SHA1_BUFSIZ, fp)) > 0) {
		sha1_update(&ctx, (const sha1_byte *)buff, len);
	}
	fclose(fp);
	sha1_final((sha1_byte *)sha1_bin, &ctx);
	nv_free(buff);
	
	return 0;
#endif	
}

void 
nv_sha1_hexstr(char *sha1_hexstr, const void *data, size_t data_len)
{
	char sha1_bin[NV_SHA1_BINARY_LEN];
	
	nv_sha1(sha1_bin, data, data_len);
	nv_sha1_bin2hexstr(sha1_hexstr, sha1_bin, NV_SHA1_BINARY_LEN);
}

int 
nv_sha1_hexstr_file(char *sha1_hexstr, const char *filename)
{
	char sha1_bin[NV_SHA1_BINARY_LEN];
	int ret;
	
	ret = nv_sha1_file(sha1_bin, filename);
	if (ret != 0) {
		return ret;
	}
	
	nv_sha1_bin2hexstr(sha1_hexstr, sha1_bin, NV_SHA1_BINARY_LEN);
	
	return 0;
}

