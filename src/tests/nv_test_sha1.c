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

#undef NDEBUG
#include "nv_core.h"
#include "nv_test.h"

void nv_test_sha1(void)
{
	char hex[NV_SHA1_HEXSTR_LEN];
	long t;
	int i;
	
	nv_sha1_hexstr_file(hex, NV_TEST_IMG);
	printf("%s\n", hex);
	NV_ASSERT(strcmp(hex, "3444d453367af67e18dd20f99cdb4d90397a1fa9") == 0);

	t = nv_clock();
	for (i = 0; i < 1000; ++i) {
		nv_sha1_hexstr_file(hex, NV_TEST_IMG);
	}
	printf("sha1 x 1000 %ldms\n", nv_clock() -t);
	fflush(stdout);
}
