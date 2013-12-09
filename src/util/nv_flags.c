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

int main(void)
{
#if NV_WINDOWS
	printf("windows\n");
#elif NV_POSIXZ
	printf("posix\n");
#endif
#if NV_ENABLE_GLIBC_BACKTRACE
	printf("glibc backtrace\n");
#endif
#if NV_ENABLE_OPENSSL
	printf("openssl\n");
#endif
#if NV_WITH_OPENCV
	printf("opencv\n");
#endif
#if NV_WITH_EIIO
	printf("eiio\n");
#endif

#if NV_ENABLE_STRICT
	printf("strict\n");
#endif

#if NV_ENABLE_SSE
	printf("sse\n");
#endif
#if NV_ENABLE_SSE2
	printf("sse2\n");	
#endif
#if NV_ENABLE_SSE3
	printf("sse3\n");		
#endif
#if NV_ENABLE_SSE4_1
	printf("sse4.1\n");		
#endif
#if NV_ENABLE_SSE4_2
	printf("sse4.2\n");			
#endif
#if NV_ENABLE_POPCNT
	printf("popcnt\n");				
#endif
#if NV_ENABLE_AVX
	printf("avx\n");
#endif

	return 0;
}
