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
#include "nv_ip_keypoint.h"
#define _GNU_SOURCE 1
#if NV_POSIX
#  include <fenv.h>
#endif

#if NV_WINDOWS

int
nv_aligned_malloc(void **memptr, size_t alignment, size_t size)
{
#if NV_MINGW  
	void *p = __mingw_aligned_malloc(size, alignment);
#else
	void *p = _aligned_malloc(size, alignment);	
#endif
	
	if (p == NULL) {
		*memptr = NULL;
		return -1;
	}
	*memptr = p;
	
	return 0;
}

#endif

void
nv_setenv(const char *name, const char *value)
{
#if NV_WINDOWS
	SetEnvironmentVariable(name, value);
#else
	setenv(name, value, 1);
#endif
}

void
nv_unsetenv(const char *name)
{
#if NV_WINDOWS
	SetEnvironmentVariable(name, NULL);	
#else
	unsetenv(name);
#endif	
}

const char *
nv_getenv(const char *name)
{
#if NV_WINDOWS
	static char value[8192];
	int n = GetEnvironmentVariable(name, value, sizeof(value)-1);
        if (n == 0) {
		return NULL;
	} else {
		return value;
	}
#else
	return getenv(name);
#endif	

}

char *
nv_strerror_r(int errnum, char *buf, size_t buflen)
{
	char *err = NULL;
	
#  if _OPENMP
#    pragma omp critical (nv_strerror_r)
#  endif
	{
		err = strerror(errnum);
		if (err != NULL) {
			strncpy(buf, err, buflen - 1);
		} else {
			memset(buf, 0, buflen);
		}
	}
	if (err == NULL) {
		return NULL;
	}
	return buf;
}

struct tm * 
nv_gmtime_r(const time_t *clock, struct tm *result) 
{ 
#if NV_MINGW
	struct tm *res;

	if (result == NULL) {
		return NULL;
	}
	
#  if _OPENMP
#    pragma omp critical (nv_gmtime_r)
#  endif
	{
		res = gmtime(clock); 
		if (res != NULL) {
			memmove(result, res, sizeof(*result));
		}
	}
	if (res == NULL) {
		return NULL;
	}
	return result;
#elif NV_MSVC
	gmtime_s(result, clock);
	return result;
#else
	return gmtime_r(clock, result);
#endif	
}

struct tm * 
nv_localtime_r(const time_t *clock, struct tm *result) 
{ 
#if NV_MINGW
	struct tm *res;

	if (result == NULL) {
		return NULL;
	}
	
#  if _OPENMP
#    pragma omp critical (nv_localtime_r)
#  endif
	{
		res = localtime(clock); 
		if (res != NULL) {
			memmove(result, res, sizeof(*result));
		}
	}
	if (res == NULL) {
		return NULL;
	}
	return result;
#elif NV_MSVC
	localtime_s(result, clock);
	return result;
#else
	return localtime_r(clock, result);
#endif	
}

time_t
nv_mkgmtime(struct tm *tm)
{
#if NV_MSVC
	return _mkgmtime(tm);
#else	
	time_t ret;
	const char *tz;
	
#  if _OPENMP
#    pragma omp critical (nv_mkgmtime)
#  endif
	{
		tz = nv_getenv("TZ");
		nv_setenv("TZ", "");
		tzset();
		ret = mktime(tm);
		if (tz) {
			nv_setenv("TZ", tz);
		} else {
			nv_unsetenv("TZ");
		}
		tzset();
	}
	
	return ret;
#endif	
}

int
nv_vsnprintf(char *str, size_t size, const char *format, va_list ap)
{
	int ret = vsnprintf(str, size, format, ap);
	
#if (NV_MINGW || NV_MSVC)
	if (ret < 0) {
		ret = _vscprintf(format, ap);
	}
#endif
	
	return ret;
}

void nv_enable_sse2_math(void)
{
#if NV_ENABLE_SSE2	
#  if (NV_MINGW || NV_MSVC)
/*	_set_SSE2_enable(1); */
#  else
#  endif
#endif
}


static int nv_initialized = 0;
static int nv_cuda_enabled_flag = 1;
static int nv_cuda_available_flag = 0;

int nv_cuda_enabled(void)
{
	return nv_cuda_enabled_flag && nv_cuda_available_flag;
}

void nv_cuda_set(int onoff)
{
	nv_cuda_enabled_flag = onoff;
}

void
nv_initialize(void)
{
	if (nv_initialized == 0) {
		nv_initialized = 1;
		nv_enable_sse2_math();

		/* initialize tinymt32 */
		nv_rand_init();
#if NV_POSIX		
		/* Enable some exceptions.  At startup all exceptions are masked.  */
		/* feenableexcept(FE_INVALID|FE_DIVBYZERO|FE_OVERFLOW); */
#ifndef NDEBUG
		feenableexcept(FE_INVALID|FE_DIVBYZERO);
#endif
#elif NV_WINDOWS
		{
			HMODULE hCudaLib = LoadLibrary("nv_cuda.dll");

			if (hCudaLib != NULL) {
				FARPROC nv_cuda_init = GetProcAddress(hCudaLib, "nv_cuda_init");
				FARPROC nv_cuda_available = GetProcAddress(hCudaLib, "nv_cuda_available");

				if (nv_cuda_init && nv_cuda_available) {
					(*nv_cuda_init)();
					if ((*nv_cuda_available)()) {
						nv_cuda_available_flag = 1;
						nv_keypoint_gpu = ((nv_cuda_keypoint_t)GetProcAddress(hCudaLib, "nv_cuda_keypoint"));
					} else {
					}
				}
			}
		}
#endif
	}
}

#ifdef NV_DLL
#  if NV_GCC
static void
__attribute__ ((constructor)) nv_dll_entry_point(void)
{
	nv_initialize();
}
#  elif NV_MSVC
BOOL WINAPI
DllMain(HINSTANCE hinstDLL,
		DWORD fdwReason,
		LPVOID lpvReserved)
{
	if (fdwReason == DLL_PROCESS_ATTACH) {
		nv_initialize();
	}
	return TRUE;
}
#  endif
#endif

void nv_assert(int flag)
{
#if (NV_GCC && !NV_MINGW)
	
#else
	assert(flag);
#endif
}
