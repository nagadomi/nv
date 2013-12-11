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

/* tick toc */

#if NV_POSIX
#  include <time.h>
#  include <sys/time.h>
#elif NV_WINDOWS
#  include <windows.h>
#endif

unsigned long 
nv_clock(void)
{
#if NV_POSIX
	unsigned long c;
	struct timeval tv;
	gettimeofday(&tv, NULL);
	c = tv.tv_sec * 1000;
	c += tv.tv_usec / 1000;
	return c;
#elif NV_WINDOWS
	return (unsigned long)GetTickCount();
#endif
}
