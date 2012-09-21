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

#if NV_ENABLE_SLEEP

#if NV_POSIX
#  include <unistd.h>
#elif NV_WINDOWS
#  include <windows.h>
#endif

void 
nv_sleep(unsigned int msec)
{
#if NV_POSIX
	usleep(msec * 1000);
#elif NV_WINDOWS
	Sleep(msec);
#endif
}

#endif
