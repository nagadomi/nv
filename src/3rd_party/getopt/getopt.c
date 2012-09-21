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

 /*
 * This is based on public domain implementation of getopt().
 * The original copyright notice follows.
 *
 * getopt.c:
 * a public domain implementation of getopt()
 *
 * The following source code is an adaptation of the public domain getopt()
 * implementation presented at the 1985 UNIFORUM conference in Dallas,
 * Texas. Slight edits have been made to improve readability and the result
 * is released into the public domain like that from which it was derived.
 */
 
#include <stdio.h>
#include <string.h>
#include "nv_util_getopt.h"

int nv_getopt_optind = 1;
int nv_getopt_optopt;
int nv_getopt_opterr = 1;
char *nv_getopt_optarg;

int
nv_getopt(int argc, char **argv, const char *opts)
{
	static int sp = 1;
	int c;
	const char *cp;

	if (sp == 1) {
		
		/* If all args are processed, finish */
		if (nv_getopt_optind >= argc) {
			return EOF;
		}
		if (argv[nv_getopt_optind][0] != '-' || argv[nv_getopt_optind][1] == '\0') {
			return EOF;
		}
		
	} else if (!strcmp(argv[nv_getopt_optind], "--")) {
		
		/* No more options to be processed after this one */
		nv_getopt_optind++;
		return EOF;
		
	}
	
	nv_getopt_optopt = c = argv[nv_getopt_optind][sp];

	/* Check for invalid option */
	if (c == ':' || (cp = strchr(opts, c)) == NULL) {
		if (nv_getopt_opterr != 0) {
			fprintf(stderr,
				"%s: illegal option -- %c\n",
				argv[0],
				c);
		}
		if (argv[nv_getopt_optind][++sp] == '\0') {
			nv_getopt_optind++;
			sp = 1;
		}
		
		return '?';
	}

	/* Does this option require an argument? */
	if (*++cp == ':') {

		/* If so, get argument; if none provided output error */
		if (argv[nv_getopt_optind][sp+1] != '\0') {
			nv_getopt_optarg = &argv[nv_getopt_optind++][sp+1];
		} else if (++nv_getopt_optind >= argc) {
			if (nv_getopt_opterr != 0) {
				fprintf(stderr,
					"%s: option requires an argument -- %c\n",
					argv[0],
					c);
			}
			sp = 1;
			return '?';
		} else {
			nv_getopt_optarg = argv[nv_getopt_optind++];
		}
		sp = 1;

	} else {
		if (argv[nv_getopt_optind][++sp] == '\0') {
			sp = 1;
			nv_getopt_optind++;
		}
		nv_getopt_optarg = NULL;
	}
	
	return c;
}
