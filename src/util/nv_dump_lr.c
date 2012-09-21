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
#include "nv_ml.h"
#include "nv_io.h"

static void
print_usage(void)
{
	puts(
		"nv_dump_lr [OPTIONS] lr_file\n"
		"    -h       display this help and exit.\n"
		"    -n name  variable name.\n"
		"    -s       static flag.\n");
}

int
main(int argc, char **argv)
{
	const char *filename = NULL;
	const char *var_name = NULL;
	int var_static = 0;
	int opt;
	nv_lr_t *lr;
	
	while ((opt = nv_getopt(argc, argv, "hn:s")) != -1){
		switch (opt) {
		case 'h':
			print_usage();
			return 0;
		case 'n':
			var_name = nv_getopt_optarg;
			break;
		case 's':
			var_static = 1;
			break;
		default:
			print_usage();
			return 0;
		}
	}
	argc -= nv_getopt_optind;
	argv += nv_getopt_optind;
	if (argc != 1) {
		print_usage();
		return -1;
	}
	filename = argv[0];
	if (var_name == NULL) {
		var_name = "VAR1";
	}

	lr = nv_load_lr(filename);
	if (lr == NULL) {
		fprintf(stderr, "error : %s\n", filename);
		return -1;
	}
	nv_lr_dump_c(stdout, lr, var_name, var_static);

	nv_lr_free(&lr);
	
	return 0;
}
