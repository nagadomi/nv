/*
 * This file is part of libnv.
 *
 * Copyright (C) 2014 nagadomi@nurs.or.jp
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
#include "nv_io_libsvm.h"

int
nv_save_libsvm_fp(FILE *fp,
				  const nv_matrix_t *data,
				  const nv_matrix_t *labels)
{
	int j;
	
	NV_ASSERT(labels->m >= data->m);
	for (j = 0; j < data->m; ++j) {
		int i;
		int loop16 = (data->n & 0xfffffff0);
		fprintf(fp, "%d ", NV_MAT_VI(labels, j, 0) + 1);

		for (i = 0; i < loop16; i += 16) {
			if (i != 0) {
				fprintf(fp, " ");
			}
			fprintf(fp,
				"%d:%E %d:%E %d:%E %d:%E %d:%E %d:%E %d:%E %d:%E %d:%E %d:%E %d:%E %d:%E %d:%E %d:%E %d:%E %d:%E",
					i + 1,
					NV_MAT_V(data, j, i + 0),
					i + 2,
					NV_MAT_V(data, j, i + 1),
					i + 3,
					NV_MAT_V(data, j, i + 2),
					i + 4,
					NV_MAT_V(data, j, i + 3),
					i + 5,
					NV_MAT_V(data, j, i + 4),
					i + 6,
					NV_MAT_V(data, j, i + 5),
					i + 7,
					NV_MAT_V(data, j, i + 6),
					i + 8,
					NV_MAT_V(data, j, i + 7),
					i + 9,
					NV_MAT_V(data, j, i + 8),
					i + 10,
					NV_MAT_V(data, j, i + 9),
					i + 11,
					NV_MAT_V(data, j, i + 10),
					i + 12,
					NV_MAT_V(data, j, i + 11),
					i + 13,
					NV_MAT_V(data, j, i + 12),
					i + 14,
					NV_MAT_V(data, j, i + 13),
					i + 15,
					NV_MAT_V(data, j, i + 14),
					i + 16,
					NV_MAT_V(data, j, i + 15));
		}
		for (i = loop16; i < data->n; ++i) {
			if (i != 0) {
				fprintf(fp, " ");
			}
			fprintf(fp, "%d:%E", i + 1, NV_MAT_V(data, j, i));
		}
		fprintf(fp, "\n");
	}
	
	return 0;
}

int
nv_save_libsvm(const char *file,
			   const nv_matrix_t *data,
			   const nv_matrix_t *labels)
{
	FILE *fp = fopen(file, "w");

	NV_ASSERT(labels->m >= data->m);

	if (fp == NULL) {
		perror(file);
		return -1;
	}
	nv_save_libsvm_fp(fp, data, labels);
	fclose(fp);

	return 0;
}
