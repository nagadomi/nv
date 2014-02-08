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
#include "nv_io.h"


int 
nv_save_mlp_text(const char *filename, const nv_mlp_t *mlp)
{
	FILE *fp;

	fp = fopen(filename, "w");
	if (fp == NULL) {
		return -1;
	}
	fprintf(fp, "%d %d %d %E %E\n",
			mlp->input, mlp->hidden, mlp->output,
			mlp->dropout, mlp->noise);
	nv_save_matrix_fp(fp, mlp->input_w);
	nv_save_matrix_fp(fp, mlp->input_bias);
	nv_save_matrix_fp(fp, mlp->hidden_w);
	nv_save_matrix_fp(fp, mlp->hidden_bias);

	fclose(fp);

	return 0;
}

nv_mlp_t *
nv_load_mlp_text(const char *filename)
{
	FILE *fp;
	int c;
	nv_mlp_t *mlp;

	fp = fopen(filename, "r");
	if (fp == NULL) {
		return NULL;
	}
	mlp = nv_alloc_type(nv_mlp_t, 1);
	c = fscanf(fp, "%d %d %d %E %E",
			   &mlp->input, &mlp->hidden, &mlp->output,
			   &mlp->dropout, &mlp->noise);
	if (c != 5) {
		nv_free(mlp);
		fclose(fp);
		return NULL;
	}
	mlp->input_w = nv_load_matrix_fp(fp);
	mlp->input_bias = nv_load_matrix_fp(fp);
	mlp->hidden_w = nv_load_matrix_fp(fp);
	mlp->hidden_bias = nv_load_matrix_fp(fp);

	return mlp;
}
