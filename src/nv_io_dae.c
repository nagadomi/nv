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
#include "nv_io.h"

int 
nv_save_dae_text(const char *filename, const nv_dae_t *dae)
{
	FILE *fp;

	fp = fopen(filename, "w");
	if (fp == NULL) {
		return -1;
	}
	fprintf(fp, "%d %d %E %E\n",
			dae->input, dae->hidden,
			dae->noise, dae->sparsity);
	nv_save_matrix_fp(fp, dae->input_w);
	nv_save_matrix_fp(fp, dae->input_bias);
	nv_save_matrix_fp(fp, dae->hidden_bias);
	
	fclose(fp);
	
	return 0;
}

nv_dae_t *
nv_load_dae_text(const char *filename)
{
	FILE *fp;
	int c;
	nv_dae_t *dae;

	fp = fopen(filename, "r");
	if (fp == NULL) {
		return NULL;
	}
	dae = nv_alloc_type(nv_dae_t, 1);
	c = fscanf(fp, "%d %d %E %E",
			   &dae->input, &dae->hidden,
			   &dae->noise, &dae->sparsity);
	if (c != 4) {
		nv_free(dae);
		fclose(fp);
		return NULL;
	}
	dae->input_w = nv_load_matrix_fp(fp);
	dae->input_bias = nv_load_matrix_fp(fp);
	dae->hidden_bias = nv_load_matrix_fp(fp);

	return dae;
}
