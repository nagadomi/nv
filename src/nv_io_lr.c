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

int
nv_save_lr_fp(FILE *fp, const nv_lr_t *irls)
{
	fprintf(fp, "%d %d\n", irls->k, irls->n);
	nv_save_matrix_fp(fp, irls->w);

	return 0;
}

int
nv_save_lr(const char *file, const nv_lr_t *irls)
{
	FILE *fp = fopen(file, "w");

	if (fp == NULL) {
		perror(file);
		return -1;
	}
	nv_save_lr_fp(fp, irls);
	fclose(fp);

	return 0;
}

nv_lr_t *
nv_load_lr_fp(FILE *fp)
{
	nv_lr_t *irls = (nv_lr_t *)nv_malloc(sizeof(nv_lr_t));
	int n;
	
	n = fscanf(fp, "%d %d", &irls->k, &irls->n);
	if (n != 2) {
		nv_free(irls);
		return NULL;
	}
	irls->w = nv_load_matrix_fp(fp);
	
	NV_ASSERT(irls != NULL);
	NV_ASSERT(irls->k == irls->w->m);
	NV_ASSERT(irls->n == irls->w->n);

	return irls;
}

nv_lr_t *
nv_load_lr(const char *filename)
{
	FILE *fp = fopen(filename, "r");
	nv_lr_t *lr;
	
	if (fp == NULL) {
		perror(filename);
		return NULL;
	}
	lr = nv_load_lr_fp(fp);
	fclose(fp);

	return lr;
}

