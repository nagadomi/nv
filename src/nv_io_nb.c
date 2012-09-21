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


void
nv_save_nb(const char *file, const nv_nb_t *nb)
{
	FILE *fp = fopen(file, "w");
	int k;

	if (fp == NULL) {
		perror(file);
		NV_ASSERT(0);
		return;
	}

	fprintf(fp, "%d %d\n", nb->k, nb->n);
	for (k = 0; k < nb->k; ++k) {
		nv_save_cov_fp(fp, nb->kcov[k]);
	}
	nv_save_matrix_fp(fp, nb->pk);

	fclose(fp);
}

nv_nb_t *
nv_load_nb(const char *filename)
{
	FILE *fp = fopen(filename, "r");
	nv_nb_t *nb = (nv_nb_t *)nv_malloc(sizeof(nv_nb_t));
	int k;
	int n;

	if (fp == NULL) {
		perror(filename);
		nv_free(nb);
		return NULL;
	}

	memset(nb, 0, sizeof(*nb));
	n = fscanf(fp, "%d %d", &nb->k, &nb->n);
	if (n != 2) {
		perror(filename);
		nv_free(nb);
		fclose(fp);
		return NULL;
	}
	nb->kcov = (nv_cov_t **)nv_malloc(sizeof(nv_cov_t *) * nb->k);
	for (k = 0; k < nb->k; ++k) {
		nb->kcov[k] = nv_load_cov_fp(fp);
		if (nb->kcov[k] == NULL) {
			perror(filename);
			perror(filename);
			nv_nb_free(&nb);
			fclose(fp);
			return NULL;
		}
	}
	nb->pk = nv_load_matrix_fp(fp);

	NV_ASSERT(nb->pk->n == 1);
	NV_ASSERT(nb->pk->m == nb->k);

	fclose(fp);

	return nb;
}

