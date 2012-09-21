/*
 * This file is part of libnv.
 *
 * Copyright (C) 2012 nagadomi@nurs.or.jp
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
nv_save_klr_tree(const char *filename, const nv_klr_tree_t *tree)
{
	FILE *fp = fopen(filename, "w");
	int x, y;
	
	if (fp == NULL) {
		perror(filename);
		return -1;
	}
	fprintf(fp, "%d %d %d\n", tree->n, tree->m, tree->height);
	for (y = 0; y < tree->height; ++y) {
		if (y != 0) {
			fprintf(fp, " ");
		}
		fprintf(fp, "%d", tree->dim[y]);
	}
	fprintf(fp, "\n");
	for (y = 0; y < tree->height; ++y) {
		if (y != 0) {
			fprintf(fp, " ");
		}
		fprintf(fp, "%d", tree->node[y]);
	}
	fprintf(fp, "\n");
	
	for (y = 0; y < tree->height; ++y) {
		for (x = 0; x < tree->node[y]; ++x) {
			nv_save_lr_fp(fp, tree->lr[y][x]);
		}
	}
	fclose(fp);
	
	return 0;
}

nv_klr_tree_t *
nv_load_klr_tree(const char *filename)
{
	FILE *fp = fopen(filename, "r");
	nv_klr_tree_t *tree = nv_alloc_type(nv_klr_tree_t, 1);
	int n, y, x;
	
	if (fp == NULL) {
		perror(filename);
		nv_free(tree);
		return NULL;
	}

	n = fscanf(fp, "%d %d %d", &tree->n, &tree->m, &tree->height);
	if (n != 3) {
		perror(filename);
		nv_free(tree);
		fclose(fp);
		return NULL;
	}
	tree->dim = nv_alloc_type(int, tree->height);
	tree->node = nv_alloc_type(int, tree->height);
	tree->lr = nv_alloc_type(nv_lr_t **, tree->height);
	
	for (y = 0; y < tree->height; ++y) {
		n = fscanf(fp, "%d", &tree->dim[y]);
		if (n != 1) {
			nv_free(tree->dim);
			nv_free(tree->node);
			nv_free(tree->lr);
			nv_free(tree);
			fclose(fp);
			return NULL;
		}
	}
	for (y = 0; y < tree->height; ++y) {
		n = fscanf(fp, "%d", &tree->node[y]);
		if (n != 1) {
			nv_free(tree->dim);
			nv_free(tree->node);
			nv_free(tree->lr);
			nv_free(tree);
			fclose(fp);
			return NULL;
		}
	}
	
	for (y = 0; y < tree->height; ++y) {
		tree->lr[y] = nv_alloc_type(nv_lr_t *, tree->node[y]);
		for (x = 0; x < tree->node[y]; ++x) {
			tree->lr[y][x] = nv_load_lr_fp(fp);
		}
	}
	
	return tree;
}
