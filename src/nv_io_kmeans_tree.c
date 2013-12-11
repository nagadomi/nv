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
nv_save_kmeans_tree_text(const char *filename, const nv_kmeans_tree_t *tree)
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
			nv_save_matrix_fp(fp, tree->mat[y][x]);
		}
	}
	fclose(fp);
	
	return 0;
}

nv_kmeans_tree_t *
nv_load_kmeans_tree_text(const char *filename)
{
	FILE *fp = fopen(filename, "r");
	nv_kmeans_tree_t *tree = nv_alloc_type(nv_kmeans_tree_t, 1);
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
	tree->mat = nv_alloc_type(nv_matrix_t **, tree->height);
	
	for (y = 0; y < tree->height; ++y) {
		n = fscanf(fp, "%d", &tree->dim[y]);
		if (n != 1) {
			nv_free(tree->dim);
			nv_free(tree->node);
			nv_free(tree->mat);
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
			nv_free(tree->mat);
			nv_free(tree);
			fclose(fp);
			return NULL;
		}
	}
	
	for (y = 0; y < tree->height; ++y) {
		tree->mat[y] = nv_alloc_type(nv_matrix_t *, tree->node[y]);
		for (x = 0; x < tree->node[y]; ++x) {
			tree->mat[y][x] = nv_load_matrix_fp(fp);
		}
	}
	
	return tree;
}

int
nv_save_kmeans_tree_bin(const char *filename, const nv_kmeans_tree_t *tree)
{
	FILE *fp = fopen(filename, "wb");
	int x, y;
	
	if (fp == NULL) {
		perror(filename);
		return -1;
	}
	fwrite(&tree->n, sizeof(tree->n), 1, fp);
	fwrite(&tree->m, sizeof(tree->m), 1, fp);
	fwrite(&tree->height, sizeof(tree->height), 1, fp);
	fwrite(tree->dim, sizeof(*tree->dim), tree->height, fp);
	fwrite(tree->node, sizeof(*tree->node), tree->height, fp);
	
	for (y = 0; y < tree->height; ++y) {
		for (x = 0; x < tree->node[y]; ++x) {
			nv_save_matrix_bin_fp(fp, tree->mat[y][x]);
		}
	}
	fclose(fp);
	
	return 0;
}

nv_kmeans_tree_t *
nv_load_kmeans_tree_bin(const char *filename)
{
	FILE *fp = fopen(filename, "rb");
	nv_kmeans_tree_t *tree = nv_alloc_type(nv_kmeans_tree_t, 1);
	int n, y, x;
	
	if (fp == NULL) {
		perror(filename);
		nv_free(tree);
		return NULL;
	}
	n = 0;
	n += fread(&tree->n, sizeof(tree->n), 1, fp);
	n += fread(&tree->m, sizeof(tree->m), 1, fp);
	n += fread(&tree->height, sizeof(tree->height), 1, fp);
	
	if (n != 3) {
		perror(filename);
		nv_free(tree);
		fclose(fp);
		return NULL;
	}
	tree->dim = nv_alloc_type(int, tree->height);
	tree->node = nv_alloc_type(int, tree->height);
	tree->mat = nv_alloc_type(nv_matrix_t **, tree->height);
	n = fread(tree->dim, sizeof(*tree->dim), tree->height, fp);
	if (n != tree->height) {
		nv_free(tree->dim);
		nv_free(tree->node);
		nv_free(tree->mat);
		nv_free(tree);
		fclose(fp);
		return NULL;
	}
	n = fread(tree->node, sizeof(*tree->node), tree->height, fp);
	if (n != tree->height) {
		nv_free(tree->dim);
		nv_free(tree->node);
		nv_free(tree->mat);
		nv_free(tree);
		fclose(fp);
		return NULL;
	}
	
	for (y = 0; y < tree->height; ++y) {
		tree->mat[y] = nv_alloc_type(nv_matrix_t *, tree->node[y]);
		for (x = 0; x < tree->node[y]; ++x) {
			tree->mat[y][x] = nv_load_matrix_bin_fp(fp);
			if (tree->mat[y][x] == NULL) {
				nv_free(tree->dim);
				nv_free(tree->node);
				// TODO: free tree->mat[][]
				nv_free(tree->mat);
				nv_free(tree);
				fclose(fp);
				return NULL;
			}
		}
	}
	fclose(fp);
	
	return tree;
}
