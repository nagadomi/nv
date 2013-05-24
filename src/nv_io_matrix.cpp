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
#include <string>
#include <sstream>
#include <vector>

#define NV_LINE_SIZE(n) (n * 32)

nv_matrix_t *
nv_load_matrix_fp(FILE *fp)
{
	int fn, m, n, rows, cols, list, i;
	nv_matrix_t *mat;
	int loop16;
	
	fn = fscanf(fp, "%d %d %d %d %d ", &list, &m, &n, &rows, &cols);
	if (fn != 5) {
		return NULL;
	}
	if (m != rows * cols && rows == 1) {
		cols = m;
	}
	mat = nv_matrix3d_list_alloc(n, rows, cols, list);
	if (mat == NULL) {
		return NULL;
	}
	nv_matrix_zero(mat);
	
	loop16 = (n & 0xfffffff0);
	
	for (list = 0; list < mat->list; ++list) {
		for (m = 0; m < mat->m; ++m) {
			for (n = 0; n < loop16; n += 16) {
				i = fscanf(fp, "%E %E %E %E %E %E %E %E %E %E %E %E %E %E %E %E ",
					&NV_MAT_LIST_V(mat, list, m, n + 0),
					&NV_MAT_LIST_V(mat, list, m, n + 1),
					&NV_MAT_LIST_V(mat, list, m, n + 2),
					&NV_MAT_LIST_V(mat, list, m, n + 3),
					&NV_MAT_LIST_V(mat, list, m, n + 4),
					&NV_MAT_LIST_V(mat, list, m, n + 5),
					&NV_MAT_LIST_V(mat, list, m, n + 6),
					&NV_MAT_LIST_V(mat, list, m, n + 7),
					&NV_MAT_LIST_V(mat, list, m, n + 8),
					&NV_MAT_LIST_V(mat, list, m, n + 9),
					&NV_MAT_LIST_V(mat, list, m, n + 10),
					&NV_MAT_LIST_V(mat, list, m, n + 11),
					&NV_MAT_LIST_V(mat, list, m, n + 12),
					&NV_MAT_LIST_V(mat, list, m, n + 13),
					&NV_MAT_LIST_V(mat, list, m, n + 14),
					&NV_MAT_LIST_V(mat, list, m, n + 15)
					);
				if (i != 16) {
					nv_matrix_free(&mat);
					return NULL;
				}
			}
			
			for (n = loop16; n < mat->n; ++n) {
				i = fscanf(fp, "%E ", &NV_MAT_LIST_V(mat, list, m, n));
				if (i == 0) {
					nv_matrix_free(&mat);
					return NULL;
				}
			}
		}
	}
	
	return mat;
}

void
nv_save_matrix_fp(FILE *fp, const nv_matrix_t *mat)
{
	int m, n;
	int loop16 = (mat->n & 0xfffffff0);

	fprintf(fp, "%d %d %d %d %d\n", mat->list, mat->m, mat->n, mat->rows, mat->cols);

	for (m = 0; m < mat->m; ++m) {

		for (n = 0; n < loop16; n += 16) {
			if (n != 0) {
				fprintf(fp, " ");
			}
			fprintf(fp,
				"%E %E %E %E %E %E %E %E %E %E %E %E %E %E %E %E",
				NV_MAT_V(mat, m, n + 0),
				NV_MAT_V(mat, m, n + 1),
				NV_MAT_V(mat, m, n + 2),
				NV_MAT_V(mat, m, n + 3),
				NV_MAT_V(mat, m, n + 4),
				NV_MAT_V(mat, m, n + 5),
				NV_MAT_V(mat, m, n + 6),
				NV_MAT_V(mat, m, n + 7),
				NV_MAT_V(mat, m, n + 8),
				NV_MAT_V(mat, m, n + 9),
				NV_MAT_V(mat, m, n + 10),
				NV_MAT_V(mat, m, n + 11),
				NV_MAT_V(mat, m, n + 12),
				NV_MAT_V(mat, m, n + 13),
				NV_MAT_V(mat, m, n + 14),
				NV_MAT_V(mat, m, n + 15)
			);
		}
		for (n = loop16; n < mat->n; ++n) {
			if (n != 0) {
				fprintf(fp, " ");
			}
			fprintf(fp, "%E", NV_MAT_V(mat, m, n));
		}
		fprintf(fp, "\n");
	}
}

nv_matrix_t *
nv_load_matrix_text(const char *filename)
{
	FILE *fp = fopen(filename, "r");
	nv_matrix_t *mat;

	if (fp == NULL) {
		return NULL;
	}
	mat = nv_load_matrix_fp(fp);
	fclose(fp);

	return mat;
}

int 
nv_save_matrix_text(const char *filename, const nv_matrix_t *mat)
{
	FILE *fp = fopen(filename, "w");
	
	if (fp == NULL) {
		return -1;
	}
	nv_save_matrix_fp(fp, mat);
	fclose(fp);
	
	return 0;
}

char *nv_serialize_matrix(const nv_matrix_t *mat)
{
	int j, i;
	std::ostringstream o;
	std::string str;
	char *s;
	
	o << "< ";
	o << mat->list;
	o << " ";
	o << mat->m;
	o << " ";
	o << mat->n;
	o << " ";
	o << mat->rows;
	o << " ";
	o << mat->cols;
	o << ": ";
	
	for (j = 0; j < mat->m; ++j) {
		for (i = 0; i < mat->n; ++i) {
			o << std::scientific << NV_MAT_V(mat, j, i);
			o << " ";
		}
	}
	o << ">";

	str = o.str();
	s = nv_alloc_type(char , str.size() + 1);
	strcpy(s, str.c_str());
	
	return s;
}

nv_matrix_t *nv_deserialize_matrix(const char *s)
{
	int list, n, m, rows, cols;
	nv_matrix_t *mat;
	std::istringstream i(s);
	char c;
	
	i >> std::skipws >> c;
	if (c != '<') {
		return NULL;
	}
	i >> std::skipws >> list;
	i >> std::skipws >> m;
	i >> std::skipws >> n;
	i >> std::skipws >> rows;
	i >> std::skipws >> cols;

	if (i.eof() || i.fail()) {
		return NULL;
	}
	
	if (m != rows * cols && rows == 1) {
		cols = m;
	}
	mat = nv_matrix3d_list_alloc(n, rows, cols, list);
	
	i >> std::skipws >> c;
	if (c != ':') {
		nv_matrix_free(&mat);
		return NULL;
	}
	
	for (list = 0; list < mat->list; ++list) {
		for (m = 0; m < mat->m; ++m) {
			for (n = 0; n < mat->n; ++n) {
				i >> std::skipws >> std::scientific >> NV_MAT_LIST_V(mat, list, m, n);
				if (i.fail()) {
					nv_matrix_free(&mat);
					return NULL;
				}
			}
		}
	}
	
	i >> std::skipws >> c;
	if (c != '>') {
		nv_matrix_free(&mat);
		return NULL;
	}
	
	return mat;
}

nv_matrix_t *
nv_load_matrix_bin_fp(FILE *fp)
{
	nv_matrix_t hdr, *mat;
	size_t n;
	size_t len;
	
	n = fread(&hdr.list, sizeof(hdr.list), 1, fp);
	if (n != 1) return NULL;
	n = fread(&hdr.n, sizeof(hdr.n), 1, fp);
	if (n != 1) return NULL;
	n = fread(&hdr.m, sizeof(hdr.m), 1, fp);
	if (n != 1) return NULL;
	n = fread(&hdr.rows, sizeof(hdr.rows), 1, fp);
	if (n != 1) return NULL;
	n = fread(&hdr.cols, sizeof(hdr.cols), 1, fp);
	if (n != 1) return NULL;
	
	mat = nv_matrix3d_list_alloc(hdr.n, hdr.rows, hdr.cols, hdr.list);
	nv_matrix_zero(mat);
	
	len = 0;
	while ((n = fread(mat->v + len, sizeof(float), (size_t)mat->list_step * mat->list - len, fp)) > 0) {
		len += n;
	}
	if (len != (size_t)mat->list_step * mat->list) {
		nv_matrix_free(&mat);
		return NULL;
	}
	
	return mat;
}

void
nv_save_matrix_bin_fp(FILE *fp, const nv_matrix_t *mat)
{
	fwrite(&mat->list, sizeof(mat->list), 1, fp);
	fwrite(&mat->n, sizeof(mat->n), 1, fp);
	fwrite(&mat->m, sizeof(mat->m), 1, fp);
	fwrite(&mat->rows, sizeof(mat->rows), 1, fp);
	fwrite(&mat->cols, sizeof(mat->cols), 1, fp);
	fwrite(mat->v, sizeof(float), (size_t)mat->list_step * mat->list, fp);
}

nv_matrix_t *
nv_load_matrix_bin(const char *filename)
{
	FILE *fp = fopen(filename, "rb");
	nv_matrix_t *mat;

	if (fp == NULL) {
		return NULL;
	}
	mat = nv_load_matrix_bin_fp(fp);
	fclose(fp);

	return mat;
}

int 
nv_save_matrix_bin(const char *filename, const nv_matrix_t *mat)
{
	FILE *fp = fopen(filename, "wb");
	
	if (fp == NULL) {
		return -1;
	}
	nv_save_matrix_bin_fp(fp, mat);
	fclose(fp);
	
	return 0;
}

int
nv_load_matrix_array_bin(const char *filename, nv_matrix_t **array, int *len)
{
	FILE *fp;
	int i;
	
	fp = fopen(filename, "rb");
	if (fp == NULL) {
		return -1;
	}
	for (i = 0; i < *len; ++i) {
		nv_matrix_t *mat = nv_load_matrix_bin_fp(fp);
		if (mat) {
			array[i] = mat;
		} else {
			break;
		}
	}
	*len = i;
	fclose(fp);
	
	return 0;
}

int
nv_save_matrix_array_bin(const char *filename, nv_matrix_t **array, int len)
{
	FILE *fp;
	int i;

	fp = fopen(filename, "wb");
	if (fp == NULL) {
		return -1;
	}
	for (i = 0; i < len; ++i) {
		nv_save_matrix_bin_fp(fp, array[i]);
	}
	fclose(fp);
	
	return 0;
}

int
nv_load_matrix_array_text(const char *filename, nv_matrix_t **array, int *len)
{
	FILE *fp;
	int i;
	
	fp = fopen(filename, "rb");
	if (fp == NULL) {
		return -1;
	}
	for (i = 0; i < *len; ++i) {
		nv_matrix_t *mat = nv_load_matrix_fp(fp);
		if (mat) {
			array[i] = mat;
		} else {
			break;
		}
	}
	*len = i;
	fclose(fp);
	
	return 0;
}

int
nv_save_matrix_array_text(const char *filename, nv_matrix_t **array, int len)
{
	FILE *fp = fopen(filename, "wb");
	int i;
	
	if (fp == NULL) {
		return -1;
	}
	for (i = 0; i < len; ++i) {
		nv_save_matrix_fp(fp, array[i]);
	}
	fclose(fp);
	
	return 0;
}
