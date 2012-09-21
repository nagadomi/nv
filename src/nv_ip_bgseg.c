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

#include "nv_ip.h"
#include "nv_num.h"

/* 参考: http://opencv.jp/sample/accumulation_of_background.html */

nv_bgseg_t *
nv_bgseg_alloc(int frame_rows, int frame_cols,
			   float zeta, float bg_v, float fg_v,
			   int size
	)
{
	nv_bgseg_t *bg = nv_alloc_type(nv_bgseg_t, 1);
	float scale = (float)size / (float)NV_MAX(frame_rows, frame_cols);
	
	bg->init_1st = 0;
	bg->init_2nd = 0;
	bg->init_1st_finished = 0;
	bg->init_2nd_finished = 0;
	bg->frame_rows = frame_rows;
	bg->frame_cols = frame_cols;
	bg->rows = NV_ROUND_INT(frame_rows * scale);
	bg->cols = NV_ROUND_INT(frame_cols * scale);
	
	bg->zeta = zeta;
	bg->bg_v = bg_v;
	bg->fg_v = fg_v;
	bg->size = size;
	
	bg->av = nv_matrix_alloc(1 * bg->rows * bg->cols, 1);
	nv_matrix_zero(bg->av);
	bg->sgm = nv_matrix_dup(bg->av);
	
	return bg;
}

void
nv_bgseg_free(nv_bgseg_t **bg)
{
	if (bg && *bg) {
		nv_matrix_free(&(*bg)->av);
		nv_matrix_free(&(*bg)->sgm);
		nv_free(*bg);
		*bg = NULL;
	}
}

static nv_matrix_t *
conv_image2vec(const nv_bgseg_t *bg,
			   const nv_matrix_t *image)
{
	nv_matrix_t *vec;
	nv_matrix_t *smooth;
	nv_matrix_t *resize = NULL, *gray = NULL;
	int i;
	float scale = (float)bg->size / (float)NV_MAX(image->rows, image->cols);

	if (scale != 1.0f) {
		resize = nv_matrix3d_alloc(image->n,
								   NV_ROUND_INT(image->rows * scale),
								   NV_ROUND_INT(image->cols * scale));
		nv_resize(resize, image);
		image = resize;
	}
	if (image->n != 1) {
		gray = nv_matrix3d_alloc(1, image->rows, image->cols);
		nv_gray(gray, image);
		image = gray;
	}
	vec = nv_matrix_alloc(image->rows * image->cols, 1);
	smooth = nv_matrix_clone(image);
	nv_gaussian5x5(smooth, 0, image, 0);

	for (i = 0; i < image->m; ++i) {
		NV_MAT_V(vec, 0, i) = NV_MAT_V(smooth, i, 0);
	}
	nv_matrix_free(&smooth);
	nv_matrix_free(&gray);
	nv_matrix_free(&resize);
	
	return vec;
}

static void
conv_mask(nv_bgseg_t *bg,
		  nv_matrix_t *mask,
		  const nv_matrix_t *mask_vec)
{
	int i;
	nv_matrix_t *tmp = nv_matrix3d_alloc(1, bg->rows, bg->cols);

	NV_ASSERT(mask->rows == bg->rows && mask->cols == bg->cols);
	
	for (i = 0; i < tmp->m; ++i) {
		NV_MAT_V(tmp, i, 0) = 1.0f - NV_MAT_V(mask_vec, 0, i);
	}
	nv_erode(mask, 0, tmp, 0);
	nv_dilate(tmp, 0, mask, 0);
	nv_dilate(mask, 0, tmp, 0);
	nv_dilate(tmp, 0, mask, 0);
	nv_dilate(mask, 0, tmp, 0);
	nv_dilate(tmp, 0, mask, 0);
	nv_dilate(mask, 0, tmp, 0);

	nv_matrix_free(&tmp);
}

static void
running_avg(nv_matrix_t *acc,
			const nv_matrix_t *image,
			float alpha,
			nv_matrix_t *mask,
			float act
	)
{
	int i;
	float b = (1.0f - alpha);
	for (i = 0; i < mask->n; ++i) {
		if (NV_MAT_V(mask, 0, i) == act) {
			NV_MAT_V(acc, 0, i) = b * NV_MAT_V(acc, 0, i)
				+ alpha * NV_MAT_V(image, 0, i);
		}
	}
}

void
nv_bgseg_init_1st_update(nv_bgseg_t *bg, const nv_matrix_t *frame)
{
	nv_matrix_t *tmp;
	
	NV_ASSERT(frame->n == 1 &&
			  bg->frame_rows == frame->rows &&
			  bg->frame_cols == frame->cols);

	bg->init_1st += 1;
	tmp = conv_image2vec(bg, frame);
	nv_vector_add(bg->av, 0, bg->av, 0, tmp, 0);
	
	nv_matrix_free(&tmp);
}

void
nv_bgseg_init_2nd_update(nv_bgseg_t *bg, const nv_matrix_t *frame)
{
	nv_matrix_t *tmp;

	NV_ASSERT(bg->init_1st_finished == 1 &&
			  frame->n == 1 &&
			  bg->frame_rows == frame->rows &&
			  bg->frame_cols == frame->cols);

	bg->init_2nd += 1;
	tmp = conv_image2vec(bg, frame);
	nv_vector_sub(tmp, 0, tmp, 0, bg->av, 0);
	nv_vector_mul(tmp, 0, tmp, 0, tmp, 0);
	nv_vector_muls(tmp, 0, tmp, 0, 2.0f);
	nv_vector_sqrt(tmp, 0, tmp, 0);
	nv_vector_add(bg->sgm, 0, bg->sgm, 0, tmp, 0);

	nv_matrix_free(&tmp);
}

void
nv_bgseg_init_1st_finish(nv_bgseg_t *bg)
{
	NV_ASSERT(bg->init_1st > 0);
	
	nv_vector_muls(bg->av, 0, bg->av, 0, 1.0f / (float)bg->init_1st);
	bg->init_1st_finished = 1;
}

void
nv_bgseg_init_2nd_finish(nv_bgseg_t *bg)
{
	NV_ASSERT(bg->init_2nd > 0);
	nv_vector_muls(bg->sgm, 0, bg->sgm, 0, 1.0f / (float)bg->init_2nd);
	bg->init_2nd_finished = 1;
}

void
nv_bgseg_update(nv_bgseg_t *bg, nv_matrix_t *mask, const nv_matrix_t *frame)
{
	nv_matrix_t *tmp, *tmp2, *mask_vec, *lower, *upper;
	
	NV_ASSERT(bg->init_2nd_finished == 1);

	tmp = conv_image2vec(bg, frame);
	lower = nv_matrix_clone(tmp);
	upper = nv_matrix_clone(tmp);
	mask_vec = nv_matrix_clone(tmp);
	tmp2 = nv_matrix_dup(tmp);
	
	// range
	nv_vector_sub(lower, 0, bg->av, 0, bg->sgm, 0);
	nv_vector_subs(lower, 0, lower, 0, bg->zeta);
	nv_vector_add(upper, 0, bg->av, 0, bg->sgm, 0);
	nv_vector_adds(upper, 0, upper, 0, bg->zeta);
	nv_vector_in_range10(mask_vec, 0, lower, 0, upper, 0, tmp, 0);
	
	// update amplitude
	nv_vector_sub(tmp, 0, tmp, 0, bg->av, 0);
	nv_vector_mul(tmp,  0, tmp, 0, tmp, 0);
	nv_vector_muls(tmp, 0, tmp, 0, 2.0f);
	nv_vector_sqrt(tmp, 0, tmp, 0);
	
	// update background
	running_avg(bg->av, tmp2, bg->bg_v, mask_vec, 1.0f);
	running_avg(bg->sgm, tmp, bg->bg_v, mask_vec, 1.0f);
	
	// update foreground
	running_avg(bg->sgm, tmp, bg->fg_v, mask_vec, 0.0f);
	
	conv_mask(bg, mask, mask_vec);
	
	nv_matrix_free(&tmp);
	nv_matrix_free(&mask_vec);
	nv_matrix_free(&tmp2);
	nv_matrix_free(&upper);
	nv_matrix_free(&lower);
}
