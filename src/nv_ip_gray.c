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
#include "nv_ip.h"
#include "nv_num.h"

void 
nv_gray(nv_matrix_t *gray, const nv_matrix_t *bgr)
{
	int m;

	NV_ASSERT(bgr->n == 3 && gray->n == 1 && gray->m == bgr->m);

#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (m = 0; m < gray->m; ++m) {
		// bgr->v 0-255
		NV_MAT_V(gray, m, 0) =
			NV_GRAY_B_RATE * NV_MAT_V(bgr, m, NV_CH_B)
			+ NV_GRAY_G_RATE * NV_MAT_V(bgr, m, NV_CH_G)
			+ NV_GRAY_R_RATE * NV_MAT_V(bgr, m, NV_CH_R);
	}
}

void 
nv_histgram_equalization(nv_matrix_t *eq, const nv_matrix_t *img, int channel)
{
	float freq[256] = {0};
	float fm;
	int m, i;
	float min_freq = FLT_MAX;

	NV_ASSERT(eq->m == img->m);
	if (img->m == 0) {
		nv_matrix_zero(eq);
		return ;
	}

	// freq
	fm = 1.0f / (float )img->m;
	for (m = 0; m < img->m; ++m) {
		int idx = (int)NV_MAT_V(img, m, channel);
		freq[idx] += 1.0f;
	}
	for (i = 1; i < 256; ++i) {
		freq[i] = freq[i] + freq[i - 1];
	}
	for (i = 0; i < 256; ++i) {
		freq[i] *= fm;
		
		if (freq[i] < min_freq) {
			min_freq = freq[i];
		}
	}
	if (min_freq == 1.0) {
		min_freq = 0.999999f;
	}

	// equalization
	for (m = 0; m < img->m; ++m) {
		int idx = (int)NV_MAT_V(img, m, channel);
		float v = (freq[idx] - min_freq) * 255.0f / (1.0f - min_freq);//255.0f * freq[idx];
		v = NV_MIN(NV_MAX(v, 0.0f), 255.0f);
		NV_MAT_V(eq, m, channel) = v;
	}
}

void 
nv_contrast(nv_matrix_t *dest, int dch,
			const nv_matrix_t *src, int sch,
			float angle)// 0.0f-90.0f
{
	int i, j;
	float deg_scale = NV_PI / 180.0f;
	nv_matrix_t *conv = nv_matrix_alloc(256, 1);

	NV_ASSERT(dest->m == src->m);
	NV_ASSERT(dch <= dest->n);
	NV_ASSERT(sch <= src->n);

	for (i = 0; i < conv->n; ++i) {
		NV_MAT_V(conv, 0, i) = tanf(angle * deg_scale) * ((float)i - 127.0f) + 127.0f;
		if (NV_MAT_V(conv, 0, i) > 255.0f) {
			NV_MAT_V(conv, 0, i) = 255.0f;
		}
		if (NV_MAT_V(conv, 0, i) < 0.0f) {
			NV_MAT_V(conv, 0, i) = 0.0f;
		}
	}
	for (j = 0; j < src->m; ++j) {
		NV_ASSERT(NV_MAT_V(src, j, sch) < 256.0f);
		NV_ASSERT(NV_MAT_V(src, j, sch) > 0.0f);

		NV_MAT_V(dest, j, dch) = NV_MAT_V(conv, 0, (int)NV_MAT_V(src, j, sch));
	}
}

void nv_contrast_sigmoid(nv_matrix_t *dest, int dch,
						 const nv_matrix_t *src, int sch,
						 float gain) // 0.1f-1.0f
{
	int i, j;
	nv_matrix_t *conv = nv_matrix_alloc(256, 1);
	float max_v = -FLT_MAX;
	float min_v = FLT_MAX;

	NV_ASSERT(dest->m == src->m);
	NV_ASSERT(dch <= dest->n);
	NV_ASSERT(sch <= src->n);

	for (j = 0; j < src->m; ++j) {
		if (max_v < NV_MAT_V(src, j, sch)) {
			max_v = NV_MAT_V(src, j, sch);
		}
		if (min_v > NV_MAT_V(src, j, sch)) {
			min_v = NV_MAT_V(src, j, sch);
		}
	}

	for (i = 0; i < conv->n; ++i) {
		float x = 255.0f * ((float)i - min_v) / max_v;
		NV_MAT_V(conv, 0, i) = 255.0f * (1.0f / (1.0f + expf(-gain * (x - 128.0f) * 0.078125f)));
	}
	for (j = 0; j < src->m; ++j) {
		NV_ASSERT(NV_MAT_V(src, j, sch) < 256.0f);
		NV_ASSERT(NV_MAT_V(src, j, sch) > 0.0f);
		NV_MAT_V(dest, j, dch) = NV_MAT_V(conv, 0, (int)NV_MAT_V(src, j, sch));
	}
}
