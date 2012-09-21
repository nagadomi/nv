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
nv_color_bgr2hsv_scalar(nv_matrix_t *hsv, int hsv_j,
						const nv_matrix_t *bgr, int bgr_j)
{
	nv_int_float_t max_v;
	nv_int_float_t min_v;
	float b = NV_MAT_V(bgr, bgr_j, NV_CH_B);
	float g = NV_MAT_V(bgr, bgr_j, NV_CH_G);	
	float r = NV_MAT_V(bgr, bgr_j, NV_CH_R);
	
	NV_ASSERT(hsv->n == 3 && bgr->n == 3);

	if (b > g) {
		if (b > r) {
			max_v.f = b;
			max_v.i = NV_CH_B;
			if (g < r) {
				min_v.f = g;
				min_v.i = NV_CH_G;
			} else {
				min_v.f = r;
				min_v.i = NV_CH_R;
			}
		} else {
			max_v.f = r;
			max_v.i = NV_CH_R;
			min_v.f = g;
			min_v.i = NV_CH_G;
		}
	} else {
		if (g > r) {
			max_v.f = g;
			max_v.i = NV_CH_G;
			
			if (b < r) {
				min_v.f = b;
				min_v.i = NV_CH_B;
			} else {
				min_v.f = r;
				min_v.i = NV_CH_R;
			}
		} else {
			max_v.f = r;
			max_v.i = NV_CH_R;
			min_v.f = b;
			min_v.i = NV_CH_B;
		}
	}
	NV_MAT_V(hsv, hsv_j, NV_CH_V) = max_v.f;

	if (max_v.f - min_v.f > 0.0f) {
		NV_MAT_V(hsv, hsv_j, NV_CH_S) = 255.0f * ((max_v.f - min_v.f) / max_v.f);
		switch (max_v.i) {
		case NV_CH_R:
			NV_MAT_V(hsv, hsv_j, NV_CH_H) = 60.0f *	((g - b) / (max_v.f - min_v.f));
			break;
		case NV_CH_G:
			NV_MAT_V(hsv, hsv_j, NV_CH_H) = 60.0f *	(b - r) / (max_v.f - min_v.f) + 120.0f;
			break;
		case NV_CH_B:
			NV_MAT_V(hsv, hsv_j, NV_CH_H) = 60.0f *	(r - g) / (max_v.f - min_v.f) + 240.0f;
			break;
		default:
			NV_ASSERT(0);
			break;			
		}
		if (NV_MAT_V(hsv, hsv_j, NV_CH_H) < 0.0f) {
			NV_MAT_V(hsv, hsv_j, NV_CH_H) += 360.0f;
		}
	} else {
		NV_MAT_V(hsv, hsv_j, NV_CH_S) = NV_MAT_V(hsv, hsv_j, NV_CH_H) = 0.0f;
	}
}

void 
nv_color_hsv2bgr_scalar(nv_matrix_t *bgr, int bgr_j,
						const nv_matrix_t *hsv, int hsv_j)
{
	NV_ASSERT(hsv->n == 3 && bgr->n == 3);

	if (NV_MAT_V(hsv, hsv_j, NV_CH_S) > 0.0f) {
		float f, p, q, t;
		int h_i;

		h_i = NV_FLOOR_INT(NV_MAT_V(hsv, hsv_j, NV_CH_H) / 60.0f) % 6;
		f = (NV_MAT_V(hsv, hsv_j, NV_CH_H) / 60.0f) - (float)h_i;
		p = NV_MAT_V(hsv, hsv_j, NV_CH_V) *
			(1.0f - (NV_MAT_V(hsv, hsv_j, NV_CH_S) / 255.0f));
		q = NV_MAT_V(hsv, hsv_j, NV_CH_V) * 
			(1.0f - (NV_MAT_V(hsv, hsv_j, NV_CH_S) / 255.0f * f));
		t = NV_MAT_V(hsv, hsv_j, NV_CH_V) *
			(1.0f - (NV_MAT_V(hsv, hsv_j, NV_CH_S) / 255.0f * (1.0f - f)));
		
		switch (h_i) {
		case 0:
			NV_MAT_V(bgr, bgr_j, NV_CH_B) =	p;
			NV_MAT_V(bgr, bgr_j, NV_CH_G) = t;
			NV_MAT_V(bgr, bgr_j, NV_CH_R) = NV_MAT_V(hsv, hsv_j, NV_CH_V);
			break;			
		case 1:
			NV_MAT_V(bgr, bgr_j, NV_CH_B) = p;
			NV_MAT_V(bgr, bgr_j, NV_CH_G) = NV_MAT_V(hsv, hsv_j, NV_CH_V);
			NV_MAT_V(bgr, bgr_j, NV_CH_R) = q;
			break;			
		case 2:
			NV_MAT_V(bgr, bgr_j, NV_CH_B) =	t;			
			NV_MAT_V(bgr, bgr_j, NV_CH_G) = NV_MAT_V(hsv, hsv_j, NV_CH_V);
			NV_MAT_V(bgr, bgr_j, NV_CH_R) = p;
			break;
		case 3:
			NV_MAT_V(bgr, bgr_j, NV_CH_B) =	NV_MAT_V(hsv, hsv_j, NV_CH_V);;
			NV_MAT_V(bgr, bgr_j, NV_CH_G) = q;
			NV_MAT_V(bgr, bgr_j, NV_CH_R) = p;
			break;			
		case 4:
			NV_MAT_V(bgr, bgr_j, NV_CH_B) = NV_MAT_V(hsv, hsv_j, NV_CH_V);
			NV_MAT_V(bgr, bgr_j, NV_CH_G) = q;
			NV_MAT_V(bgr, bgr_j, NV_CH_R) = p;
			break;			
		case 5:
			NV_MAT_V(bgr, bgr_j, NV_CH_B) =	q;			
			NV_MAT_V(bgr, bgr_j, NV_CH_G) = p;
			NV_MAT_V(bgr, bgr_j, NV_CH_R) = NV_MAT_V(hsv, hsv_j, NV_CH_V);
			break;
		default:
			NV_ASSERT(0);
			break;
		}
	} else {
		NV_MAT_V(bgr, bgr_j, NV_CH_R) =
			NV_MAT_V(bgr, bgr_j, NV_CH_G) =
			NV_MAT_V(bgr, bgr_j, NV_CH_B) = NV_MAT_V(hsv, hsv_j, NV_CH_V);
	}
}

void
nv_color_bgr2hsv(nv_matrix_t *hsv, const nv_matrix_t *bgr)
{
	int j;
	
	NV_ASSERT(hsv->m >= bgr->m);
	NV_ASSERT(hsv->n == 3 && bgr->n == 3);
	NV_ASSERT(hsv->v != bgr->v);

	for (j = 0; j < bgr->m; ++j) {
		nv_color_bgr2hsv_scalar(hsv, j, bgr, j);
	}
}

void
nv_color_hsv2bgr(nv_matrix_t *bgr, const nv_matrix_t *hsv)
{
	int j;
	
	NV_ASSERT(hsv->m >= bgr->m);	
	NV_ASSERT(hsv->n == 3 && bgr->n == 3);
	NV_ASSERT(hsv->v != bgr->v);
	
	for (j = 0; j < hsv->m; ++j) {
		nv_color_hsv2bgr_scalar(bgr, j, hsv, j);
	}
}
