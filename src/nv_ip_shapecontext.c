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
#include "nv_ip_shapecontext.h"

/* テキトウに作った後使っていないので使うならいろいろ直す必要がある  */
#if 0

#define NV_SC_TAN_RATE 0.625f

static void nv_shapecontext_edge_laplacian(nv_matrix_t *edge, const nv_matrix_t *img)
{
	nv_laplacian1(edge, img, 2.0f);
}

static void nv_shapecontext_edge_image(nv_matrix_t *edge, const nv_matrix_t *img)
{
	nv_shapecontext_edge_laplacian(edge, img);
}
#if 0
static float tangent_angle(float r, 
						   float y, float x,
						   const nv_matrix_t *points, int pc)
{
	int i, dot_m = 0;
	float tan_angle;
	int info;
#if 0
	nv_matrix_t *dot = nv_matrix_alloc(2, (int)(2.0f * r * 2.0f *r));
#else
	nv_matrix_t *dot = nv_matrix_alloc(2, 10 * 8);
#endif
	nv_matrix_t *test_val = nv_matrix_alloc(2, 1);
	nv_matrix_t *test_ret = nv_matrix_alloc(2, 1);
	nv_cov_t *cov = nv_cov_alloc(2);

	NV_MAT_V(dot, 0, 0) = 0.0f;
	NV_MAT_V(dot, 0, 1) = 0.0f;
	++dot_m;

	for (i = 0; i < pc; ++i) {
		float dy = NV_MAT_V(points, i, 0) - y;
		float dx = NV_MAT_V(points, i, 1) - x;
#if 0
		if ((dx * dx + dy * dy) <= r * r) {
#else
		if ((dx * dx + dy * dy) <= 4.0f * 4.0f) {
#endif
			NV_MAT_V(dot, dot_m, 0) = dy;
			NV_MAT_V(dot, dot_m, 1) = dx;
			++dot_m;
		}
	}
	nv_matrix_m(dot, dot_m);
	nv_cov(cov->cov, cov->u, cov->s, dot);
	info = nv_eigen_sym(cov->eigen_vec, cov->eigen_val, cov->cov, 50);
	NV_MAT_V(test_val, 0, 0) = 0.0f;
	NV_MAT_V(test_val, 0, 1) = 1.0f;
	nv_gemv(test_ret, 0, NV_MAT_TR, cov->eigen_vec, test_val, 0);

	y = NV_MAT_V(test_ret, 0, 0);
	x = NV_MAT_V(test_ret, 0, 1);
/*
	if (x < 0.0f) {
		x *= -1.0f;
		y *= -1.0f;
	}
*/
	tan_angle = atan2f(x, y);
/*
	if (tan_angle < 0.0f) {
		tan_angle = 2.0f * NV_PI + tan_angle;
	}
*/
	nv_matrix_free(&dot);
	nv_matrix_free(&test_val);
	nv_matrix_free(&test_ret);
	nv_cov_free(&cov);

	return tan_angle;
}
#endif

nv_shapecontext_t *nv_shapecontext_alloc(int n)
{
	nv_shapecontext_t *sctx = (nv_shapecontext_t *)nv_malloc(sizeof(nv_shapecontext_t));
	sctx->sctx = nv_matrix3d_list_alloc(1, NV_SC_LOG_R_BIN, NV_SC_THETA_BIN, n);
	sctx->tan_angle = nv_matrix_alloc(1, n);
	sctx->coodinate = nv_matrix_alloc(2, n);
	sctx->radius = nv_matrix_alloc(1, n);

	return sctx;
}

void nv_shapecontext_free(nv_shapecontext_t **sctx)
{
	if (*sctx) {
		nv_matrix_free(&(*sctx)->sctx);
		nv_matrix_free(&(*sctx)->tan_angle);
		nv_matrix_free(&(*sctx)->coodinate);
		nv_matrix_free(&(*sctx)->radius);
		nv_free(*sctx);
		*sctx = NULL;
	}
}


void nv_shapecontext_feature(nv_shapecontext_t *sctx,
							const nv_matrix_t *img,
							float r
)
{
	int m, row, col, pc, i, l;
	nv_matrix_t *edge = nv_matrix3d_alloc(1, img->rows, img->cols);
	nv_matrix_t *points = nv_matrix_alloc(2, img->m);
	int *rand_idx = (int *)nv_malloc(sizeof(int) * img->m);
	float u_x, u_y, p_x, p_y, r_e;
	int pn;

	// 細線化
	nv_matrix_zero(points);
	nv_shapecontext_edge_image(edge, img);
	pc = 0;
	u_x = 0.0f;
	u_y = 0.0f;
	for (row = 0; row < edge->rows; ++row) {
		for (col = 0; col < edge->cols; ++col) {
			if (NV_MAT3D_V(edge, row, col, 0) > 50.0f) {
				NV_MAT_V(points, pc, 0) = (float)row;
				NV_MAT_V(points, pc, 1) = (float)col;
				++pc;
				u_y += (float)row;
				u_x += (float)col;
			}
		}
	}
	u_x /= pc;
	u_y /= pc;
	// 指定数の特徴にする（ランダム）
	pn = NV_MIN(pc, sctx->sctx->list);
	nv_shuffle_index(rand_idx, 0, pc);
#if 1
	{
		float max_x, max_y;

		if (pc < sctx->sctx->list) {
			// 足りないときはランダムに増やす
			for (i = pc; i < sctx->sctx->list; ++i) {
				rand_idx[i] = (int)(nv_rand() * pn);
			}
		}
		pc = pn = sctx->sctx->list;

		// 半径を求める

		max_x = 0.0f;
		max_y = 0.0f;
		for (m = 0; m < pn; ++m) {
			float yd = fabsf(NV_MAT_V(points, rand_idx[m], 0) - u_y);
			float xd = fabsf(NV_MAT_V(points, rand_idx[m], 1) - u_x);
			max_x = NV_MAX(max_x, xd);
			max_y = NV_MAX(max_y, yd);
		}
		r = (float)img->rows/2.0f;//NV_MAX(max_x, max_y) * 1.0f;
	}
#endif

	// log(r) = 5の基底定数を求める
	r_e = powf(r, 1.0f / NV_SC_LOG_R_BIN);

	// histgramを計算する
	sctx->n = pn;
	nv_matrix_zero(sctx->sctx);
	nv_matrix_zero(sctx->tan_angle);

	for (l = 0; l < pn; ++l) {
		// tangent angle
#if 0
		float max_bin = 0.0f, min_bin = FLT_MAX;
		float tan_angle = tangent_angle(
			r,
			NV_MAT_V(points, rand_idx[l], 0),
			NV_MAT_V(points, rand_idx[l], 1),
			points, pc);
#else
		float tan_angle = 0.0f;
#endif
		p_y = NV_MAT_V(points, rand_idx[l], 0);
		p_x = NV_MAT_V(points, rand_idx[l], 1);
		NV_MAT_V(sctx->tan_angle, l, 0) = tan_angle;
		NV_MAT_V(sctx->coodinate, l, 0) = p_y;
		NV_MAT_V(sctx->coodinate, l, 1) = p_x;
		NV_MAT_V(sctx->radius, l, 0) = r;

		// shape context
		for (i = 0; i < pn; ++i) {
			// # i ≠ l判定はとりあえずしない
			float xd = NV_MAT_V(points, rand_idx[i], 1) - p_x;
			float yd = NV_MAT_V(points, rand_idx[i], 0) - p_y;
			//int row = i / img->rows;
			//int col = i % img->rows;
			//float xd = col - p_x;
			//float yd = row - p_y;
			float theta;
			float log_r = logf(sqrtf(xd * xd + yd * yd)) / logf(r_e);
			float atan_r = atan2f(xd, yd);

			//if (NV_MAT3D_V(img, row, col, 0) == 0.0f) {
			//	continue;
			//}
			if (i == l) {
				continue;
			}

			if (atan_r < 0.0f) {
				atan_r = 2.0f * NV_PI + atan_r;
			}
			if (tan_angle > 0.0f) {
				if (atan_r + tan_angle > 2.0f * NV_PI) {
					atan_r = atan_r + tan_angle - 2.0f * NV_PI;
				} else {
					atan_r += tan_angle;
				}
			} else {
				if (atan_r + tan_angle < 0.0f) {
					atan_r = 2.0f * NV_PI + (atan_r + tan_angle);
				} else {
					atan_r += tan_angle;
				}
			}

			theta = atan_r / (2.0f * NV_PI / NV_SC_THETA_BIN);
			if (theta < NV_SC_THETA_BIN && log_r < NV_SC_LOG_R_BIN) {
				NV_MAT3D_LIST_V(sctx->sctx, l, (int)log_r, (int)theta, 0) += 1.0f;
			}
		}
#if 0
		for (row = 0; row < NV_SC_LOG_R_BIN; ++row) {
			for (col = 0; col < NV_SC_THETA_BIN; ++col) {
				max_bin = NV_MAX(max_bin, NV_MAT3D_LIST_V(sctx->sctx, l, row, col, 0));
				min_bin = NV_MIN(min_bin, NV_MAT3D_LIST_V(sctx->sctx, l, row, col, 0));
			}
		}
		if (max_bin > 0.0f) {
			for (row = 0; row < NV_SC_LOG_R_BIN; ++row) {
				for (col = 0; col < NV_SC_THETA_BIN; ++col) {
					NV_MAT3D_LIST_V(sctx->sctx, l, row, col, 0) 
						= (NV_MAT3D_LIST_V(sctx->sctx, l, row, col, 0) - min_bin) / (max_bin - min_bin);
				}
			}
		}
#endif
	}
	nv_matrix_free(&edge);
	nv_matrix_free(&points);
	nv_free(rand_idx);
}

#if 0
static float cos_distance(const nv_matrix_t *sctx1,
						  int l1,
						  const nv_matrix_t *sctx2,
						  int l2)
{
	float dotproduct = 0.0f;
	float cosdist = 0.0f;
	float norm1 = 0.0f;
	float norm2 = 0.0f;
	int logr, theta;


	for (logr = 0; logr < NV_SC_LOG_R_BIN; ++logr) {
		for (theta = 0; theta < NV_SC_THETA_BIN; ++theta) {
			dotproduct += NV_MAT3D_LIST_V(sctx1, l1, logr, theta, 0) * NV_MAT3D_LIST_V(sctx2, l2, logr, theta, 0);
			norm1 += NV_MAT3D_LIST_V(sctx1, l1, logr, theta, 0) * NV_MAT3D_LIST_V(sctx1, l1, logr, theta, 0);
			norm2 += NV_MAT3D_LIST_V(sctx2, l2, logr, theta, 0) * NV_MAT3D_LIST_V(sctx2, l2, logr, theta, 0);
		}
	}
	if (norm1 > 0.0f && norm2 > 0.0f) {
		cosdist = 1.0f - dotproduct / (sqrtf(norm1) * sqrtf(norm2));
	}

	return cosdist;
}
#endif

static float x2_test(const nv_matrix_t *sctx1,
					 int l1,
					 const nv_matrix_t *sctx2,
					 int l2)
{
	float x2 = 0.0f;
	int logr, theta;

	for (logr = 0; logr < NV_SC_LOG_R_BIN; ++logr) {
		for (theta = 0; theta < NV_SC_THETA_BIN; ++theta) {
			float x = NV_MAT3D_LIST_V(sctx1, l1, logr, theta, 0) - NV_MAT3D_LIST_V(sctx2, l2, logr, theta, 0);
			float m = NV_MAT3D_LIST_V(sctx1, l1, logr, theta, 0) + NV_MAT3D_LIST_V(sctx2, l2, logr, theta, 0);
			if (m != 0.0f) {
				x2 += (x * x)/ m;
			}
		}
	}

	return 0.5f * x2;
}

float nv_shapecontext_distance(const nv_shapecontext_t *sctx1,
							   const nv_shapecontext_t *sctx2)
{
	float distance = 0.0f;
	int points = NV_MIN(sctx1->n, sctx2->n);
	int m, n;
	nv_matrix_t *cost_matrix = nv_matrix_alloc(points, points);
	nv_matrix_t *mincost = nv_matrix_alloc(points, 1);

#ifdef _DEBUG
	FILE *f1 = fopen("1.dat", "w");
	FILE *f2 = fopen("2.dat", "w");
	FILE *fd = fopen("d.dat", "w");

	if (sctx1->n != points) {
		const nv_shapecontext_t *t1 = sctx1;
		sctx1 = sctx2;
		sctx2 = t1;
	}
#endif

	// cosine distance
	nv_matrix_zero(cost_matrix);
	for (m = 0; m < points; ++m) {
		for (n = 0; n < points; ++n) {
			float cosdist = x2_test(sctx1->sctx, m, sctx2->sctx, n);//cos_distance(sctx1->sctx, m, sctx2->sctx, n);
			float dy = NV_MAT_V(sctx1->coodinate, m, 0) - NV_MAT_V(sctx2->coodinate, n, 0);
			float dx = NV_MAT_V(sctx1->coodinate, m, 1) - NV_MAT_V(sctx2->coodinate, n, 1);
			float rx2 = (NV_MAT_V(sctx1->radius, m, 0) + NV_MAT_V(sctx2->radius, n, 0));
			float eudist = sqrtf(dy * dy + dx * dx)/sqrtf(rx2*rx2);
			float v = 1.0f * eudist + 0.9f * cosdist;
			NV_MAT_V(cost_matrix, m, n) = v;
		}
	}
	distance += nv_munkres(mincost, cost_matrix) / points;

#ifdef _DEBUG
	for (m = 0; m < sctx1->n; ++m) {
		fprintf(f1, "%f %f\n", 
			NV_MAT_V(sctx1->coodinate, m, 1),
			NV_MAT_V(sctx1->coodinate, m, 0));
	}
	for (n = 0; n < sctx2->n; ++n) {
		fprintf(f2, "%f %f\n", 
			NV_MAT_V(sctx2->coodinate, n, 1),
			NV_MAT_V(sctx2->coodinate, n, 0));
	}

	for (n = 0; n < sctx2->n;++n) {
		fprintf(fd, "%f %f\n", 
			NV_MAT_V(sctx2->coodinate, n, 1),
			NV_MAT_V(sctx2->coodinate, n, 0));

		fprintf(fd, "%f %f\n", 
			NV_MAT_V(sctx1->coodinate,(int)NV_MAT_V(mincost, 0, n), 1),
			NV_MAT_V(sctx1->coodinate,(int)NV_MAT_V(mincost, 0, n), 0));
		fprintf(fd, "\n\n");
	}
	fclose(f1);
	fclose(f2);
	fclose(fd);
#endif

	nv_matrix_free(&cost_matrix);
	nv_matrix_free(&mincost);

	return distance;
}

#endif

