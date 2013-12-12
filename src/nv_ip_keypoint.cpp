/*
 * This file is part of libnv.
 *
 * Copyright (C) 2010-2012 nagadomi@nurs.or.jp
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

/* ちょっと古いバージョンだけど説明: http://d.hatena.ne.jp/ultraist/20100313/1268466320  */

#include "nv_core.h"
#include "nv_num.h"
#include "nv_ip.h"
#include "nv_ml.h"
#include <limits.h>

#define NV_KEYPOINT_STAR_R(r) NV_ROUND_INT(r / 6.0f)

#define NV_KEYPOINT_NOSET (FLT_MAX)
#define NV_KEYPOINT_MIN_POINT_R  6       /* 検出される最小の半径(だった値...) */
#define NV_KEYPOINT_SCALE_FACTOR 1.090508f/* 2^(1/8) 8ステップして2倍になる */
#define NV_KEYPOINT_DESC_SCALE   2.0f     /* 記述子を計算する際の半径のスケール。
											 変えるな危険。 */

/* オリエンテーションの量子化数 */
#define NV_KEYPOINT_ORIENTATION_HIST 72

/* 勾配のサンプル数。
   多いほうが特徴量が安定するが遅くなる。
   (6 * 3 + 1) : 精度重視
   (6 * 2 + 1) : 普通
   (6 * 1 + 1) : 速度重視
 */
#define NV_KEYPOINT_HIST_SAMPLE (6 * 3 + 1)

#define NV_KEYPOINT_MULTI_ORIENTATION 1

nv_cuda_keypoint_t nv_keypoint_gpu = NULL;

struct nv_keypoint_ctx {
	nv_keypoint_param_t param;
	int dim;
	nv_matrix_t *outer_r;
	nv_matrix_t *inner_r;
	nv_matrix_t *gauss_w;
};

// based on Cecil H. Hastings approximation atan2
static inline float
approximate_atan2f(float y, float x)
{
	if (x == 0.0f) {
		if (y > 0.0f) {
			return NV_PI;
		} else if (y == 0.0f) {
			return 0.0f;
		}
		return -NV_PI_DIV2;
	}
	const float z = y / x;
	const float z2 = z * z;
	if (z2 < 1.0f) {
		float theta = z / (1.0f + 0.28f * z2);
		if (x < 0.0f) {
			if (y < 0.0f) {
				return theta - NV_PI;
			}
			return theta + NV_PI;
		}
		return theta;
	} else {
		float theta = NV_PI_DIV2 - z / (0.28f + z2);
		if (y < 0.0f) {
			return theta - NV_PI;
		}
		return theta;
	}
}
typedef struct
{
	inline float operator()(float y, float x) const
	{
		return atan2f(y, x);
	}
} native_atan2_t;

typedef struct
{
	inline float operator()(float y, float x) const
	{
		return approximate_atan2f(y, x);
	}
} approximate_atan2_t;

/*
 * スケール空間のフィルタサイズを計算する.
 *
 * 外側の半径は内側の半径の2倍とする.
 * 
 */
static void 
nv_keypoint_radius(const nv_keypoint_ctx_t *ctx, nv_matrix_t *outer_r, nv_matrix_t *inner_r)
{
	float cur_r = ctx->param.min_r;
	int prev_r = NV_ROUND_INT(cur_r);
	int s;
	int r;

	r = NV_ROUND_INT(cur_r);
	NV_MAT_V(inner_r, 0, 0) = (float)r;
	NV_MAT_V(outer_r, 0, 0) = (r * 2.0f);

	for (s = 1; s < ctx->param.level;) {
		cur_r *= NV_KEYPOINT_SCALE_FACTOR;
		r = NV_ROUND_INT(cur_r);
		if (r - prev_r > 0) {
			prev_r = r;
			NV_MAT_V(inner_r, 0, s) = (float)r;
			NV_MAT_V(outer_r, 0, s) = (r * 2.0f);
			
			++s;
		}
	}
	//nv_matrix_print(stdout, g_inner_r);
}

nv_keypoint_ctx_t *
nv_keypoint_ctx_alloc(const nv_keypoint_param_t *param)
{
	int r_max;
	int i;
	nv_keypoint_ctx_t *ctx = nv_alloc_type(nv_keypoint_ctx_t, 1);

	memmove(&ctx->param, param, sizeof(ctx->param));
	ctx->inner_r = nv_matrix_alloc(ctx->param.level, 1);
	ctx->outer_r = nv_matrix_alloc(ctx->param.level, 1);
	
	nv_keypoint_radius(ctx, ctx->outer_r, ctx->inner_r);
	
	r_max = NV_ROUND_INT(NV_MAT_V(ctx->outer_r, 0, param->level-1)) + 1;
	ctx->gauss_w = nv_matrix_alloc(r_max, r_max);
	
	for (i = 1; i < r_max; ++i) {
		int j;
		for (j = 0; j < r_max; ++j) {
			float dist = (float)j / (i);
			NV_MAT_V(ctx->gauss_w, i, j) = expf(-(dist * dist) / (2.0f * 0.2f));
		}
	}
	
	return ctx;
}

void
nv_keypoint_ctx_free(nv_keypoint_ctx_t **ctx)
{
	if (ctx && *ctx) {
		nv_matrix_free(&(*ctx)->inner_r);
		nv_matrix_free(&(*ctx)->outer_r);
		nv_matrix_free(&(*ctx)->gauss_w);
		nv_free(*ctx);
		*ctx = NULL;
	}
}

static inline int
nv_keypoint_edge(const nv_keypoint_ctx_t *ctx,
				 const nv_matrix_t *img, int offset, int ky, int kx)
{
	const int sy = ky - offset;
	const int ey = ky + offset + 1;
	const int sx = kx - offset;
	const int ex = kx + offset + 1;
	int y;
	float trace;
	float det;
	float dxs = 0.0f;
	float dys = 0.0f;
	float dxdys = 0.0f;
	
	for (y = sy; y < ey; ++y) {
		int x;
		for (x = sx; x < ex; ++x) {
			float dy = NV_MAT_V(img,  y - 1, x) - NV_MAT_V(img, y + 1, x);
			float dx = NV_MAT_V(img, y, x - 1) - NV_MAT_V(img, y, x + 1);
			dxs += dx * dx;
			dys += dy * dy;
			dxdys += dx * dy;
		}
	}
	trace = dxs + dys;
	det = dxs * dys - dxdys * dxdys;
	return (trace * trace) / det >= ctx->param.edge_th;
}

static int
nv_keypoint_memo_free(int key, void *data)
{
	if (data != NULL) {
		nv_matrix_t *p = (nv_matrix_t *)data;
		nv_matrix_free(&p);
	}
	return 0;
}

static inline float 
nv_keypoint_scale_diff(const nv_matrix_t *img_integral, 
					   const nv_matrix_t *img_integral_tilted,
					   int y, int x,
					   int outer_r, int inner_r)
{
	float inner = nv_star_integral(
		img_integral, img_integral_tilted, y, x, inner_r);
	float inner_response = inner * NV_STAR_INTEGRAL_AREA_INV(inner_r);
	float outer = nv_star_integral(img_integral, img_integral_tilted,
		y, x, outer_r);
	float outer_response = (outer - inner) 
		/ (NV_STAR_INTEGRAL_AREA(outer_r) - NV_STAR_INTEGRAL_AREA(inner_r));

	return outer_response - inner_response;
}

static void
nv_keypoint_scale_search(const nv_keypoint_ctx_t *ctx,
						 nv_matrix_t *  grid_response, 
						 const nv_matrix_t *  img_integral,
						 const nv_matrix_t *  img_integral_tilted
	)
{
	int row;
	int offset = NV_ROUND_INT(NV_MAT_V(ctx->outer_r, 0, 0) * NV_KEYPOINT_DESC_SCALE);
	const int img_rows = img_integral->rows - 1;
	const int img_cols = img_integral->cols - 1;
	int erow = img_rows - offset;
	int ecol = img_cols - offset;
	const int threads = nv_omp_procs();
	nv_matrix_t *  scale_response = nv_matrix_alloc(ctx->param.level, threads);
	const nv_matrix_t *outer_r = ctx->outer_r;
	const nv_matrix_t *inner_r = ctx->inner_r;
	
	if (offset % 2 != 0) {
		offset += 1;
	}
	erow = img_rows - offset;
	ecol = img_cols - offset;
	
	/* 各画素でスケール空間の極点を検出 */
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1) num_threads(threads)
#endif
	for (row = offset; row < erow; row += 2) {
		int thread_idx = nv_omp_thread_id();
		int col;
		const int rowh = row / 2;
		
		for (col = offset; col < ecol; col += 2) {
			int level_bound = 0, s;
			const int colh = col / 2;

			for (s = 0; s < ctx->param.level; s += 2) {
				int i_r = NV_ROUND_INT(NV_MAT_V(inner_r, 0, s));
				int o_r = NV_ROUND_INT(NV_MAT_V(outer_r, 0, s));
				int o_r_offset = NV_ROUND_INT(o_r * NV_KEYPOINT_DESC_SCALE);

				if (row - o_r_offset >= 0
					&& col - o_r_offset >= 0
					&& row + o_r_offset < img_rows
					&& col + o_r_offset < img_cols)
				{
					NV_MAT_V(scale_response, thread_idx, s) = nv_keypoint_scale_diff(
						img_integral, img_integral_tilted,
						row, col, o_r, i_r);
					level_bound = s;
				} else {
					break;
				}
			}
			
			/* スケール空間の極値をとる */
			for (s = 0; s < level_bound - 3; s += 2) {
				float response_0 = NV_MAT_V(scale_response, thread_idx, s);
				float response_1 = NV_MAT_V(scale_response, thread_idx, s + 2);
				float response_2 = NV_MAT_V(scale_response, thread_idx, s + 4);
#ifndef NDEBUG
				{
					int o_r = NV_ROUND_INT(NV_MAT_V(ctx->outer_r, 0, s + 4));
					int o_r_offset = NV_ROUND_INT(o_r * NV_KEYPOINT_DESC_SCALE);
					
					NV_ASSERT(row - o_r_offset >= 0
							  && col - o_r_offset >= 0
							  && row + o_r_offset < img_rows
							  && col + o_r_offset < img_cols);
				}
#endif
				if (response_1 > 0.0f && response_0 < response_1 && response_2 < response_1) {
					float response_1_0 = nv_keypoint_scale_diff(
						img_integral, img_integral_tilted,
						row, col,
						NV_ROUND_INT(NV_MAT_V(outer_r, 0, s + 1)),
						NV_ROUND_INT(NV_MAT_V(inner_r, 0, s + 1)));
					float response_1_2 = nv_keypoint_scale_diff(
						img_integral, img_integral_tilted,
						row, col,
						NV_ROUND_INT(NV_MAT_V(outer_r, 0, s + 3)),
						NV_ROUND_INT(NV_MAT_V(inner_r, 0, s + 3)));

					if (response_1 < response_1_0) {
						if (response_1_0 > response_1_2) {
							NV_MAT3D_V(grid_response, s + 1, rowh, colh) = response_1_0;
						} else {
							NV_MAT3D_V(grid_response, s + 3, rowh, colh) = response_1_2;
						}
					} else {
						if (response_1 > response_1_2) {
							NV_MAT3D_V(grid_response, s + 2, rowh, colh) = response_1;
						} else {
							NV_MAT3D_V(grid_response, s + 3, rowh, colh) = response_1_2;
						}
					}
				} else if (response_1 < 0.0f && response_0 > response_1 && response_2 > response_1) {
					float response_1_0 = nv_keypoint_scale_diff(
						img_integral, img_integral_tilted,
						row, col,
						NV_ROUND_INT(NV_MAT_V(outer_r, 0, s + 1)),
						NV_ROUND_INT(NV_MAT_V(inner_r, 0, s + 1)));
					float response_1_2 = nv_keypoint_scale_diff(
						img_integral, img_integral_tilted,
						row, col,
						NV_ROUND_INT(NV_MAT_V(outer_r, 0, s + 3)),
						NV_ROUND_INT(NV_MAT_V(inner_r, 0, s + 3)));

					if (response_1 > response_1_0) {
						if (response_1_0 < response_1_2) {
							NV_MAT3D_V(grid_response, s + 1, rowh, colh) = response_1_0;
						} else {
							NV_MAT3D_V(grid_response, s + 3, rowh, colh) = response_1_2;
						}
					} else {
						if (response_1 < response_1_2) {
							NV_MAT3D_V(grid_response, s + 2, rowh, colh) = response_1;
						} else {
							NV_MAT3D_V(grid_response, s + 3, rowh, colh) = response_1_2;
						}
					}
				}
			}
		}
	}

	nv_matrix_free(&scale_response);
}

static int
nv_keypoint_desc_cmp(const void *p1, const void *p2)
{
	const float *f1 = (const float*) p1;
	const float *f2 = (const float*) p2;
	float ab1;
	float ab2;

	/* 半径でソート */
	if (f1[NV_KEYPOINT_RADIUS_IDX] < f2[NV_KEYPOINT_RADIUS_IDX]) {
		return 1;
	} else if (f1[NV_KEYPOINT_RADIUS_IDX] > f2[NV_KEYPOINT_RADIUS_IDX]) {
		return -1;
	}
	/* 応答の絶対値でソート */
	ab1 = fabsf(f1[NV_KEYPOINT_RESPONSE_IDX]);
	ab2 = fabsf(f2[NV_KEYPOINT_RESPONSE_IDX]);
	if (ab1 < ab2) {
		return 1;
	} else if (ab1 > ab2) {
		return -1;
	}

	return 0;
}

static void
nv_keypoint_select(const nv_keypoint_ctx_t *ctx,
				   nv_matrix_t *keypoints,
				   int *  nkeypoint,
				   const nv_matrix_t *  grid_response,
				   const int img_rows,
				   const int img_cols,
				   nv_imap_t *memo
	)
{
	int row;
	const int threads = nv_omp_procs();
	int offset = NV_ROUND_INT(NV_MAT_V(ctx->outer_r, 0, 0) * NV_KEYPOINT_DESC_SCALE);
	int erow;
	int ecol;
	const int el = ctx->param.level - 1;
	nv_matrix_t *  keypoints_work = nv_matrix_list_alloc(keypoints->n, keypoints->m, threads);
	int *nkeypoint_work = nv_alloc_type(int, threads);
	int nnmax2 = NV_ROUND_INT(NV_MAT_V(ctx->outer_r, 0, ctx->param.level - 1) * ctx->param.nn) * 2 + 2;
	nv_matrix_t *  circle_x = nv_matrix_alloc(nnmax2, ctx->param.level);
	int i, j, count;
	const nv_matrix_t *outer_r = ctx->outer_r;
	const float param_star_th = ctx->param.star_th;
	const float param_nn = ctx->param.nn;
	
	if (offset % 2 != 0) {
		offset += 1;
	}
	erow = img_rows - offset;
	ecol = img_cols - offset;

	/* 半径nnの円のYがcyのときのXを計算 */
	nv_matrix_zero(circle_x);
	for (i = 1; i < el; ++i) {
		int nn = NV_ROUND_INT(NV_MAT_V(outer_r, 0, i) * param_nn);
		int nn2 = nn * 2;
		int y;
		for (y = 0; y < nn2; ++y) {
			int cy = abs(nn - y);
			int cx = NV_ROUND_INT(sqrtf(nn * nn - cy * cy));
			NV_MAT_V(circle_x, i, cy) = (float)cx;
		}
	}
	memset(nkeypoint_work, 0, sizeof(int) * threads);

#ifdef _OPENMP
#pragma omp parallel for num_threads(threads) schedule(dynamic, 1)
#endif
	for (row = offset; row < erow; row += 2) {
		int thread_idx = nv_omp_thread_id();
		int col;
		int row_idx = row / 2;
		for (col = offset; col < ecol; col += 2) {
			int s;
			int col_idx = col / 2;

			for (s = 1; s < el; ++s) {
				int key = NV_MAT_VI(outer_r, 0, s);
				nv_matrix_t *memobuf = (nv_matrix_t *)nv_imap_find(memo, key);
				int nn, sy, sx, ey, ex;
				float response = NV_MAT3D_V(grid_response, s, row_idx, col_idx);

				NV_ASSERT(memobuf != NULL);
				if ((fabsf(response) < param_star_th)) {
					/* 応答が閾値より小さい場合は選択しない. 
					 * 極値でない応答は0が入っているのでここで弾かれる
					 */
					continue;
				}
				
				/* 近傍 半径 * NV_KEYPOINT_NN */
				nn = NV_ROUND_INT(NV_MAT_V(outer_r, 0, s) * param_nn);
				sy = row - nn;
				sx = col - nn;
				ey = row + nn + 1;
				ex = col + nn + 1;

				if (sy < 0 
					|| sx < 0
					|| ex >= img_cols
					|| ey >= img_rows)
				{
					/* 近傍の矩形範囲が画像に入らない場合は選択しない. */
					continue;
				}

				if (response > 0.0f) {
					/* 最大値か */
					int y, x, selected = 1;
					sy /= 2;
					ey /= 2;
					for (y = sy; y < ey; ++y) {
						int cx = NV_MAT_VI(circle_x, s, abs(row - y * 2));
						sx = (col - cx) / 2;
						ex = (col + cx + 1) / 2;
						for (x = sx; x < ex; ++x) {
							if (response < NV_MAT3D_V(grid_response, s, y, x)) {
								selected = 0;
								goto nv_nonmax;
							}
						}
					}
				nv_nonmax:
					if (!selected) {
						continue;
					}
				} else if (response < 0.0f) {
					/* 最小値か */
					int y, x, selected = 1;
					sy /= 2;
					ey /= 2;
					for (y = sy; y < ey; ++y) {
						int cx = NV_MAT_VI(circle_x, s, abs(row - y * 2));
						sx = (col - cx) / 2;
						ex = (col + cx + 1) / 2;
						
						for (x = sx; x < ex; ++x) {
							if (response > NV_MAT3D_V(grid_response, s, y, x)) {
								selected = 0;
								goto nv_nonmin;
							}
						}
					}
				nv_nonmin:
					if (!selected)	{
						continue;
					}
				}
				if (nv_keypoint_edge(ctx, memobuf, NV_ROUND_INT(key), row, col)) {
					continue;
				}
				/* ここまできたら特徴点選択 */
				NV_MAT_LIST_V(keypoints_work, thread_idx, nkeypoint_work[thread_idx], NV_KEYPOINT_RESPONSE_IDX) = response;
				NV_MAT_LIST_V(keypoints_work, thread_idx, nkeypoint_work[thread_idx], NV_KEYPOINT_Y_IDX) = (float)(row);
				NV_MAT_LIST_V(keypoints_work, thread_idx, nkeypoint_work[thread_idx], NV_KEYPOINT_X_IDX) = (float)(col);
				NV_MAT_LIST_V(keypoints_work, thread_idx, nkeypoint_work[thread_idx], NV_KEYPOINT_LEVEL_IDX) = (float)s;
				++nkeypoint_work[thread_idx];
			}
		}
	}
	/* 統合 */
	count = 0;
	for (j = 0; j < threads; ++j) {
		for (i = 0; i < nkeypoint_work[j]; ++i) {
			NV_MAT_V(keypoints, count, NV_KEYPOINT_RESPONSE_IDX) = NV_MAT_LIST_V(keypoints_work, j, i, NV_KEYPOINT_RESPONSE_IDX);
			NV_MAT_V(keypoints, count, NV_KEYPOINT_Y_IDX) = NV_MAT_LIST_V(keypoints_work, j, i, NV_KEYPOINT_Y_IDX);
			NV_MAT_V(keypoints, count, NV_KEYPOINT_X_IDX) = NV_MAT_LIST_V(keypoints_work, j, i, NV_KEYPOINT_X_IDX);
			NV_MAT_V(keypoints, count, NV_KEYPOINT_LEVEL_IDX) = NV_MAT_LIST_V(keypoints_work, j, i, NV_KEYPOINT_LEVEL_IDX);
			NV_MAT_V(keypoints, count, NV_KEYPOINT_RADIUS_IDX) = NV_MAT_V(outer_r, 0, (int)NV_MAT_LIST_V(keypoints_work, j, i, NV_KEYPOINT_LEVEL_IDX));
			++count;
		}
	}
	*nkeypoint = count;

	nv_matrix_free(&keypoints_work);
	nv_matrix_free(&circle_x);
	nv_free(nkeypoint_work);
}

template<int HIST_N, typename ATAN2>
static void
nv_keypoint_hist(const nv_keypoint_ctx_t *ctx,
				 nv_matrix_t *hist, int hist_m, 
				 int ky, int kx, float f_r, float angle,
				 const nv_matrix_t *  img_integral,
				 const nv_matrix_t *  img_integral_tilted,
				 const nv_matrix_t *  memo
	)
{
	const int r = NV_ROUND_INT(f_r);
	const int star_r = NV_KEYPOINT_STAR_R(f_r);
	const int star_tilted_r = NV_ROUND_INT((float)star_r * NV_SQRT2_INV);
	const int sy = (ky - r) + star_r * 2;
	const int ey = (ky + r) - star_r * 2;
	const int sx = (kx - r) + star_r * 2;
	const int ex = (kx + r) - star_r * 2;
	const float step_scale = (ex - sx) / (float)NV_KEYPOINT_HIST_SAMPLE;
	const float r2 = (float)((ex - sx) / 2) * ((ex - sx) / 2);
	int i, yi;
	const float pi_angle = NV_PI - angle;
	int *xx = nv_alloc_type(int, NV_KEYPOINT_HIST_SAMPLE);
	float *fdist_x = nv_alloc_type(float, NV_KEYPOINT_HIST_SAMPLE);
	const nv_matrix_t *gauss_w = ctx->gauss_w;
	ATAN2 atan_func;
	
	NV_ASSERT(ky + f_r < img_integral->rows-1);
	NV_ASSERT(kx + f_r < img_integral->cols-1);
	NV_ASSERT(ky - f_r >= 0);
	NV_ASSERT(kx - f_r >= 0);
	NV_ASSERT(ky - r >= 0);
	NV_ASSERT(kx - r >= 0);
	NV_ASSERT(sy > 0);
	NV_ASSERT(sx > 0);
	NV_ASSERT(ey < img_integral->rows -1);
	NV_ASSERT(ex < img_integral->cols - 1);
	
	for (i = 0; i < NV_KEYPOINT_HIST_SAMPLE; ++i) {
		xx[i] = NV_ROUND_INT(((float)sx + step_scale * i));
		fdist_x[i] = ((float)kx - xx[i]) * ((float)kx - xx[i]);
		if (xx[i] > ex) {
			xx[i] = 0;
		}
	}
	nv_vector_zero(hist, hist_m);

	/* 特徴点のr近傍から勾配ヒストグラムを作成する. */
	for (yi = 0; yi < NV_KEYPOINT_HIST_SAMPLE; ++yi) {
		const float yp = ((float)sy + step_scale * yi);
		int y = NV_ROUND_INT(yp);
		const float yd = ((float)ky - y) * ((float)ky - y);
		int xi;
		const float *pn0 = &NV_MAT_V(memo, y, 0);
		const float *pn1 = &NV_MAT_V(memo, y - star_r, 0);
		const float *pn2 = &NV_MAT_V(memo, y + star_r, 0);
		const float *pt1 = &NV_MAT_V(memo, y - star_tilted_r, 0);
		const float *pt2 = &NV_MAT_V(memo, y + star_tilted_r, 0);

		if (y >= ey) {
			y = ey - 1;
		}
		for (xi = 0; xi < NV_KEYPOINT_HIST_SAMPLE; ++xi) {
			float fdist = yd + fdist_x[xi];
			if (fdist > r2) {
				if (xi > NV_KEYPOINT_HIST_SAMPLE / 2) {
					break;
				} else {
					continue;
				}
			}
			const int x = xx[xi];
			NV_ALIGNED(float, magnitude[2], 32);
			NV_ALIGNED(float, d[4], 32);
			NV_ALIGNED(float, theta[2], 32);
			int bin[2];
			int dist = NV_ROUND_INT(sqrtf(fdist));
			
			/* (x, y)を中心とした■◆を重ねた8つの頂点から勾配の方向と強さを求める.
			 * イラストなどは局所的な変化が激しいので8点から勾配を求め平均する.
			 */
			NV_ASSERT(NV_MAT_V(memo, y, x + star_r) != NV_KEYPOINT_NOSET);
			NV_ASSERT(NV_MAT_V(memo, y, x - star_r) != NV_KEYPOINT_NOSET);
			NV_ASSERT(NV_MAT_V(memo, y + star_r, x) != NV_KEYPOINT_NOSET);
			NV_ASSERT(NV_MAT_V(memo, y - star_r, x) != NV_KEYPOINT_NOSET);
			
			NV_ASSERT(NV_MAT_V(memo, y + star_tilted_r, x + star_tilted_r) != NV_KEYPOINT_NOSET);
			NV_ASSERT(NV_MAT_V(memo, y - star_tilted_r, x - star_tilted_r) != NV_KEYPOINT_NOSET);
			NV_ASSERT(NV_MAT_V(memo, y + star_tilted_r, x - star_tilted_r) != NV_KEYPOINT_NOSET);
			NV_ASSERT(NV_MAT_V(memo, y - star_tilted_r, x + star_tilted_r) != NV_KEYPOINT_NOSET);

			d[0] = pn0[x + star_r] - pn0[x - star_r];
			d[1] = pn2[x] - pn1[x];
			d[2] = pt2[x + star_tilted_r] - pt1[x - star_tilted_r];
			d[3] = pt2[x - star_tilted_r] - pt1[x + star_tilted_r];
			
			magnitude[0] = sqrtf(d[0] * d[0] + d[1] * d[1]);
			magnitude[1] = sqrtf(d[2] * d[2] + d[3] * d[3]);
			
			theta[0] = atan_func(d[1], d[0]) + pi_angle;
			theta[1] = atan_func(d[3], d[2]) + pi_angle;
			
			if (theta[0] < 0.0f) {
				theta[0] = NV_PI * 2.0f + theta[0];
			}
			if (theta[1] < 0.0f) {
				theta[1] = NV_PI * 2.0f + theta[1];
			}
			
			bin[0] = NV_ROUND_INT(HIST_N * NV_PI2_INV * theta[0]);
			bin[1] = NV_ROUND_INT(HIST_N * NV_PI2_INV * theta[1]) + (HIST_N * 45 / 360);
			
			if (bin[0] >= HIST_N) {
				bin[0] = bin[0] - HIST_N;
			}
			NV_ASSERT(bin[0] < HIST_N);
			
			if (bin[1] >= HIST_N) {
				bin[1] = bin[1] - HIST_N;
			}
			NV_ASSERT(bin[1] < HIST_N);
			
			NV_MAT_V(hist, hist_m, bin[0]) += magnitude[0] * NV_MAT_V(gauss_w, r, dist);
			NV_MAT_V(hist, hist_m, bin[1]) += magnitude[1] * NV_MAT_V(gauss_w, r, dist);
		}
	}

	nv_free(xx);
	nv_free(fdist_x);
}

static void
nv_keypoint_make_scale_space(const nv_keypoint_ctx_t *ctx,
							 nv_imap_t *memo,
							 const nv_matrix_t *  img_integral,
							 const nv_matrix_t *  img_integral_tilted)
{
	int i;
	
	const int threads = nv_omp_procs();

	for (i = 1; i < ctx->param.level - 1; ++i) {
		float f_r = NV_MAT_V(ctx->outer_r, 0, i);
		int r = NV_ROUND_INT(f_r);
		
		if (nv_imap_find(memo, r) == NULL) {
			nv_matrix_t *  memobuf = nv_matrix_alloc(img_integral->cols,
													 img_integral->rows);
			int y;
			const int star_r = NV_KEYPOINT_STAR_R(f_r);
			const int sy = star_r + 1;
			const int ey = img_integral->rows -1 - star_r;
			const int sx = star_r + 1;
			const int ex = img_integral->cols -1 - star_r;
			const float area_inv = NV_STAR_INTEGRAL_AREA_INV(star_r);
			const int side_half = NV_ROUND_INT(star_r * NV_SQRT2_INV);

#ifndef NDEBUG
			nv_matrix_fill(memobuf, NV_KEYPOINT_NOSET);
#endif

#ifdef _OPENMP
#pragma omp parallel for num_threads(threads)
#endif
			for (y = sy; y < ey; ++y) {
				int x;
				for (x = sx; x < ex; ++x) {
					//v = nv_star_integral(img_integral, img_integral_tilted, y, x, star_r);
					float v = NV_INTEGRAL_V(img_integral,
											x - side_half, y - side_half,
											x + side_half,
											y + side_half)
						+ NV_MAT3D_V(img_integral_tilted, (y - star_r), x, 0)
						- NV_MAT3D_V(img_integral_tilted, y, (x - star_r), 0)
						- NV_MAT3D_V(img_integral_tilted, y, (x + star_r), 0)
						+ NV_MAT3D_V(img_integral_tilted, (y + star_r), x, 0);
					if (v < 0.0f) {
						v = 0.0f;
					}
					NV_MAT_V(memobuf, y, x) = v * area_inv;
				}
			}
			nv_imap_insert(memo, r, memobuf);
		}
	}
}

static void
nv_keypoint_orientation(const nv_keypoint_ctx_t *ctx,
						nv_matrix_t *keypoints,
						int *nkeypoint,
						const nv_matrix_t *  img_integral,
						const nv_matrix_t *  img_integral_tilted,
						nv_imap_t *memo
	)
{
	int i;
	const int threads = nv_omp_procs();
	nv_matrix_t *hist = nv_matrix_alloc(NV_KEYPOINT_ORIENTATION_HIST, threads);
#if NV_KEYPOINT_MULTI_ORIENTATION
	nv_matrix_t *multi = nv_matrix3d_alloc(keypoints->n, threads, *nkeypoint);
	int *multi_i = nv_alloc_type(int, threads);
	memset(multi_i, 0, sizeof(int) * threads);
#endif
	
#ifdef _OPENMP
#pragma omp parallel for num_threads(threads) schedule(dynamic, 1)
#endif
	for (i = 0; i < *nkeypoint; ++i) {
		int i1;
		
		const int thread_idx = nv_omp_thread_id();
		nv_matrix_t *  memobuf = (nv_matrix_t *)nv_imap_find(memo, 
			NV_ROUND_INT(NV_MAT_V(keypoints, i, NV_KEYPOINT_RADIUS_IDX)));
		NV_ASSERT(memobuf != NULL);

		/* 勾配ヒストグラムのTOPを記述子の正規化方向とする. */
		nv_keypoint_hist<NV_KEYPOINT_ORIENTATION_HIST, native_atan2_t>(ctx,
						 hist, thread_idx, 
						 NV_ROUND_INT(NV_MAT_V(keypoints, i, NV_KEYPOINT_Y_IDX)),
						 NV_ROUND_INT(NV_MAT_V(keypoints, i, NV_KEYPOINT_X_IDX)),
						 NV_MAT_V(keypoints, i, NV_KEYPOINT_RADIUS_IDX), 0.0f,
						 img_integral,
						 img_integral_tilted, memobuf);
		i1 = nv_vector_max_n(hist, thread_idx);
		NV_MAT_V(keypoints, i, NV_KEYPOINT_ORIENTATION_IDX) = ((2.0f * NV_PI) / (float)hist->n) * i1 - NV_PI;
		
#if NV_KEYPOINT_MULTI_ORIENTATION
		{
			/* TOP1,2の値が近くて角度が離れている場合は正規化角度の異なる複製を作る. */
			int i2;
			float v1, v2;
			int copy = 0;
			
			v1 = NV_MAT_V(hist, thread_idx, i1);
			NV_MAT_V(hist, thread_idx, i1) = 0.0f;
			i2 = nv_vector_max_n(hist, thread_idx);
			v2 = NV_MAT_V(hist, thread_idx, i2);
			
			if (v1 > 0.0f && v2 / v1 > 0.8f) {
				if (i1 > i2) {
					if (i1 == NV_KEYPOINT_ORIENTATION_HIST - 1) {
						if (!(i2 == 0 || i2 == NV_KEYPOINT_ORIENTATION_HIST - 2)) {
							copy = 1;
						}
					} else {
						if (i1 - i2 != 1) {
							copy = 1;
						}
					}
				} else {
					if (i2 == NV_KEYPOINT_ORIENTATION_HIST - 1) {
						if (!(i1 == 0 || i1 == NV_KEYPOINT_ORIENTATION_HIST - 2)) {
							copy = 1;
						}
					} else {
						if (i2 - i1 != 1) {
							copy = 1;
						}
					}
				}
			}
			if (copy) {
				nv_vector_copy(multi, NV_MAT_M(multi, thread_idx, multi_i[thread_idx]), keypoints, i);
				NV_MAT3D_V(multi, thread_idx, multi_i[thread_idx], NV_KEYPOINT_ORIENTATION_IDX) = ((2.0f * NV_PI) / (float)hist->n) * i2 - NV_PI;
				++multi_i[thread_idx];
			}
		}
#endif		
	}
#if NV_KEYPOINT_MULTI_ORIENTATION
	{
		int j;
		for (i = 0; i < threads; ++i) {
			for (j = 0; j < multi_i[i]; ++j) {
				nv_vector_copy(keypoints, *nkeypoint, multi, NV_MAT_M(multi, i, j));
				++*nkeypoint;
			}
		}
		nv_matrix_free(&multi);
		nv_free(multi_i);
	}
#endif
	
	nv_matrix_free(&hist);
}

static void
nv_keypoint_sort(nv_matrix_t *keypoints,
				 const nv_matrix_t *keypoints_tmp,
				 int nkeypoint, int limit)
{
	if (nkeypoint > 0) {
		qsort(keypoints_tmp->v, nkeypoint,
			  keypoints_tmp->step * sizeof(float), nv_keypoint_desc_cmp);
		nv_matrix_copy(keypoints, 0, keypoints_tmp, 0, NV_MIN(limit, nkeypoint));
	}
}

/*
 * 特徴点の特徴ベクトルを算出する.
 * 勾配ヒストグラム特徴は,
 * 特徴点を中心とした半径desc_rの円上の
 * 8点(PI/4ごと)を中心とした半径desc_rの円内と
 * (勾配ヒストグラムは半径desc_r * 2の範囲から計算される)
 * 中心の勾配ヒストグラム(8bin)とする.
 * (8 * (8 + 1) = 72次元の特徴ベクトルとなる)
*/
static void 
nv_keypoint_gradient_histogram(const nv_keypoint_ctx_t *ctx,
							   nv_matrix_t *  desc, 
							   const nv_matrix_t *  keypoints,
							   const int keypoint_m,
							   const nv_matrix_t *  integral,
							   const nv_matrix_t *  integral_tilted,
							   nv_matrix_t *  memo)
{
	NV_ALIGNED(static const float, circle_steps[8], 32) = {
		0.0f,
		NV_PI / 4.0f * 1.0f, NV_PI / 4.0f * 2.0f,
		NV_PI / 4.0f * 3.0f, NV_PI / 4.0f * 4.0f,
		NV_PI / 4.0f * 5.0f, NV_PI / 4.0f * 6.0f,
		NV_PI / 4.0f * 7.0f
	};
	float y = NV_MAT_V(keypoints, keypoint_m, NV_KEYPOINT_Y_IDX);
	float x = NV_MAT_V(keypoints, keypoint_m, NV_KEYPOINT_X_IDX);
	float desc_r = NV_MAT_V(keypoints, keypoint_m, NV_KEYPOINT_RADIUS_IDX);
	float angle = NV_PI + NV_MAT_V(keypoints, keypoint_m, NV_KEYPOINT_ORIENTATION_IDX);
	int i;
	nv_matrix_t *hist = nv_matrix_alloc(8, 1);
	
#if NV_ENABLE_SSE3
#  define NV_KEYPOINT_GRADIENT_HISTOGRAM_NORMALIZE(hist)	{			\
		__m128 a, b, c, d;												\
		float norm;														\
		a = _mm_load_ps(&NV_MAT_V(hist, 0, 0));							\
		b = _mm_load_ps(&NV_MAT_V(hist, 0, 4));							\
		c = _mm_mul_ps(a, a);											\
		d = _mm_mul_ps(b, b);											\
		d = _mm_add_ps(c, d);											\
		d = _mm_hadd_ps(d, d);											\
		d = _mm_hadd_ps(d, d);											\
	    _mm_store_ss(&norm, d);											\
		if (norm > 0.0f) {												\
			c = _mm_set1_ps(1.0f / sqrtf(norm));						\
			_mm_store_ps(&NV_MAT_V(hist, 0, 0), _mm_mul_ps(a, c));		\
			_mm_store_ps(&NV_MAT_V(hist, 0, 4), _mm_mul_ps(b, c));		\
		}																\
     }
#else
#  define NV_KEYPOINT_GRADIENT_HISTOGRAM_NORMALIZE(hist) nv_vector_normalize(hist, 0);
#endif

	NV_ASSERT(y + desc_r < integral->rows-1);
	NV_ASSERT(x + desc_r < integral->cols-1);
	NV_ASSERT(y - desc_r >= 0);
	NV_ASSERT(x - desc_r >= 0);
	
	/* 特徴点を中心とした半径desc_rの円上の8点について各勾配ヒストグラムを計算する */
	for (i = 0; i < 8; ++i) {
		float theta = circle_steps[i] + angle;
		if (theta > 2.0f * NV_PI) {
			theta = (theta - 2.0f * NV_PI);
		}
		nv_keypoint_hist<8, approximate_atan2_t>(
			ctx,
			hist, 0, 
			NV_ROUND_INT(desc_r * sinf(theta) + y),
			NV_ROUND_INT(desc_r * cosf(theta) + x),
			desc_r, angle,
			integral, integral_tilted, memo
		);
		NV_KEYPOINT_GRADIENT_HISTOGRAM_NORMALIZE(hist);
		memmove(&NV_MAT_V(desc, keypoint_m, i * hist->n),
			   &NV_MAT_V(hist, 0, 0),
			   sizeof(float) * hist->n);
	}
	/* 中心 */
	nv_keypoint_hist<8, approximate_atan2_t>(
		ctx,
		hist, 0, 
		y, x, desc_r, angle,
		integral, integral_tilted, memo
		);
	NV_KEYPOINT_GRADIENT_HISTOGRAM_NORMALIZE(hist);
	memmove(&NV_MAT_V(desc, keypoint_m, 8 * hist->n),
		   &NV_MAT_V(hist, 0, 0),
		   sizeof(float) * hist->n);
	nv_matrix_free(&hist);
}

/*
 * 8方向面特徴を計算する
 */
static void
nv_keypoint_rectangle_feature8(nv_matrix_t *  hist, int hist_m, 
							   int ky, int kx, float r,
							   float angle,
							   const nv_matrix_t *integral,
							   const nv_matrix_t *integral_tilted
	)
{
	static const float filters[4][2] = {
		{ 0.0f, NV_PI },
		{ NV_PI / 2.0f, NV_PI / 2.0f * 3.0f },
		{ NV_PI / 4.0f * 1.0f, NV_PI / 4.0f * 5.0f },
		{ NV_PI / 4.0f * 3.0f, NV_PI / 4.0f * 7.0f }
	};
	const int star_r = NV_ROUND_INT(r * 0.5f);
	const int side_half = NV_ROUND_INT(star_r * NV_SQRT2_INV);
	int i;
	
	for (i = 0; i < 4; ++i) {
		static const float op[2] = { 1.0f, -1.0f };
		float p = 0.0f;
		int j;
		
		for (j = 0; j < 2; ++j) {
			int x, y;
			float theta, v;
			
			theta = filters[i][j] + angle;
			if (theta > 2.0f * NV_PI) {
				theta = theta - 2.0f * NV_PI;
			}
			y = NV_ROUND_INT(star_r * sinf(theta) + ky);
			x = NV_ROUND_INT(star_r * cosf(theta) + kx);
			
			v = NV_INTEGRAL_V(integral,
							 x - side_half, y - side_half,
							 x - side_half + (side_half * 2),
							 y - side_half + (side_half * 2))
			+ NV_MAT3D_V(integral_tilted, (y - star_r), x, 0)
			- NV_MAT3D_V(integral_tilted, y, (x - star_r), 0)
			- NV_MAT3D_V(integral_tilted, y, (x + star_r), 0)
			+ NV_MAT3D_V(integral_tilted, (y + star_r), x, 0);
			if (v < 0.0f) {
				v = 0.0f;
			}
			p += v * op[j];
		}
		if (p > 0.0f) {
			NV_MAT_V(hist, hist_m, i * 2) = p;
			NV_MAT_V(hist, hist_m, i * 2 + 1) = 0.0f;
		} else {
			NV_MAT_V(hist, hist_m, i * 2) = 0.0f;
			NV_MAT_V(hist, hist_m, i * 2 + 1) = -p;
		}
	}
}

static void 
nv_keypoint_rectangle_feature(nv_matrix_t *  desc, 
							  const nv_matrix_t *  keypoints,
							  const int keypoint_m,
							  const nv_matrix_t *  integral,
							  const nv_matrix_t *  integral_tilted)
{
	NV_ALIGNED(static const float, circle_steps[8], 32) = {
		0.0f,
		NV_PI / 4.0f * 1.0f, NV_PI / 4.0f * 2.0f,
		NV_PI / 4.0f * 3.0f, NV_PI / 4.0f * 4.0f,
		NV_PI / 4.0f * 5.0f, NV_PI / 4.0f * 6.0f,
		NV_PI / 4.0f * 7.0f
	};
	float y = NV_MAT_V(keypoints, keypoint_m, NV_KEYPOINT_Y_IDX);
	float x = NV_MAT_V(keypoints, keypoint_m, NV_KEYPOINT_X_IDX);
	float desc_r = NV_MAT_V(keypoints, keypoint_m, NV_KEYPOINT_RADIUS_IDX);
	float angle = NV_PI + NV_MAT_V(keypoints, keypoint_m, NV_KEYPOINT_ORIENTATION_IDX);
	int i, j;
	nv_matrix_t *hist = nv_matrix_alloc(8, 1);
	float theta;
	int ay;
	int ax;
	
	NV_ASSERT(y + desc_r < integral->rows-1);
	NV_ASSERT(x + desc_r < integral->cols-1);
	NV_ASSERT(y - desc_r >= 0);
	NV_ASSERT(x - desc_r >= 0);
	
	/* 特徴点を中心とした半径desc_rの円上の8点について8方向面特徴を計算する */
	for (i = 0; i < 8; ++i) {
		theta = circle_steps[i] + angle;
		if (theta > 2.0f * NV_PI) {
			theta = (theta - 2.0f * NV_PI);
		}
		
		ay = NV_ROUND_INT(desc_r * sinf(theta) + y);
		ax = NV_ROUND_INT(desc_r * cosf(theta) + x);
		nv_keypoint_rectangle_feature8(
			hist, 0, 
			ay, ax, desc_r, angle,
			integral, integral_tilted
		);
		nv_vector_normalize(hist, 0);
		
		for (j = 0; j < hist->n; ++j) {
			NV_MAT_V(desc, keypoint_m, i * hist->n + j) = NV_MAT_V(hist, 0, j);
		}
	}
	/* 全体(desc_r * 2) */
	nv_keypoint_rectangle_feature8(
		hist, 0, 
		y, x, desc_r * 2.0f, angle,
		integral, integral_tilted);
	nv_vector_normalize(hist, 0);
	for (j = 0; j < hist->n; ++j) {
		NV_MAT_V(desc, keypoint_m, hist->n * 8 + j) = NV_MAT_V(hist, 0, j);
	}
	
	nv_matrix_free(&hist);
}

static void 
nv_keypoint_detect(const nv_keypoint_ctx_t *ctx,
				   nv_matrix_t *keypoints, int *nkeypoint,
				   const nv_matrix_t *integral,
				   const nv_matrix_t *integral_tilted,							
				   const int limit,
				   nv_imap_t *memo
				   
	)
{
	const int img_rows = integral->rows - 1;
	const int img_cols = integral->cols - 1;	
	nv_matrix_t *grid_response = nv_matrix3d_alloc(img_cols / 2,
												   ctx->param.level,
												   img_rows / 2
												   );
	nv_matrix_t *keypoints_tmp = nv_matrix_alloc(NV_KEYPOINT_KEYPOINT_N,
												 img_rows * img_cols);
	
	NV_ASSERT(keypoints->m >= limit);
	NV_ASSERT(keypoints->n >= NV_KEYPOINT_KEYPOINT_N);
	
	nv_matrix_zero(grid_response);

	/* 画素ごとに特徴点（候補）のスケールを探索する. */
	nv_keypoint_scale_search(
		ctx, grid_response,
		integral, integral_tilted);
	
	/* 特徴点を選択する.  */
	nv_keypoint_select(ctx,
					   keypoints_tmp, nkeypoint, grid_response,
					   img_rows, img_cols,
					   memo
		);

	if (*nkeypoint > 0) {
		/* 各特徴点のオリエンテーションを求める.
		 * オリエンテーションが不安定そうな点は正規化の方向を複数持つためここで点の数が増える.
		 */
		nv_keypoint_orientation(
			ctx, keypoints_tmp, nkeypoint,
			integral, integral_tilted, memo);
		/* ソートしてlimitが少ない場合は下位を消す */
		nv_keypoint_sort(keypoints, keypoints_tmp, *nkeypoint, limit);
		if (*nkeypoint > limit) {
			*nkeypoint = limit;
		}
	}

	nv_matrix_free(&keypoints_tmp);
	nv_matrix_free(&grid_response);
}

static void 
nv_keypoint_vector(const nv_keypoint_ctx_t *ctx,
				   nv_matrix_t *desc,
				   const nv_matrix_t *keypoints,
				   const int nkeypoint,
				   const nv_matrix_t *integral,
				   const nv_matrix_t *integral_tilted,
				   nv_imap_t *memo
	)
{
	int i;
	const int threads = nv_omp_procs();

	switch (ctx->param.descriptor) {
	case NV_KEYPOINT_DESCRIPTOR_GRADIENT_HISTOGRAM:
#ifdef _OPENMP
#pragma omp parallel for num_threads(threads) schedule(dynamic, 1)
#endif
		for (i = 0; i < nkeypoint; ++i) {
			nv_matrix_t *memobuf = (nv_matrix_t *)
				nv_imap_find(memo, 
							 NV_ROUND_INT(NV_MAT_V(keypoints, i, NV_KEYPOINT_RADIUS_IDX)));
			NV_ASSERT(memobuf != NULL);
			nv_keypoint_gradient_histogram(ctx, desc, keypoints, i,
										   integral, integral_tilted, memobuf);
		}
		break;
	case NV_KEYPOINT_DESCRIPTOR_RECTANGLE_FEATURE:
#ifdef _OPENMP
#pragma omp parallel for num_threads(threads) schedule(dynamic, 1)
#endif
		for (i = 0; i < nkeypoint; ++i) {
			nv_keypoint_rectangle_feature(desc, keypoints, i,
										  integral, integral_tilted);			
		}
		break;
	default:
		NV_ASSERT("unknown descriptor " == NULL);
		break;
	}
}

static int
nv_keypoint_cpu(const nv_keypoint_ctx_t *ctx,
				nv_matrix_t *keypoints,
				nv_matrix_t *desc,
				const nv_matrix_t *img,
				const int channel
	)
{
	nv_matrix_t *integral = nv_matrix3d_alloc(1, img->rows + 1, img->cols + 1);
	nv_matrix_t *integral_tilted = nv_matrix3d_alloc(1, img->rows + 1, img->cols + 1);
	int nkeypoint = 0;
	nv_imap_t *memo = nv_imap_alloc();
	
	NV_ASSERT(desc->n == NV_KEYPOINT_DESC_N);
	
	nv_integral(integral, img, channel);
	nv_integral_tilted(integral_tilted, img, channel);

	nv_keypoint_make_scale_space(ctx,
								 memo,
								 integral,
								 integral_tilted
		);
	nv_keypoint_detect(ctx, keypoints, &nkeypoint, integral, integral_tilted,
					   keypoints->m, memo);
	nv_keypoint_vector(ctx, desc, keypoints, nkeypoint,
					   integral, integral_tilted, memo);
	nv_matrix_free(&integral);
	nv_matrix_free(&integral_tilted);
	nv_imap_foreach(memo, nv_keypoint_memo_free);
	nv_imap_free(&memo);
	
	return nkeypoint;
}

int
nv_keypoint_ex(const nv_keypoint_ctx_t *ctx,
			   nv_matrix_t *keypoints,
			   nv_matrix_t *desc,
			   const nv_matrix_t *img,
			   const int channel
	)
{
	if (nv_cuda_enabled() 
		&& nv_keypoint_gpu != NULL 
		&& memcmp(&ctx->param,NV_KEYPOINT_PARAM_DEFAULT, sizeof(ctx->param)) == 0)
	{
		return (*nv_keypoint_gpu)(keypoints, desc, img, channel);
	}
	return nv_keypoint_cpu(ctx, keypoints, desc, img, channel);
}

int
nv_keypoint(nv_matrix_t *keypoints,
			nv_matrix_t *desc,
			const nv_matrix_t *img,
			const int channel)
{
	nv_keypoint_ctx_t *ctx = nv_keypoint_ctx_alloc(NV_KEYPOINT_PARAM_DEFAULT);
	int nkeypoint = nv_keypoint_ex(ctx, keypoints, desc, img, channel);
	nv_keypoint_ctx_free(&ctx);
	
	return nkeypoint;
}

int
nv_keypoint_dense(nv_matrix_t *keypoints,
				  nv_matrix_t *desc,
				  const nv_matrix_t *img,
				  const int channel,
				  const nv_keypoint_dense_t *dense,
				  int n_dense)
{
	nv_keypoint_ctx_t *ctx = nv_keypoint_ctx_alloc(NV_KEYPOINT_PARAM_DEFAULT);
	int ret = nv_keypoint_dense_ex(ctx,
								   keypoints, desc, img, channel, dense, n_dense);
	nv_keypoint_ctx_free(&ctx);
	return ret;
}

int
nv_keypoint_dense_ex(const nv_keypoint_ctx_t *ctx,
					 nv_matrix_t *keypoints,
					 nv_matrix_t *desc,
					 const nv_matrix_t *img,
					 const int channel,
					 const nv_keypoint_dense_t *dense,
					 int n_dense
	)
{
	int i;
	nv_matrix_t *integral = nv_matrix3d_alloc(1, img->rows + 1, img->cols + 1);
	nv_matrix_t *integral_tilted = nv_matrix3d_alloc(1, img->rows + 1, img->cols + 1);
	nv_imap_t *memo = nv_imap_alloc();
	int km = 0;
	const nv_matrix_t *outer_r = ctx->outer_r;
	
	nv_integral(integral, img, 0);
	nv_integral_tilted(integral_tilted, img, 0);
	nv_keypoint_make_scale_space(ctx, memo, integral, integral_tilted);

	for (i = 0; i < n_dense; ++i) {
		int level = -1;
		float min_dist = FLT_MAX;
		int j;
		int x, y;
		float sy, ey, sx, ex;
		float xstep, ystep;
		
		for (j = 1; j < ctx->param.level - 1; ++j) {
			float dist = (NV_MAT_V(ctx->outer_r, 0, j) - dense[i].r) *
				(NV_MAT_V(ctx->outer_r, 0, j) - dense[i].r);
			if (min_dist > dist) {
				min_dist = dist;
				level = j;
			}
		}
		sx = sy = NV_MAT_V(ctx->outer_r, 0, level) * NV_KEYPOINT_DESC_SCALE;
		ex = (img->cols - 1) - sx;
		ey = (img->rows - 1) - sy;
		xstep = (ex - sx) / dense[i].cols;
		ystep = (ey - sy) / dense[i].rows;
		xstep += xstep / dense[i].cols;
		ystep += ystep / dense[i].rows;
		
		for (y = 0; y < dense[i].rows; ++y) {
			for (x = 0; x < dense[i].cols; ++x) {
				float ky = sy + y * ystep;
				float kx = sx + x * xstep;
				float response = nv_keypoint_scale_diff(
					integral, integral_tilted,
					NV_FLOOR_INT(ky),
					NV_FLOOR_INT(kx),
					NV_MAT_V(ctx->outer_r, 0, level),
					NV_MAT_V(ctx->inner_r, 0, level));
				if (response != 0.0f) {
					NV_MAT_V(keypoints, km, NV_KEYPOINT_RESPONSE_IDX) =
						NV_MAT_V(keypoints, km, NV_KEYPOINT_Y_IDX) = ky;
					NV_MAT_V(keypoints, km, NV_KEYPOINT_X_IDX) = kx;
					NV_MAT_V(keypoints, km, NV_KEYPOINT_RADIUS_IDX) = NV_MAT_V(outer_r, 0, level);
					++km;
				}
			}
		}
	}
	nv_keypoint_orientation(ctx, keypoints, &km, integral, integral_tilted, memo);
	nv_keypoint_vector(ctx, desc, keypoints, km, integral, integral_tilted, memo);

	nv_matrix_free(&integral);
	nv_matrix_free(&integral_tilted);
	nv_imap_foreach(memo, nv_keypoint_memo_free);
	nv_imap_free(&memo);
	
	return km;
}

const nv_keypoint_param_t *
nv_keypoint_param_gradient_histogram_default(void)
{
	static const nv_keypoint_param_t s_param = {
		NV_KEYPOINT_THRESH,
		NV_KEYPOINT_EDGE_THRESH,
		NV_KEYPOINT_MIN_R,
		NV_KEYPOINT_LEVEL,
		NV_KEYPOINT_NN,
		NV_KEYPOINT_DETECTOR_STAR,
		NV_KEYPOINT_DESCRIPTOR_GRADIENT_HISTOGRAM
	};
	return &s_param;
}

const nv_keypoint_param_t *
nv_keypoint_param_rectangle_feature_default(void)
{
	static const nv_keypoint_param_t s_param = {
		NV_KEYPOINT_THRESH,
		NV_KEYPOINT_EDGE_THRESH,
		NV_KEYPOINT_MIN_R * powf(NV_KEYPOINT_SCALE_FACTOR, 4),
		NV_KEYPOINT_LEVEL,
		NV_KEYPOINT_NN,
		NV_KEYPOINT_DETECTOR_STAR,
		NV_KEYPOINT_DESCRIPTOR_RECTANGLE_FEATURE
	};
	return &s_param;
}
