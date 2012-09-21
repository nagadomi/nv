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

#include <cutil_inline.h>
#include "nv_core.h"
#include "nv_ip.h"
#include "nv_cuda.h"
#include "nv_cuda_keypoint.h"

#define NV_KEYPOINT_STAR_R(r) NV_ROUND_INT(r / 6.0f)
#define NV_PI2_INV (1.0f / (NV_PI * 2.0f))

#define NV_KEYPOINT_MIN_POINT_R  6       /* 検出される最小の半径(だった値...) */
#define NV_KEYPOINT_SCALE_FACTOR 1.090508f// 2^(1/8) 8ステップして2倍になる */
#define NV_KEYPOINT_MIN_R        5.187362f/* 探索開始の半径 */
#define NV_KEYPOINT_LEVEL        17       /* 探索する半径の階層数 */
#define NV_KEYPOINT_DESC_SCALE   2.0f     /* 記述子を計算する際の半径のスケール。
											 変えるな危険。 */

/* オリエンテーションの量子化数 */
#define NV_KEYPOINT_ORIENTATION_HIST 64

/* 勾配のサンプル数。
   多いほうが特徴量が安定するが遅くなる。
   (6 * 3 + 1) : 精度重視
   (6 * 2 + 1) : 普通
   (6 * 1 + 1) : 速度重視
 */
#define NV_KEYPOINT_HIST_SAMPLE (6 * 3 + 1)  


#define BENCHMARK 0

#define NV_CUDA_KEYPOINT_HIST_DIR 8
#define NV_CUDA_KEYPOINT_DESC_M   9
#define NV_CUDA_KEYPOINT_RADIUS_MAX 50

static __device__ float
nv_cuda_star_integral(const nv_matrix_t *integral,
					  const nv_matrix_t *integral_tilted,
					  int row, int col, int r)
{
	float intl_norm, intl_tilt;

	/* 正方形の一辺の長さの半分 */
	const int side_half = NV_ROUND_INT(r * NV_SQRT2_INV);
	const int cs = col - side_half;
	const int rs = row - side_half;
	const int sh2 = side_half * 2;
	
	/* 正方形■の積分 */

	intl_norm = NV_INTEGRAL_V(integral, cs, rs, cs + sh2, rs + sh2);

	/* 正方形◆の積分 */
	intl_tilt =
		NV_MAT3D_V(integral_tilted, (row - r), col, 0)
		- NV_MAT3D_V(integral_tilted, row, (col - r), 0)
		- NV_MAT3D_V(integral_tilted, row, (col + r), 0)
		+ NV_MAT3D_V(integral_tilted, (row + r), col, 0);

	/* 合計を計算する(誤差によって負になることがあるので0.0に) */
	if (intl_norm < 0.0f) {
		intl_norm = 0.0f;
	}
	if (intl_tilt < 0.0f) {
		intl_tilt = 0.0f;
	}
	return intl_norm + intl_tilt;
}

static __global__ void
nv_cuda_keypoint_make_scale_space(
								  nv_matrix_t **memo,
								  const float *area_inv_table,
								  const nv_matrix_t *outer_r,
								  const nv_matrix_t *inner_r,
								  const nv_matrix_t *img_integral,
								  const nv_matrix_t *img_integral_tilted)
{
	int i = blockIdx.y;
	int my_m = blockDim.x * blockIdx.x + threadIdx.x;
	int x = my_m % img_integral->cols;
	int y = my_m / img_integral->cols;

	if (1 <= i && i < NV_KEYPOINT_LEVEL - 1) {
		float f_r = NV_MAT_V(outer_r, 0, i);
		const int star_r = NV_KEYPOINT_STAR_R(f_r);
		const int sy = star_r;
		const int ey = img_integral->rows -1 - star_r;
		const int sx = star_r;
		const int ex = img_integral->cols -1 - star_r;
		const float area_inv = area_inv_table[star_r];
		const int side_half = NV_ROUND_INT(star_r * NV_SQRT2_INV);

		if (sy <= y  && y < ey && sx <= x && x < ex) {
			//float v = nv_cuda_star_integral(img_integral, img_integral_tilted, y, x, star_r);
			float v = NV_INTEGRAL_V(img_integral,
				x - side_half, y - side_half,
				x - side_half + (side_half * 2),
				y - side_half + (side_half * 2))
				+ NV_MAT3D_V(img_integral_tilted, (y - star_r), x, 0)
				- NV_MAT3D_V(img_integral_tilted, y, (x - star_r), 0)
				- NV_MAT3D_V(img_integral_tilted, y, (x + star_r), 0)
				+ NV_MAT3D_V(img_integral_tilted, (y + star_r), x, 0);
			if (v < 0.0f) {
				v = 0.0f;
			}
			NV_MAT_V(memo[i], y, x) = v * area_inv;
		}
	}
}


static __device__ float 
nv_cuda_keypoint_scale_diff(const nv_matrix_t *img_integral, 
					   const nv_matrix_t *img_integral_tilted,
					   const float *area_table,
					   const float *area_inv_table,
					   int y, int x,
					   int outer_r, int inner_r)
{
	float inner = nv_cuda_star_integral(
		img_integral, img_integral_tilted, y, x, inner_r);
	float inner_response = inner * area_inv_table[inner_r];
	float outer = nv_cuda_star_integral(img_integral, img_integral_tilted,
		y, x, outer_r);
	float outer_response = (outer - inner) 
		/ (area_table[outer_r] - area_table[inner_r]);

	return outer_response - inner_response;
}

/* 
 * 画素ごとに特徴点（候補）のスケールを推定する.
 *
 * 
 */
static __global__ void
nv_cuda_keypoint_scale_search(nv_matrix_t *grid_response, 
							  nv_matrix_t *scale_response,
							  const float *area_table,
							  const float *area_inv_table,
							  const nv_matrix_t *  img_integral,
							  const nv_matrix_t *  img_integral_tilted,
							  const nv_matrix_t *  outer_r,
							  const nv_matrix_t *  inner_r,
							  const int img_rows,
							  const int img_cols
							  )
{
	const int my_m = blockDim.x * blockIdx.x + threadIdx.x;
	const int col = (my_m % (img_cols / 2)) * 2;
	const int row = (my_m / (img_cols / 2)) * 2;
	const int thread_idx = my_m;
	const int offset = NV_ROUND_INT(NV_MAT_V(outer_r, 0, 0) * NV_KEYPOINT_DESC_SCALE) & ~1;
	const int erow = img_rows - offset;
	const int ecol = img_cols - offset;

	/* 各画素でスケール空間の極点を検出 */
	if (offset <= row && row < erow && offset <= col && col < ecol) {
		int level_bound = 0, s;
		const int col_idx = col / 2;
		const int row_idx = row / 2;

		for (s = 0; s < NV_KEYPOINT_LEVEL; s += 2) {
			const int i_r = NV_ROUND_INT(NV_MAT_V(inner_r, 0, s));
			const int o_r = NV_ROUND_INT(NV_MAT_V(outer_r, 0, s));
			const int o_r_offset = NV_ROUND_INT(o_r * NV_KEYPOINT_DESC_SCALE);

			if (row - o_r_offset >= 0
				&& col - o_r_offset >= 0
				&& row + o_r_offset < img_rows
				&& col + o_r_offset < img_cols)
			{
				__syncthreads();
				NV_MAT_V(scale_response, thread_idx, s) = nv_cuda_keypoint_scale_diff(
					img_integral, img_integral_tilted, area_table, area_inv_table,
					row, col, o_r, i_r);
				level_bound = s;
			} else {
				break;
			}
		}
		/* スケール空間の極値をとる */
		for (s = 0; s < level_bound - 3; s += 2) {
			const float response_0 = NV_MAT_V(scale_response, thread_idx, s);
			const float response_1 = NV_MAT_V(scale_response, thread_idx, s + 2);
			const float response_2 = NV_MAT_V(scale_response, thread_idx, s + 4);
			if (response_1 > 0.0f && response_0 < response_1 && response_2 < response_1) {
				const float response_1_0 = nv_cuda_keypoint_scale_diff(
					img_integral, img_integral_tilted, area_table, area_inv_table,
					row, col,
					NV_ROUND_INT(NV_MAT_V(outer_r, 0, s + 1)),
					NV_ROUND_INT(NV_MAT_V(inner_r, 0, s + 1)));
				const float response_1_2 = nv_cuda_keypoint_scale_diff(
					img_integral, img_integral_tilted, area_table, area_inv_table,
					row, col,
					NV_ROUND_INT(NV_MAT_V(outer_r, 0, s + 3)),
					NV_ROUND_INT(NV_MAT_V(inner_r, 0, s + 3)));

				__syncthreads();
				if (response_1 < response_1_0) {
					if (response_1_0 > response_1_2) {
						// response_1_0が最大
						NV_MAT3D_V(grid_response, s + 1, row_idx, col_idx) = response_1_0;
					} else {
						// response_1_2が最大
						NV_MAT3D_V(grid_response, s + 3, row_idx, col_idx) = response_1_2;
					}
				} else {
					if (response_1 > response_1_2) {
						// response_1が最大
						NV_MAT3D_V(grid_response, s + 2, row_idx, col_idx) = response_1;
					} else {
						// response_1_2が最大
						NV_MAT3D_V(grid_response, s + 3, row_idx, col_idx) = response_1_2;
					}
				}
			} else if (response_1 < 0.0f && response_0 > response_1 && response_2 > response_1) {
				const float response_1_0 = nv_cuda_keypoint_scale_diff(
					img_integral, img_integral_tilted, area_table, area_inv_table,
					row, col,
					NV_ROUND_INT(NV_MAT_V(outer_r, 0, s + 1)),
					NV_ROUND_INT(NV_MAT_V(inner_r, 0, s + 1)));
				const float response_1_2 = nv_cuda_keypoint_scale_diff(
					img_integral, img_integral_tilted, area_table, area_inv_table,
					row, col,
					NV_ROUND_INT(NV_MAT_V(outer_r, 0, s + 3)),
					NV_ROUND_INT(NV_MAT_V(inner_r, 0, s + 3)));

				__syncthreads();
				if (response_1 > response_1_0) {
					if (response_1_0 < response_1_2) {
						// response_1_0が最小
						NV_MAT3D_V(grid_response, s + 1, row_idx, col_idx) = response_1_0;
					} else {
						// response_1_2が最小
						NV_MAT3D_V(grid_response, s + 3, row_idx, col_idx) = response_1_2;
					}
				} else {
					if (response_1 < response_1_2) {
						// response_1が最小
						NV_MAT3D_V(grid_response, s + 2, row_idx, col_idx) = response_1;
					} else {
						// response_1_2が最小
						NV_MAT3D_V(grid_response, s + 3, row_idx, col_idx) = response_1_2;
					}
				}
			}
		}
	}
}

static __device__ int
nv_cuda_keypoint_edge_like(const nv_matrix_t *img, int offset, int step, int ky, int kx)
{
#if 0
	float dxs = 0.0f;
	float dys = 0.0f;
	float dxdys = 0.0f;
	int y, x;
	float trace;
	float det;
	
	for (y = ky - offset; y <= ky + offset; y += step) {
		for (x = kx - offset; x <= kx + offset; x += step) {
			float dx = NV_MAT_V(img, y, x - step) - NV_MAT_V(img, y, x + step);
			float dy = NV_MAT_V(img,  y - step, x) - NV_MAT_V(img, y + step, x);
			dxs += dx * dx;
			dys += dy * dy;
			dxdys += dx * dy;
		}
	}
	trace = dxs + dys;
	det = dxs * dys - dxdys * dxdys;
	return trace*trace/det >= NV_KEYPOINT_EDGE_THRESH;
#else
	return 0;
#endif
}

static __global__ void
nv_cuda_keypoint_edge_thresh(const int nkeypoint,
							 const nv_matrix_t *keypoints,
							 const nv_matrix_t *outer_r,
							 nv_matrix_t **memo)
{
	const int i = blockIdx.x;
	const int j = threadIdx.x;
	__shared__ float shm[NV_CUDA_KEYPOINT_RADIUS_MAX * 2 + 1][3];

	if (i < nkeypoint) {
		const int s = (int)NV_MAT_V(keypoints, i, NV_KEYPOINT_LEVEL_IDX);
		const int offset = (int)NV_MAT_V(outer_r, 0, s); // max 50
		const int n = (offset * 2 + 1);
		if (j < n) {
			const nv_matrix_t *img = memo[s];
			const int ky = (int)NV_MAT_V(keypoints, i, NV_KEYPOINT_Y_IDX);
			const int kx = (int)NV_MAT_V(keypoints, i, NV_KEYPOINT_X_IDX);
			const int y = ky - offset + j;
			int x;
			float dxs = 0.0f, dys = 0.0f, dxdys = 0.0f;
			for (x = kx - offset; x <= kx + offset; ++x) {
				float dx = NV_MAT_V(img, y, x - 1) - NV_MAT_V(img, y, x + 1);
				float dy = NV_MAT_V(img, y - 1, x) - NV_MAT_V(img, y + 1, x);
				dxs += dx * dx;
				dys += dy * dy;
				dxdys += dx * dy;
			}
			shm[j][0] = dxs;
			shm[j][1] = dys;
			shm[j][2] = dxdys;
			__syncthreads();
			if (j == 0) {
				int k;
				float trace, det;
				dxs = 0.0f, dys = 0.0f, dxdys = 0.0f;
				for (k = 0; k < n; ++k) {
					dxs += shm[k][0];
					dys += shm[k][1];
					dxdys += shm[k][2];
				}
				trace = dxs + dys;
				det = dxs * dys - dxdys * dxdys;
				if (trace * trace / det >= NV_KEYPOINT_EDGE_THRESH) {
					NV_MAT_V(keypoints, i, NV_KEYPOINT_RESPONSE_IDX) = 0.0f; // clear
				}
			}
		}
	}
}

/* キーポイントの選択 
*
* ある点を選択する条件は、
* 1. スケール空間でフィルタ応答が極値になっている.
* 2. 応答が閾値より強い(絶対値が大きい）．
* 3. "同じスケール"の近傍nnの円内で最大/最小のフィルタ応答を返している.
* となる.
* 計算量が多いので非選択条件はできるだけ早く適用する．
*/
static __global__ void
nv_cuda_keypoint_select(nv_matrix_t *keypoints,
						int *nkeypoint,
						int *lock_mem,
						const nv_matrix_t *grid_response,
						const nv_matrix_t *outer_r,
						const int img_rows,
						const int img_cols,
						nv_matrix_t **memo
	)
{
	const int my_m = blockDim.x * blockIdx.x + threadIdx.x;
	const int s = threadIdx.y + 1;
	const int col = (my_m % (img_cols / 2)) * 2;
	const int row = (my_m / (img_cols / 2)) * 2;
	const int offset = NV_ROUND_INT(NV_MAT_V(outer_r, 0, 0) * NV_KEYPOINT_DESC_SCALE) & ~1;
	const int erow = img_rows - offset;
	const int ecol = img_cols - offset;

	if (1 <= s && s < NV_KEYPOINT_LEVEL - 1 && 
		offset <= row && row < erow &&
		offset <= col && col < ecol) 
	{
		const int row_idx = row / 2;
		const int col_idx = col / 2;
		int  sy, sx, ey, ex;
		const float response = NV_MAT3D_V(grid_response, s, row_idx, col_idx);

		if ((fabsf(response) < NV_KEYPOINT_THRESH)) {
			/* 応答が閾値より小さい場合は選択しない. 
			* 極値でない応答は0が入っているのでここで弾かれる
			*/
			return;
		}

		/* 近傍 半径 * NV_KEYPOINT_NN */
		const int nn = NV_ROUND_INT(NV_MAT_V(outer_r, 0, s) * NV_KEYPOINT_NN);

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
			return;
		}
				
		if (response > 0.0f) {
			/* 最大値か */
			int y, x;
			for (y = sy; y < ey; y += 2) {
				const int y_idx = y / 2;
				const int cy = abs(row - y);
				const int cx = NV_ROUND_INT(sqrtf(nn * nn - cy * cy));
				int sx;

				sx = (col - cx);
				ex = (col + cx);

				for (x = sx; x < ex; x += 2) {
					if (response < NV_MAT3D_V(grid_response, s, y_idx, x / 2))
					{
						return;
					}
				}
			}
		} else if (response < 0.0f) {
			/* 最小値か */
			int y, x;
			for (y = sy; y < ey; y += 2) {
				const int y_idx = y / 2;
				const int cy = abs(row - y);
				const int cx = NV_ROUND_INT(sqrtf(nn * nn - cy * cy));
				int sx;

				sx = (col - cx);
				ex = (col + cx);

				for (x = sx; x < ex; x += 2) {
					if (response > NV_MAT3D_V(grid_response, s, y_idx, x / 2)) 
					{
						return;
					}
				}
			}
		}
		/* 特徴点選択 */
		bool needlock = true;
		while (needlock) {
			if (atomicCAS(lock_mem, 0, 1) == 0) {
				/* critical section */
				NV_MAT_V(keypoints, *nkeypoint, NV_KEYPOINT_RESPONSE_IDX) = response;
				NV_MAT_V(keypoints, *nkeypoint, NV_KEYPOINT_Y_IDX) = (float)(row);
				NV_MAT_V(keypoints, *nkeypoint, NV_KEYPOINT_X_IDX) = (float)(col);
				NV_MAT_V(keypoints, *nkeypoint, NV_KEYPOINT_RADIUS_IDX) = NV_MAT_V(outer_r, 0, s);
				NV_MAT_V(keypoints, *nkeypoint, NV_KEYPOINT_LEVEL_IDX) = (float)s;
				++*nkeypoint;

				atomicExch(lock_mem, 0);
				needlock = false;
			}
		}
	}
}

typedef struct {
	int i0;
	float v0;
	int i1;
	float v1;
} nv_cuda_keypoint_histdata_t;

/* 勾配ヒストグラムのためのデータを作成 */
static inline __device__ void
nv_cuda_keypoint_histdata(nv_cuda_keypoint_histdata_t *histdata,
						  const int hist_n, 
						  const int hist_i,
						  const int yi, const int xi,
						  const int ky, const int kx, 
						  float f_r, float angle,
						  const nv_matrix_t *  memo

	)
{
	const int r = NV_ROUND_INT(f_r);
	const int star_r = NV_KEYPOINT_STAR_R(f_r);
	const int star_r2 = star_r * 2;
	const int star_tilted_r = NV_ROUND_INT((float)star_r * NV_SQRT2_INV);
	const int n = NV_KEYPOINT_HIST_SAMPLE;
	const int sy = (ky - r) + star_r2;
	const int ey = (ky + r) - star_r2;
	const int sx = (kx - r) + star_r2;
	const int ex = (kx + r) - star_r2;
	const float step_scale = (ex - sx) / (float)n;
	const float r2 = (float)((ex - sx) / 2) * ((ex - sx) / 2);
	const int angle45 = NV_ROUND_INT(hist_n / 360.0f * 45.0f);

	/* 特徴点のr近傍から勾配ヒストグラムを作成する. */
	const float yp = ((float)sy + step_scale * yi);
	int y = NV_ROUND_INT(yp);
	const float yd = ((float)ky - y) * ((float)ky - y);

	if (y >= ey) {
		y = ey - 1;
	}
	const int x = NV_ROUND_INT(((float)sx + step_scale * xi));

	if (x <= ex) {
		const float fdist = yd + ((float)kx - x) * ((float)kx - x);
		if (fdist <= r2) {
			float dx[2], dy[2];
			float magnitude[2], theta[2];
			int bin[2];
			const int dist = NV_ROUND_INT(sqrtf(fdist));
			const float g = ((float)dist / (r + 1));
			const float w = expf(-(g * g) / (2.0f * 0.2f));
			nv_cuda_keypoint_histdata_t *p = &histdata[hist_i];

			/* (x, y)を中心とした■◆を重ねた8つの頂点から勾配の方向と強さを求める.
			* イラストなどは局所的な変化が激しいので8点から勾配を求め平均する.
			*/
			dx[0] = NV_MAT_V(memo, y, x + star_r) - NV_MAT_V(memo, y, x - star_r);
			dy[0] = NV_MAT_V(memo, y + star_r, x) - NV_MAT_V(memo, y - star_r, x);
			magnitude[0] = sqrtf(dx[0] * dx[0] + dy[0] * dy[0]);
			theta[0] = atan2f(dy[0], dx[0]) + NV_PI;
			theta[0] -= angle;
			if (theta[0] < 0.0f) {
				theta[0] = NV_PI * 2.0f + theta[0];
			}
			bin[0] = NV_ROUND_INT((float)hist_n * theta[0] * NV_PI2_INV);
			if (bin[0] >= hist_n) {
				bin[0] -= hist_n;
			}
			p->i0 = bin[0];
			p->v0 = magnitude[0] * w;

			dx[1] = NV_MAT_V(memo, y + star_tilted_r, x + star_tilted_r)
				- NV_MAT_V(memo, y - star_tilted_r, x - star_tilted_r);
			dy[1] = NV_MAT_V(memo, y + star_tilted_r, x - star_tilted_r)
				- NV_MAT_V(memo, y - star_tilted_r, x + star_tilted_r);
			
			magnitude[1] = sqrtf(dx[1] * dx[1] + dy[1] * dy[1]);
			theta[1] = atan2f(dy[1], dx[1]) + NV_PI;
			theta[1] -= angle;
			if (theta[1] < 0.0f) {
				theta[1] = NV_PI * 2.0f + theta[1];
			}
			bin[1] = NV_ROUND_INT((float)hist_n * theta[1] * NV_PI2_INV) + angle45;
			if (bin[1] >= hist_n) {
				bin[1] -= hist_n;
			}
			p->i1 = bin[1];
			p->v1 = magnitude[1] * w;
		}
	}
}

static __global__ void
nv_cuda_keypoint_orientation_histdata(int nkeypoint,
									   nv_matrix_t *keypoints,
									   nv_matrix_t **memo,
									   nv_cuda_keypoint_histdata_t *hist

	)
{
	const int i = blockIdx.x;
	const int yi = threadIdx.y;
	const int xi = threadIdx.x;
	const int hist_i = (i * NV_KEYPOINT_HIST_SAMPLE * NV_KEYPOINT_HIST_SAMPLE) + yi * NV_KEYPOINT_HIST_SAMPLE + xi;

	if (i < nkeypoint 
		&& xi < NV_KEYPOINT_HIST_SAMPLE 
		&& yi < NV_KEYPOINT_HIST_SAMPLE) 
	{
		hist[hist_i].i0 = -1; // deny flag

		nv_cuda_keypoint_histdata(
			hist, NV_KEYPOINT_ORIENTATION_HIST,
			hist_i,	yi, xi,
			NV_ROUND_INT(NV_MAT_V(keypoints, i, NV_KEYPOINT_Y_IDX)),
			NV_ROUND_INT(NV_MAT_V(keypoints, i, NV_KEYPOINT_X_IDX)),
			NV_MAT_V(keypoints, i, NV_KEYPOINT_RADIUS_IDX), 0.0f,
			memo[(int)NV_MAT_V(keypoints, i, NV_KEYPOINT_LEVEL_IDX)]);
	}
}

static __global__ void
nv_cuda_keypoint_orientation(int nkeypoint,
							 nv_matrix_t *keypoints,
							 nv_cuda_keypoint_histdata_t *histdata
							 )
{
	const int i = blockIdx.x;
	const int k = threadIdx.x;
	__shared__ float temp[NV_KEYPOINT_HIST_SAMPLE][NV_KEYPOINT_ORIENTATION_HIST];

	if (i < nkeypoint && k < NV_KEYPOINT_HIST_SAMPLE) {
		const int hist_i = i * NV_KEYPOINT_HIST_SAMPLE * NV_KEYPOINT_HIST_SAMPLE;
		float *hist = &temp[k][0];
		int j, l;
		const int j_offset_i = hist_i + k * NV_KEYPOINT_HIST_SAMPLE;
#pragma unroll
		for (j = 0; j < NV_KEYPOINT_ORIENTATION_HIST; ++j) {
			hist[j] = 0.0f;
		}
#pragma unroll
		for (j = 0; j < NV_KEYPOINT_HIST_SAMPLE; ++j) {
			const nv_cuda_keypoint_histdata_t *p = &histdata[j_offset_i + j];
			if (p->i0 >= 0) {
				hist[p->i0] += p->v0;
				hist[p->i1] += p->v1;
			}
		}
		__syncthreads();
		if (k == 0) {
			int max_n = -1;
			float max_v = -FLT_MAX;
			for (j = 1; j < NV_KEYPOINT_HIST_SAMPLE; ++j) {
				for (l = 0; l < NV_KEYPOINT_ORIENTATION_HIST; ++l) {
					temp[0][l] += temp[j][l];
				}
			}
#pragma unroll
			for (j = 0; j < NV_KEYPOINT_ORIENTATION_HIST; ++j) {
				float v = temp[0][j];
				if (max_v < v) {
					max_v = v;
					max_n = j;
				}
			}
			NV_MAT_V(keypoints, i, NV_KEYPOINT_ORIENTATION_IDX) = ((2.0f * NV_PI) / (float)NV_KEYPOINT_ORIENTATION_HIST) * max_n - NV_PI;
		}
	}
}



/*
 * 特徴点の特徴ベクトルを算出する.
 * 特徴ベクトルは, 特徴点を中心とした半径desc_rの円上の
 * 8点(PI/4ごと)°を中心とした
 * 半径desc_rの円内の勾配ヒストグラム(8bin)とする.
 * つまり, 8 * 8 = 64次元の特徴ベクトルとなる.
*/
static __constant__ float g_circle_steps[8] = {
	0.0f,
	NV_PI / 4.0f * 1.0f, NV_PI / 4.0f * 2.0f,
	NV_PI / 4.0f * 3.0f, NV_PI / 4.0f * 4.0f,
	NV_PI / 4.0f * 5.0f, NV_PI / 4.0f * 6.0f,
	NV_PI / 4.0f * 7.0f
};
static __global__ void 
nv_cuda_keypoint_desc_histdata(int nkeypoint,
							   const nv_matrix_t *keypoints,
							   nv_matrix_t **memo,
							   nv_cuda_keypoint_histdata_t *histdata)
{
	const int i = blockIdx.x; // keypoints
	const int k = blockIdx.y; // sub descs
	const int xi = threadIdx.x; // sample
	const int yi = threadIdx.y; // sample
	const int hist_i = (i * NV_KEYPOINT_HIST_SAMPLE * NV_KEYPOINT_HIST_SAMPLE * NV_CUDA_KEYPOINT_DESC_M) 
		+ (k * NV_KEYPOINT_HIST_SAMPLE * NV_KEYPOINT_HIST_SAMPLE) + yi * NV_KEYPOINT_HIST_SAMPLE + xi;

	if (i < nkeypoint) {
		const float desc_r = NV_MAT_V(keypoints, i, NV_KEYPOINT_RADIUS_IDX);
		const float angle = NV_PI + NV_MAT_V(keypoints, i, NV_KEYPOINT_ORIENTATION_IDX);

		histdata[hist_i].i0 = -1;

		if (k == NV_CUDA_KEYPOINT_DESC_M - 1) {
			nv_cuda_keypoint_histdata(
				histdata, NV_CUDA_KEYPOINT_HIST_DIR,
				hist_i,
				yi, xi,
				NV_MAT_V(keypoints, i, NV_KEYPOINT_Y_IDX),
				NV_MAT_V(keypoints, i, NV_KEYPOINT_X_IDX),
				desc_r, angle,
				memo[(int)NV_MAT_V(keypoints, i, NV_KEYPOINT_LEVEL_IDX)]
			);
		} else if (k < NV_CUDA_KEYPOINT_DESC_M) {
			float theta = g_circle_steps[k] + angle;
			if (theta > 2.0f * NV_PI) {
				theta = (theta - 2.0f * NV_PI);
			}
			/* 勾配ヒストグラムを求める点に移動 */
			nv_cuda_keypoint_histdata(
				histdata, NV_CUDA_KEYPOINT_HIST_DIR,
				hist_i,
				yi, xi,
				NV_ROUND_INT(desc_r * sinf(theta) + NV_MAT_V(keypoints, i, NV_KEYPOINT_Y_IDX)),
				NV_ROUND_INT(desc_r * cosf(theta) + NV_MAT_V(keypoints, i, NV_KEYPOINT_X_IDX)),
				desc_r, angle,
				memo[(int)NV_MAT_V(keypoints, i, NV_KEYPOINT_LEVEL_IDX)]
			);
		}
	}
}

static __global__ void 
nv_cuda_keypoint_desc(int nkeypoint,
					  nv_matrix_t *desc, 
					  nv_cuda_keypoint_histdata_t *histdata
					  )

{
	__shared__ float temp[NV_KEYPOINT_HIST_SAMPLE][NV_CUDA_KEYPOINT_HIST_DIR];
	__shared__ float scale;
	const int i = blockIdx.x; // keypoint
	const int k = blockIdx.y; // sub desc
	const int l = threadIdx.x; // hist samples

	if (i < nkeypoint && l < NV_KEYPOINT_HIST_SAMPLE && k < NV_CUDA_KEYPOINT_DESC_M) {
		float *hist = &temp[l][0];
		int j;
		const int hist_i = (i * NV_KEYPOINT_HIST_SAMPLE * NV_KEYPOINT_HIST_SAMPLE * NV_CUDA_KEYPOINT_DESC_M) 
			+ (k * NV_KEYPOINT_HIST_SAMPLE * NV_KEYPOINT_HIST_SAMPLE);
		const int j_offset_i = hist_i + l * NV_KEYPOINT_HIST_SAMPLE;

#pragma unroll
		for (j = 0; j < NV_CUDA_KEYPOINT_HIST_DIR; ++j) {
			hist[j] = 0.0f;
		}
#pragma unroll
		for (j = 0; j < NV_KEYPOINT_HIST_SAMPLE; ++j) {
			const nv_cuda_keypoint_histdata_t *p = &histdata[j_offset_i + j];
			if (0 <= p->i0) {
				hist[p->i0] += p->v0;
				hist[p->i1] += p->v1;
			}
		}
		if (l < NV_CUDA_KEYPOINT_HIST_DIR) {
			__syncthreads();
#pragma unroll
			for (j = 1; j < NV_KEYPOINT_HIST_SAMPLE; ++j) {
				temp[0][l] += temp[j][l];
			}
			__syncthreads();

			// vector_normalize
			if (l == 0) {
				float v = 0.0f;
				float norm;
#pragma unroll
				for (j = 0; j < NV_CUDA_KEYPOINT_HIST_DIR; ++j) {
					v += hist[j] * hist[j];
				}
				norm = sqrtf(v);
				if (norm != 0.0f) {
					scale = 1.0f / norm;
				} else {
					scale = 0.0f;
				}
			}
			__syncthreads();
			NV_MAT_V(desc, i, k * NV_CUDA_KEYPOINT_HIST_DIR + l) = temp[0][l] * scale;
		}
	}
}

/*
 * スケール空間のフィルタサイズを計算する.
 *
 * 外側の半径は内側の半径の2倍とする.
 * 
 */
static __host__ void 
nv_cuda_keypoint_radius(nv_matrix_t *outer_r, nv_matrix_t *inner_r)
{
	float cur_r = NV_KEYPOINT_MIN_R;
	int prev_r = NV_KEYPOINT_MIN_R - 1;
	int s;
	int r;

	r = NV_ROUND_INT(NV_KEYPOINT_MIN_R);
	NV_MAT_V(inner_r, 0, 0) = (float)r;
	NV_MAT_V(outer_r, 0, 0) = (r * 2.0f);

	for (s = 1; s < NV_KEYPOINT_LEVEL;) {
		cur_r *= NV_KEYPOINT_SCALE_FACTOR;
		r = NV_ROUND_INT(cur_r);
		if (r - prev_r > 1) {
			prev_r = r;
			NV_MAT_V(inner_r, 0, s) = (float)r;
			NV_MAT_V(outer_r, 0, s) = (r * 2.0f);

			++s;
		}
	}
}

static __host__ int
nv_cuda_keypoint_desc_cmp(const void *p1, const void *p2)
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
	/* 応答の絶対値でソート 
	   TODO: 
	*/
	ab1 = fabsf(f1[NV_KEYPOINT_RESPONSE_IDX]);
	ab2 = fabsf(f2[NV_KEYPOINT_RESPONSE_IDX]);
	if (ab1 < ab2) {
		return 1;
	} else if (ab1 > ab2) {
		return -1;
	}

	return 0;
}


int
nv_cuda_keypoint(nv_matrix_t *keypoints,
				 nv_matrix_t *desc,
				 const nv_matrix_t *img,
				 const int channel)
{
	if (img->rows < 16 || img->cols < 16) {
		return 0;
	}
	int i;
	nv_matrix_t *integral = nv_matrix3d_alloc(1, img->rows + 1, img->cols + 1);
	nv_matrix_t *integral_tilted = nv_matrix3d_alloc(1, img->rows + 1, img->cols + 1);
	const int img_rows = integral->rows - 1;
	const int img_cols = integral->cols - 1;
	int nkeypoint = 0;
	nv_matrix_t **memo = NULL;
	nv_matrix_t *inner_r = nv_matrix_alloc(NV_KEYPOINT_LEVEL, 1);
	nv_matrix_t *outer_r = nv_matrix_alloc(NV_KEYPOINT_LEVEL, 1);
	nv_matrix_t *keypoints_tmp = nv_matrix_alloc(keypoints->n, (img_rows / 2) * (img_cols / 2));
	nv_matrix_t **memo_dev = NULL;
	nv_matrix_t *inner_r_dev = NULL;
	nv_matrix_t *outer_r_dev = NULL;
	nv_matrix_t *integral_dev = NULL;
	nv_matrix_t *integral_tilted_dev = NULL;
	float *area_inv_table_dev, *area_table_dev = NULL;
	nv_matrix_t *grid_response_dev = NULL;
	nv_matrix_t *scale_response_dev = NULL;;
	int *nkeypoint_dev = NULL;
	nv_matrix_t *keypoints_tmp_dev = nv_cuda_matrix_clone(keypoints_tmp);
	nv_matrix_t *keypoints_dev = NULL;
	nv_matrix_t *desc_dev = NULL;
	nv_cuda_keypoint_histdata_t *histdata_dev = NULL;

	int *lock_mem_dev;

	long t = nv_clock();
	int max_r;

	/* */
	NV_ASSERT(desc->n == 72);

	nv_cuda_keypoint_radius(outer_r, inner_r);
	nv_integral(integral, img, channel);
	nv_integral_tilted(integral_tilted, img, channel);
  	t = nv_clock();

	/* 入力データ作成 */ 
	inner_r_dev = nv_cuda_matrix_dup(inner_r);
	outer_r_dev = nv_cuda_matrix_dup(outer_r);
	integral_dev = nv_cuda_matrix_dup(integral);
	integral_tilted_dev = nv_cuda_matrix_dup(integral_tilted);

	scale_response_dev = nv_cuda_matrix_alloc(NV_KEYPOINT_LEVEL, (img_cols / 2) * (img_rows / 2));
	max_r = NV_ROUND_INT(NV_MAT_V(outer_r, 0, NV_KEYPOINT_LEVEL-1)) + 4;

	memo = nv_alloc_type(nv_matrix_t *, NV_KEYPOINT_LEVEL);
	for (i = 0; i < NV_KEYPOINT_LEVEL; ++i) {
		nv_matrix_t *memobuf = nv_cuda_matrix_alloc(integral->cols, integral->rows);
		memo[i] = memobuf;
	}

	CUDA_SAFE_CALL(cudaMalloc(&memo_dev, sizeof(nv_matrix_t *) * NV_KEYPOINT_LEVEL));
	CUDA_SAFE_CALL(
		cudaMemcpy(memo_dev,
		memo,
		sizeof(nv_matrix_t *) * NV_KEYPOINT_LEVEL, 
		cudaMemcpyHostToDevice));

	CUDA_SAFE_CALL(
		cudaMalloc(&area_inv_table_dev,
		sizeof(float) * max_r));
	CUDA_SAFE_CALL(
		cudaMemcpy(area_inv_table_dev, 
		nv_star_integral_area_inv_static, sizeof(float) * max_r, 
		cudaMemcpyHostToDevice));

	CUDA_SAFE_CALL(
		cudaMalloc(&area_table_dev,
		sizeof(float) * max_r));
	CUDA_SAFE_CALL(
		cudaMemcpy(area_table_dev, 
		nv_star_integral_area_static, sizeof(float) * max_r, 
		cudaMemcpyHostToDevice));

	grid_response_dev = nv_cuda_matrix3d_alloc_zero(
		img_cols / 2,
		NV_KEYPOINT_LEVEL,
		img_rows / 2);

	CUDA_SAFE_CALL(cudaMalloc(&nkeypoint_dev, sizeof(int)));
	CUDA_SAFE_CALL(cudaMemset(nkeypoint_dev, 0, sizeof(int)));
	CUDA_SAFE_CALL(cudaMalloc(&lock_mem_dev, sizeof(int)));
	CUDA_SAFE_CALL(cudaMemset(lock_mem_dev, 0, sizeof(int)));

	/* detect*/
	{
		int m = (img_cols / 2) * (img_rows / 2);
		dim3 blocks(nv_cuda_block(m));
		dim3 threads(nv_cuda_thread(m));

		nv_cuda_keypoint_scale_search<<<blocks, threads>>>(
			grid_response_dev,
			scale_response_dev,
			area_table_dev,
			area_inv_table_dev,
			integral_dev, integral_tilted_dev, outer_r_dev, inner_r_dev, img_rows, img_cols);
		CUT_CHECK_ERROR("nv_cuda_keypoint_scale_search() failed\n");
	}

#if BENCHMARK
	printf("- detect : %ldms\n", nv_clock() - t);
	t = nv_clock();
#endif

	/* run make_scale_space kernel */
	{
		int m = integral->cols * integral->rows;
		dim3 blocks(nv_cuda_block(m), NV_KEYPOINT_LEVEL);
		dim3 threads(nv_cuda_thread(m));
	
		nv_cuda_keypoint_make_scale_space<<<blocks, threads>>>(
			memo_dev,
			area_inv_table_dev,
			outer_r_dev,
			inner_r_dev,
			integral_dev,
			integral_tilted_dev);
		CUT_CHECK_ERROR("nv_cuda_keypoint_make_scale_space() failed\n");

#if BENCHMARK
		printf("- make_scale_space : %ldms\n", nv_clock() - t);
		t = nv_clock();
#endif
	}
	/* select */
	{
		int s = NV_KEYPOINT_LEVEL - 2; // s += 1; 16 <= s
		int m = (img_cols / 2) * (img_rows / 2);
		dim3 select_blocks(m / 8 + (m % 8 == 0 ? 0:1));
		dim3 select_threads(8, s);
		float *tmp;

		nv_cuda_keypoint_select<<<select_blocks, select_threads>>>(
			keypoints_tmp_dev,
			nkeypoint_dev,
			lock_mem_dev,
			grid_response_dev,
			outer_r_dev,
			img_rows,
			img_cols,
			memo_dev
			);
		CUT_CHECK_ERROR("nv_cuda_keypoint_select() failed\n");
		// sync
		CUDA_SAFE_CALL(cudaMemcpy(&nkeypoint, nkeypoint_dev, sizeof(int), cudaMemcpyDeviceToHost));

		nv_cuda_keypoint_edge_thresh<<<dim3(nkeypoint), dim3(NV_CUDA_KEYPOINT_RADIUS_MAX * 2 + 1)>>>(
			nkeypoint,
			keypoints_tmp_dev,
			outer_r_dev,
			memo_dev);

		tmp = keypoints_tmp->v;
		CUDA_SAFE_CALL(cudaMemcpy(keypoints_tmp, keypoints_tmp_dev, 
			sizeof(nv_matrix_t), cudaMemcpyDeviceToHost));
		CUDA_SAFE_CALL(cudaMemcpy(tmp, keypoints_tmp->v, 
			nkeypoint * keypoints_tmp->step * sizeof(float), cudaMemcpyDeviceToHost));
		keypoints_tmp->v = tmp;
		{
			int i, j = 0;
			for (i = 0; i < nkeypoint; ++i) {
				if (fabsf(NV_MAT_V(keypoints_tmp, i, NV_KEYPOINT_RESPONSE_IDX)) > NV_KEYPOINT_THRESH) {
					if (i != j) {
						nv_vector_copy(keypoints_tmp, j, keypoints_tmp, i);
					}
					++j;
				}
			}
			nkeypoint = j;
		}
		qsort(keypoints_tmp->v, nkeypoint, keypoints_tmp->step * sizeof(float), nv_cuda_keypoint_desc_cmp);
		nkeypoint = NV_MIN(keypoints->m, nkeypoint);
		nv_matrix_m(keypoints_tmp, nkeypoint);

		keypoints_dev = nv_cuda_matrix_dup(keypoints_tmp);
#if BENCHMARK
		printf("- select : %ldms\n", nv_clock() - t);
		t = nv_clock();
#endif
	}
	/* orientation */
	{
		dim3 histdata_blocks(NV_MAX(nkeypoint, 1));
		dim3 histdata_threads(NV_KEYPOINT_HIST_SAMPLE, NV_KEYPOINT_HIST_SAMPLE);
		dim3 orientation_blocks(NV_MAX(nkeypoint, 1));
		dim3 orientation_threads(NV_KEYPOINT_HIST_SAMPLE);

		float *tmp;	

		CUDA_SAFE_CALL(cudaMalloc(&histdata_dev,
			sizeof(nv_cuda_keypoint_histdata_t) * nkeypoint * NV_KEYPOINT_HIST_SAMPLE * NV_KEYPOINT_HIST_SAMPLE));

		nv_cuda_keypoint_orientation_histdata<<<histdata_blocks, histdata_threads>>>(
			nkeypoint,
			keypoints_dev,
			memo_dev,
			histdata_dev
			);
		CUT_CHECK_ERROR("nv_cuda_keypoint_orientation_histdata() failed\n");
		nv_cuda_keypoint_orientation<<<orientation_blocks, orientation_threads>>>(
			nkeypoint,
			keypoints_dev,
			histdata_dev
			);
		CUT_CHECK_ERROR("nv_cuda_keypoint_orientation() failed\n");
		tmp = keypoints->v;
		CUDA_SAFE_CALL(cudaMemcpy(keypoints, keypoints_dev, 
			sizeof(nv_matrix_t), cudaMemcpyDeviceToHost));
		CUDA_SAFE_CALL(cudaMemcpy(tmp, keypoints->v, 
			nkeypoint * keypoints->step * sizeof(float), cudaMemcpyDeviceToHost));
		keypoints->v = tmp;
#if BENCHMARK
		printf("- orientation : %ldms\n", nv_clock() - t);
		t = nv_clock();
#endif
	}
	{
		dim3 histdata_blocks(NV_MAX(nkeypoint, 1), NV_CUDA_KEYPOINT_DESC_M);
		dim3 histdata_threads(NV_KEYPOINT_HIST_SAMPLE, NV_KEYPOINT_HIST_SAMPLE);
		dim3 desc_blocks(NV_MAX(nkeypoint, 1), NV_CUDA_KEYPOINT_DESC_M);
		dim3 desc_threads(NV_KEYPOINT_HIST_SAMPLE);
		float *tmp;

		cudaFree(histdata_dev);
		CUDA_SAFE_CALL(cudaMalloc(&histdata_dev,
			sizeof(nv_cuda_keypoint_histdata_t) * nkeypoint 
			* NV_KEYPOINT_HIST_SAMPLE * NV_KEYPOINT_HIST_SAMPLE * NV_CUDA_KEYPOINT_DESC_M));
		desc_dev = nv_cuda_matrix_alloc_zero(desc->n, nkeypoint);

		nv_cuda_keypoint_desc_histdata<<<histdata_blocks, histdata_threads>>>(
			nkeypoint,
			keypoints_dev,
			memo_dev,
			histdata_dev
			);
		//CUT_CHECK_ERROR("nv_cuda_keypoint_desc_histdata() failed\n");
		nv_cuda_keypoint_desc<<<desc_blocks, desc_threads>>>(
			nkeypoint,
			desc_dev,
			histdata_dev);
		//CUT_CHECK_ERROR("nv_cuda_keypoint_desc() failed\n");
		tmp = desc->v;
		CUDA_SAFE_CALL(cudaMemcpy(desc, desc_dev, 
			sizeof(nv_matrix_t), cudaMemcpyDeviceToHost));
		CUDA_SAFE_CALL(cudaMemcpy(tmp, desc->v, 
			nkeypoint * desc->step * sizeof(float), cudaMemcpyDeviceToHost));
		desc->v = tmp;
#if BENCHMARK
		printf("- desc : %ldms\n", nv_clock() - t);
		t = nv_clock();
#endif
	}
	nv_cuda_matrix_free(inner_r_dev);
	nv_cuda_matrix_free(outer_r_dev);
	nv_cuda_matrix_free(integral_dev);
	nv_cuda_matrix_free(integral_tilted_dev);
	nv_cuda_matrix_free(scale_response_dev);
	nv_cuda_matrix_free(keypoints_tmp_dev);
	nv_cuda_matrix_free(keypoints_dev);
	nv_cuda_matrix_free(grid_response_dev);
	nv_cuda_matrix_free(desc_dev);

	for (i = 0; i < NV_KEYPOINT_LEVEL; ++i) {
		nv_cuda_matrix_free(memo[i]);
	}
	cudaFree(memo_dev);
	cudaFree(nkeypoint_dev);
	cudaFree(lock_mem_dev);
	cudaFree(area_inv_table_dev);
	cudaFree(area_table_dev);
	cudaFree(histdata_dev);

	nv_matrix_free(&keypoints_tmp);
	nv_matrix_free(&outer_r);
	nv_matrix_free(&inner_r);
	nv_matrix_free(&integral);
	nv_matrix_free(&integral_tilted);
	nv_free(memo);

	return nkeypoint;
}