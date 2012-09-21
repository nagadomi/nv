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

#ifndef NV_IP_KEYPOINT_H
#define NV_IP_KEYPOINT_H
#include "nv_core.h"
#include "nv_ip_integral.h"

#ifdef __cplusplus
extern "C" {
#endif

#define NV_KEYPOINT_MIN_R        6.168850f/* 探索開始の半径 */
#define NV_KEYPOINT_LEVEL        17       /* 探索する半径の階層数 */
#define NV_KEYPOINT_THRESH       18.0f    /* 検出する応答の閾値 */
#define NV_KEYPOINT_EDGE_THRESH  12.0f    /* エッジ判定の閾値 */

/* 近傍の最大/最小値を判定する半径のスケール */
#define NV_KEYPOINT_NN              0.5f
#define NV_KEYPOINT_DESC_N          72   /* 記述子の次元 */
#define NV_KEYPOINT_KEYPOINT_N      6    /* キーポイントの要素数 */

/* 添え字 */
#define NV_KEYPOINT_RESPONSE_IDX    0
#define NV_KEYPOINT_Y_IDX           1
#define NV_KEYPOINT_X_IDX           2
#define NV_KEYPOINT_RADIUS_IDX      3
#define NV_KEYPOINT_LEVEL_IDX       4
#define NV_KEYPOINT_ORIENTATION_IDX 5

typedef enum {
	NV_KEYPOINT_DESCRIPTOR_GRADIENT_HISTOGRAM,
	NV_KEYPOINT_DESCRIPTOR_RECTANGLE_FEATURE
} nv_keypoint_descriptor_e;

typedef enum {
	NV_KEYPOINT_DETECTOR_STAR
} nv_keypoint_detector_e;

typedef struct {
	float star_th;
	float edge_th;
	float min_r;
	int level;
	float nn;
	nv_keypoint_detector_e detector;
	nv_keypoint_descriptor_e descriptor;
} nv_keypoint_param_t;
typedef struct nv_keypoint_ctx nv_keypoint_ctx_t;
nv_keypoint_ctx_t *nv_keypoint_ctx_alloc(const nv_keypoint_param_t *param);
void nv_keypoint_ctx_free(nv_keypoint_ctx_t **ctx);

const nv_keypoint_param_t *nv_keypoint_param_gradient_histogram_default(void);
const nv_keypoint_param_t *nv_keypoint_param_rectangle_feature_default(void);
#define NV_KEYPOINT_PARAM_DEFAULT nv_keypoint_param_gradient_histogram_default()

int nv_keypoint(nv_matrix_t *keypoints,
				nv_matrix_t *desc,
				const nv_matrix_t *img,
				const int channel);

int nv_keypoint_ex(const nv_keypoint_ctx_t *ctx,
				   nv_matrix_t *keypoints,
				   nv_matrix_t *desc,
				   const nv_matrix_t *img,
				   const int channel);

typedef int (*nv_cuda_keypoint_t)(nv_matrix_t *keypoints,
								  nv_matrix_t *desc,
								  const nv_matrix_t *img,
								  const int channel);

extern NV_DECLARE_DATA nv_cuda_keypoint_t nv_keypoint_gpu;

typedef struct {
	int r;
	int rows;
	int cols;
} nv_keypoint_dense_t;

int nv_keypoint_dense(nv_matrix_t *keypoints,
					  nv_matrix_t *desc,
					  const nv_matrix_t *img,
					  const int channel,
					  const nv_keypoint_dense_t *dence,
					  int n_dence
	);
int nv_keypoint_dense_ex(const nv_keypoint_ctx_t *ctx,
						 nv_matrix_t *keypoints,
						 nv_matrix_t *desc,
						 const nv_matrix_t *img,
						 const int channel,
						 const nv_keypoint_dense_t *dense,
						 int n_dense
	);

#ifdef __cplusplus
}
#endif
#endif
