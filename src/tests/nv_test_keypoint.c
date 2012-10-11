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

#undef NDEBUG
#include "nv_core.h"
#include "nv_io.h"
#include "nv_ip.h"
#include "nv_test.h"

#define RESIZE 512.0f
#define KEYPOINT_MAX 2000

static int
nv_test_keypoint_extract(nv_matrix_t *key_vec,
						 nv_matrix_t *desc_vec,
						 const nv_matrix_t *image,
						 const nv_keypoint_param_t *param
	)
{
	int n;
	float scale = RESIZE / (float)NV_MAX(image->rows, image->cols);
	nv_matrix_t *resize = nv_matrix3d_alloc(image->n, (int)(image->rows * scale),
											(int)(image->cols * scale));
	nv_matrix_t *gray = nv_matrix3d_alloc(1, resize->rows, resize->cols);
	nv_matrix_t *smooth = nv_matrix3d_alloc(1, resize->rows, resize->cols);
	nv_keypoint_ctx_t *ctx = nv_keypoint_ctx_alloc(param);
	
	nv_resize(resize, image);
	nv_gray(gray, resize);
	nv_gaussian5x5(smooth, 0, gray, 0);
	n = nv_keypoint_ex(ctx, key_vec, desc_vec, smooth, 0);
	
	nv_matrix_free(&gray);
	nv_matrix_free(&resize);
	nv_matrix_free(&smooth);
	nv_keypoint_ctx_free(&ctx);

	return n;
}

static float
nv_test_keypoint_match(const nv_keypoint_param_t *param, const char *test_img1, const char *test_img2)
{
	nv_matrix_t *image1 = nv_load_image(test_img1);
	nv_matrix_t *image2 = nv_load_image(test_img2);
	nv_matrix_t *key_vec1 = nv_matrix_alloc(NV_KEYPOINT_KEYPOINT_N, KEYPOINT_MAX);
	nv_matrix_t *desc_vec1 = nv_matrix_alloc(NV_KEYPOINT_DESC_N, KEYPOINT_MAX);
	nv_matrix_t *key_vec2 = nv_matrix_alloc(NV_KEYPOINT_KEYPOINT_N, KEYPOINT_MAX);
	nv_matrix_t *desc_vec2 = nv_matrix_alloc(NV_KEYPOINT_DESC_N, KEYPOINT_MAX);
	
	nv_knn_result_t knn[2];
	int n1, n2;
	int i;
	int ok = 0;
	long t;

	NV_TEST_NAME;

	NV_ASSERT(image1 != NULL);
	NV_ASSERT(image2 != NULL);

	t = nv_clock();
	n1 = nv_test_keypoint_extract(key_vec1, desc_vec1, image1, param);
	printf("%s: %dpts %ldms\n", test_img1, n1, nv_clock() - t);
	t = nv_clock();
	n2 = nv_test_keypoint_extract(key_vec2, desc_vec2, image2, param);
	printf("%s: %dpts %ldms\n", test_img2, n2, nv_clock() - t);
	
	nv_matrix_m(key_vec1, n1);
	nv_matrix_m(desc_vec1, n1);
	nv_matrix_m(key_vec2, n2);
	nv_matrix_m(desc_vec2, n2);

	//nv_matrix_print(stdout, desc_vec1);
	
	t = nv_clock();
	for (i = 0; i < key_vec1->m; ++i) {
		float d1;
		int j1;
		
		nv_knn_ex(knn, 2, desc_vec2, desc_vec1, i, nv_cosine);
		j1 = knn[0].index;
		d1 = knn[0].dist;
		//printf("%f %f\n", d1, d2);
		if (nv_sign(NV_MAT_V(key_vec1, i, NV_KEYPOINT_RESPONSE_IDX))
			!= nv_sign(NV_MAT_V(key_vec2, j1, NV_KEYPOINT_RESPONSE_IDX)))
		{
			continue;
		}
		if (d1 < 0.05f) {
			// j1-j2 match
			++ok;
		}
	}
	printf("knn match: %f(%dpts/%dpts) %ldms\n",
		   (float)ok/NV_MIN(n1, n2), ok, NV_MIN(n1, n2),
		   nv_clock() - t
		);

	nv_matrix_free(&image1);
	nv_matrix_free(&key_vec1);
	nv_matrix_free(&desc_vec1);
	nv_matrix_free(&image2);
	nv_matrix_free(&key_vec2);
	nv_matrix_free(&desc_vec2);

	return (float)ok/NV_MIN(n1, n2);
}

void nv_test_keypoint(void)
{
	int i;
	int n = 1;
	float match;
	nv_keypoint_param_t param1 = *nv_keypoint_param_gradient_histogram_default();
	nv_keypoint_param_t param2 = *nv_keypoint_param_rectangle_feature_default();
	
	for (i = 0; i < n; ++i) {
		match = nv_test_keypoint_match(&param1, NV_TEST_IMG, NV_TEST_IMG_ROTATE);
		NV_ASSERT(match > 0.5f);
	}
	match = nv_test_keypoint_match(&param1, NV_TEST_IMG, NV_TEST_NEGA);
	NV_ASSERT(match < 0.5f);
	
	for (i = 0; i < n; ++i) {
		match = nv_test_keypoint_match(&param2, NV_TEST_IMG, NV_TEST_IMG_ROTATE);
		NV_ASSERT(match > 0.5f);
	}
	match = nv_test_keypoint_match(&param2, NV_TEST_IMG, NV_TEST_NEGA);
	NV_ASSERT(match < 0.5f);

	fflush(stdout);
}
