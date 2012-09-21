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

#include <stdio.h>
#include "nv_core.h"
#include "nv_ip.h"
#include "nv_ml.h"
#include "nv_io.h"
#include "nv_cuda.h"
#include "nv_cuda_keypoint.h"

#include "opencv/cv.h"
#include "opencv/cxcore.h"
#include "opencv/highgui.h"

#include <cutil_inline.h>
#define KEYPOINT_M 10000


IplImage *nv_conv_nv2ipl(const nv_matrix_t *img)
{
	IplImage *cv = NULL;
	int x, y;

	if (img->n >= 3) {
		cv = cvCreateImage(cvSize(img->cols, img->rows), IPL_DEPTH_8U, 3);
		for (y = 0; y < img->rows; ++y) {
			for (x = 0; x < img->cols; ++x) {
				cvSet2D(cv, y, x, 
					cvScalar(
						NV_MAT3D_V(img, y, x, 0),
						NV_MAT3D_V(img, y, x, 1),
						NV_MAT3D_V(img, y, x, 2),
						0
					)
				);
			}
		}
	} else {
		cv = cvCreateImage(cvSize(img->cols, img->rows), IPL_DEPTH_8U, 1);
		for (y = 0; y < img->rows; ++y) {
			for (x = 0; x < img->cols; ++x) {
				cvSet2D(cv, y, x, 
					cvScalar(
					NV_MAT3D_V(img, y, x, 0), 0, 0, 0
					)
				);
			}
		}
	}

	return cv;
}

void nv_cuda_keypoint_debug(void)
{
	long t = nv_clock();
	nv_matrix_t *image = nv_load_image("lena.jpg");
	nv_matrix_t *key_vec;
	nv_matrix_t *desc_vec;
	int desc_m;
	float scale = 512.0f / (float)NV_MAX(image->rows, image->cols);
	nv_matrix_t *resize = nv_matrix3d_alloc(3, (int)(image->rows * scale),
		(int)(image->cols * scale));
	nv_matrix_t *gray = nv_matrix3d_alloc(1, resize->rows, resize->cols);
	nv_matrix_t *smooth = nv_matrix3d_alloc(1, resize->rows, resize->cols);

	nv_resize(resize, image);
	nv_gray(gray, resize);
	nv_gaussian5x5(smooth, 0, gray, 0);

	key_vec = nv_matrix_alloc(NV_KEYPOINT_KEYPOINT_N, KEYPOINT_M);
	desc_vec = nv_matrix_alloc(NV_KEYPOINT_DESC_N, KEYPOINT_M);

	nv_matrix_zero(desc_vec);
	nv_matrix_zero(key_vec);

	desc_m = nv_keypoint(key_vec, desc_vec, smooth, 0);

	printf("%dkeypoints, %ldms\n", desc_m, nv_clock() -t);
	t = nv_clock();

	desc_m = nv_cuda_keypoint(key_vec, desc_vec, smooth, 0);
	printf("%dkeypoints, %ldms\n", desc_m, nv_clock() -t);

	nv_matrix_free(&image);
	nv_matrix_free(&key_vec);
	nv_matrix_free(&desc_vec);
	nv_matrix_free(&gray);
	nv_matrix_free(&resize);
	nv_matrix_free(&smooth);
}

#define BENCHMARK_N 100

void nv_cuda_keypoint_bench(int mode, const char *file)
{
	nv_matrix_t *img = nv_load(file);
	nv_matrix_t *gray = nv_matrix3d_alloc(1, img->rows, img->cols);
	nv_matrix_t *smooth = nv_matrix3d_alloc(1, img->rows, img->cols);
	int i;
	long t;
	const char *names[] = {"CPU", "GPU", "CPU+GPU"};

	nv_gray(gray, img);
	nv_gaussian5x5(smooth, 0, gray, 0);

	t = nv_clock();
	if (mode == 0) {
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1)
#endif
		for (i = 0; i < BENCHMARK_N; ++i) {
			nv_matrix_t *keypoints = nv_matrix_alloc(NV_KEYPOINT_KEYPOINT_N, KEYPOINT_M);
			nv_matrix_t *desc = nv_matrix_alloc(NV_KEYPOINT_DESC_N, keypoints->m);
			int nkeypoint = nv_keypoint(
				keypoints, desc,
				smooth, 0);
			nv_matrix_free(&keypoints);
			nv_matrix_free(&desc);
		}
	} else if (mode == 1) {
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1) num_threads(8)
#endif
		for (i = 0; i < BENCHMARK_N; ++i) {
			nv_matrix_t *keypoints = nv_matrix_alloc(NV_KEYPOINT_KEYPOINT_N, KEYPOINT_M);
			nv_matrix_t *desc = nv_matrix_alloc(NV_KEYPOINT_DESC_N, keypoints->m);
			int nkeypoint = nv_cuda_keypoint(
				keypoints, desc,
				smooth, 0);
			nv_matrix_free(&keypoints);
			nv_matrix_free(&desc);
		}
	} else {
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1) num_threads(8)
#endif
		for (i = 0; i < BENCHMARK_N; ++i) {
			nv_matrix_t *keypoints = nv_matrix_alloc(NV_KEYPOINT_KEYPOINT_N, KEYPOINT_M);
			nv_matrix_t *desc = nv_matrix_alloc(NV_KEYPOINT_DESC_N, keypoints->m);
			if (i % 3 != 0) {
				int nkeypoint = nv_cuda_keypoint(
					keypoints, desc,
					smooth, 0);
			} else {
				int nkeypoint = nv_keypoint(
					keypoints, desc,
					smooth, 0);
			}
			nv_matrix_free(&keypoints);
			nv_matrix_free(&desc);
		}
	}
	t = nv_clock() - t;
	printf("%s: %ffps\n", names[mode], ((float)BENCHMARK_N/(float)t) * 1000.0f);

	nv_matrix_free(&img);
	nv_matrix_free(&gray);
	nv_matrix_free(&smooth);
}

void nv_cuda_keypoint_plot(int cuda, const char *window, const char *file)
{
	nv_matrix_t *img = nv_load(file);
	nv_matrix_t *gray = nv_matrix3d_alloc(1, img->rows, img->cols);
	nv_matrix_t *smooth = nv_matrix3d_alloc(1, img->rows, img->cols);
	IplImage *cv = nv_conv_nv2ipl(img);//cvCreateImage(cvSize(img->cols, img->rows), IPL_DEPTH_8U, 3);//
	IplImage *id = cvCreateImage(cvSize(img->cols, img->rows), IPL_DEPTH_8U, 3);//
	nv_matrix_t *keypoints = nv_matrix_alloc(NV_KEYPOINT_KEYPOINT_N, KEYPOINT_M);
	nv_matrix_t *desc = nv_matrix_alloc(NV_KEYPOINT_DESC_N, keypoints->m);
	int nkeypoint;
	int i;
	long t, t2, t3;
	cvZero(id);

	nv_gray(gray, img);
	nv_gaussian5x5(smooth, 0, gray, 0);

	t = nv_clock();
	if (cuda) {
		nkeypoint = nv_cuda_keypoint(
			keypoints, desc,
			smooth, 0);
	} else {
		nkeypoint = nv_keypoint(
			keypoints, desc,
			smooth, 0);
	}
	nv_matrix_m(keypoints, nkeypoint);
	nv_matrix_m(desc, nkeypoint);
	//nv_matrix_print(stdout, desc);

	t2 = nv_clock();
	t3 = nv_clock();
	printf("%s: %dpts,%ldms:%ldms:%ldms\n", window, nkeypoint, t2 - t, t3 - t2, t3 - t);

	for (i = 0; i < nkeypoint; ++i) {
		float x = 0.0f;
		float y = NV_MAT_V(keypoints, i, NV_KEYPOINT_RADIUS_IDX) * 2.0f;
		float theta = NV_MAT_V(keypoints, i, NV_KEYPOINT_ORIENTATION_IDX); 
		float nx = x * cosf(theta) - y * sinf(theta);
		float ny = x * sinf(theta) + y * cosf(theta);
		int j;
		float step = NV_PI / 4.0f;
		float angle = NV_MAT_V(keypoints, i, NV_KEYPOINT_ORIENTATION_IDX); 
		float sub_desc_r = NV_MAT_V(keypoints, i, 2);
		float pi2 = NV_PI * 2.0f;
		CvScalar rand_color = cvScalar(255 * nv_rand(), 255 * nv_rand(), 255 * nv_rand(), 0);

		nx += NV_MAT_V(keypoints, i, NV_KEYPOINT_X_IDX);
		ny += NV_MAT_V(keypoints, i, NV_KEYPOINT_Y_IDX);

		CvScalar color = NV_MAT_V(keypoints, i, NV_KEYPOINT_RESPONSE_IDX) > 0.0f ? cvScalar(0, 0, 255, 0):cvScalar(255, 0, 0, 0);
		cvCircle(
			cv, 
			cvPoint(NV_MAT_V(keypoints, i, NV_KEYPOINT_X_IDX), NV_MAT_V(keypoints, i, NV_KEYPOINT_Y_IDX)),
			NV_MAT_V(keypoints, i, NV_KEYPOINT_RADIUS_IDX) * 2, color, 1, 8, 0);
		cvCircle(
			cv, 
			cvPoint(NV_MAT_V(keypoints, i, NV_KEYPOINT_X_IDX), NV_MAT_V(keypoints, i, NV_KEYPOINT_Y_IDX)),
			1, color, 2, CV_AA, 0);
		cvLine(cv,
			cvPoint(NV_MAT_V(keypoints, i, NV_KEYPOINT_X_IDX), NV_MAT_V(keypoints, i, NV_KEYPOINT_Y_IDX)),
			cvPoint(nx, ny), cvScalar(0, 255, 0), 1, 8, 0);

		cvCircle(
			id, 
			cvPoint(NV_MAT_V(keypoints, i, NV_KEYPOINT_X_IDX), NV_MAT_V(keypoints, i, NV_KEYPOINT_Y_IDX)),
			1, color, 2, CV_AA, 0);
	}

	cvNamedWindow(window, 1);
	cvShowImage(window, cv);
	cvWaitKey(0);
	cvReleaseImage(&cv);
	nv_matrix_free(&img);
	nv_matrix_free(&gray);
	nv_matrix_free(&smooth);
}

float
nv_cuda_keypoint_match(const char *window, const char *file1, const char *file2)
{
	nv_matrix_t *img1 = nv_load(file1);
	nv_matrix_t *gray1= nv_matrix3d_alloc(1, img1->rows, img1->cols);
	nv_matrix_t *smooth1 = nv_matrix3d_alloc(1, img1->rows, img1->cols);
	IplImage *cv1 = nv_conv_nv2ipl(img1);//cvCreateImage(cvSize(img->cols, img->rows), IPL_DEPTH_8U, 3);//
	nv_matrix_t *img2 = nv_load(file2);
	nv_matrix_t *gray2= nv_matrix3d_alloc(1, img2->rows, img2->cols);
	nv_matrix_t *smooth2 = nv_matrix3d_alloc(1, img2->rows, img2->cols);
	IplImage *cv2 = nv_conv_nv2ipl(img2);//cvCreateImage(cvSize(img->cols, img->rows), IPL_DEPTH_8U, 3);//
	IplImage *cv = cvCreateImage(cvSize(img1->cols + img2->cols, NV_MAX(img1->rows,img2->rows)), IPL_DEPTH_8U, 3);

	nv_matrix_t *keypoints1 = nv_matrix_alloc(NV_KEYPOINT_KEYPOINT_N, KEYPOINT_M);
	nv_matrix_t *desc1 = nv_matrix_alloc(NV_KEYPOINT_DESC_N, img1->m);
	int nkeypoint1;
	nv_matrix_t *keypoints2 = nv_matrix_alloc(NV_KEYPOINT_KEYPOINT_N, KEYPOINT_M);
	nv_matrix_t *desc2 = nv_matrix_alloc(NV_KEYPOINT_DESC_N, img2->m);
	int nkeypoint2;
	int m;
	nv_knn_result_t knn_index[2];
	int ok;
	long t;
	char wname[256];
	float dist_sum = 0.0f;
	int dist_count = 0;

#if 0
	cvZero(cv);
#endif

	t = nv_clock();
	nv_gray(gray1, img1);
	nv_gaussian5x5(smooth1, 0, gray1, 0);

	nkeypoint1 = nv_keypoint(
		keypoints1, desc1,
		smooth1, 0);
	printf("detecting: %dms, %dkeypoints\n", nv_clock() - t, nkeypoint1);
	nv_matrix_m(desc1, nkeypoint1);
	t = nv_clock();
	nv_gray(gray2, img2);
	nv_gaussian5x5(smooth2, 0, gray2, 0);

	nkeypoint2 = nv_cuda_keypoint(
		keypoints2, desc2,
		smooth2, 0);

	printf("detecting: %dms, %dkeypoints\n", nv_clock() - t, nkeypoint2);
	nv_matrix_m(desc2, nkeypoint2);

	cvSetImageROI(cv, cvRect(0, 0, img1->cols, img1->rows));
	cvCopy(cv1, cv);
	cvResetImageROI(cv);
	cvSetImageROI(cv, cvRect(img1->cols, 0, img2->cols, img2->rows));
	cvCopy(cv2, cv);
	cvResetImageROI(cv);
	t = nv_clock();

	ok = 0;
	for (m = 0; m < nkeypoint1; ++m) {
		float d1, d2;
		int c, c2;
		nv_knn(knn_index, 2, desc2, desc1, m);
		c = knn_index[0].index;
		c2 = knn_index[1].index;
		d1 = knn_index[0].dist;
		d2 = knn_index[1].dist;

		if (nv_sign(NV_MAT_V(keypoints1, m, NV_KEYPOINT_RESPONSE_IDX)) != nv_sign(NV_MAT_V(keypoints2, c, NV_KEYPOINT_RESPONSE_IDX))) {
			continue;
		}

		if (d1 < d2 * 0.7f) {

			int y1 = NV_MAT_V(keypoints1, m, NV_KEYPOINT_Y_IDX);
			int x1 = NV_MAT_V(keypoints1, m, NV_KEYPOINT_X_IDX);
			int y2 = NV_MAT_V(keypoints2, c, NV_KEYPOINT_Y_IDX);
			int x2 = NV_MAT_V(keypoints2, c, NV_KEYPOINT_X_IDX) + img1->cols;

			cvLine(
				cv,
				cvPoint(x1, y1),
				cvPoint(x2, y2),
				cvScalar(255 * nv_rand(), 255 * nv_rand(), 255 * nv_rand(), 0),
				1, 8, 0);

		CvScalar color1 = NV_MAT_V(keypoints1, m, NV_KEYPOINT_RESPONSE_IDX) > 0.0f ? cvScalar(0, 0, 255, 0):cvScalar(255, 0, 0, 0);
		cvCircle(
			cv, 
			cvPoint(NV_MAT_V(keypoints1, m, NV_KEYPOINT_X_IDX), NV_MAT_V(keypoints1, m, NV_KEYPOINT_Y_IDX)),
			1, color1, 2, CV_AA, 0);
		cvCircle(
			cv, 
			cvPoint(NV_MAT_V(keypoints1, m, NV_KEYPOINT_X_IDX), NV_MAT_V(keypoints1, m, NV_KEYPOINT_Y_IDX)),
			NV_MAT_V(keypoints1, m, NV_KEYPOINT_RADIUS_IDX) * 2, color1, 1, 8, 0);


		CvScalar color2 = NV_MAT_V(keypoints2, c, NV_KEYPOINT_RESPONSE_IDX) > 0.0f ? cvScalar(0, 0, 255, 0):cvScalar(255, 0, 0, 0);
		cvCircle(
			cv, 
			cvPoint(NV_MAT_V(keypoints2, c, NV_KEYPOINT_X_IDX) + img1->cols, NV_MAT_V(keypoints2, c, NV_KEYPOINT_Y_IDX)),
			1, color2, 2, CV_AA, 0);
			++ok;
		}else {
			//printf("NG\n");
		}
	}
	printf("matching: %dms\n", nv_clock() - t);
	printf("OK: %d, NG: %d, %f\n\n", ok, nkeypoint1 - ok, (float)ok/(sqrtf(nkeypoint1) * sqrtf(nkeypoint2)));

	cvNamedWindow(window, 1);
	cvShowImage(window, cv);
	cvWaitKey(0);
	nv_matrix_free(&img1);
	nv_matrix_free(&gray1);
	nv_matrix_free(&smooth1);
	nv_matrix_free(&desc1);
	nv_matrix_free(&keypoints1);
	nv_matrix_free(&img2);
	nv_matrix_free(&gray2);
	nv_matrix_free(&smooth2);
	nv_matrix_free(&desc2);
	nv_matrix_free(&keypoints2);

	cvReleaseImage(&cv);
	cvReleaseImage(&cv1);
	cvReleaseImage(&cv2);

	return (float)ok/(sqrtf(nkeypoint1) * sqrtf(nkeypoint2));
}

void nv_cuda_debug(void)
{
	nv_cuda_keypoint_debug();
}

int main(int argc, char **argv)
{
	//CUT_DEVICE_INIT(argc, argv);

	// Debug mode only!!
	nv_cuda_init();

	//nv_cuda_debug();

	//nv_cuda_keypoint_plot(1, "GPU", "lena.jpg");
	//nv_cuda_keypoint_plot(1, "GPU", "lena.jpg");

	//nv_cuda_keypoint_plot(1, "GPU", "lena.jpg");
	//nv_cuda_keypoint_plot(0, "CPU", "lena.jpg");
	nv_cuda_keypoint_match("DEBUG", "lena.jpg", "lena.jpg");
	//nv_cuda_keypoint_bench(0, "lena.jpg");
	//nv_cuda_keypoint_bench(1, "lena.jpg");
	//nv_cuda_keypoint_bench(2, "lena.jpg");
	//getchar();
	return 0;
}
