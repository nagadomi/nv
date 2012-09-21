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

#if NV_WITH_OPENCV
#include "nv_io.h"
#include "opencv/cxcore.h"
#include "opencv/cv.h"

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

nv_matrix_t *nv_conv_ipl2nv(const IplImage *img)
{
	nv_matrix_t *nv = NULL;
	int x, y;

	if (img->nChannels >= 3) {
		nv = nv_matrix3d_alloc(3, img->height, img->width);
		for (y = 0; y < img->height; ++y) {
			for (x = 0; x < img->width; ++x) {
				CvScalar v = cvGet2D(img, y, x);
				NV_MAT3D_V(nv, y, x, 0) = (float)v.val[0];
				NV_MAT3D_V(nv, y, x, 1) = (float)v.val[1];
				NV_MAT3D_V(nv, y, x, 2) = (float)v.val[2];
			}
		}
	} else {
		nv = nv_matrix3d_alloc(1, img->height, img->width);
		for (y = 0; y < img->height; ++y) {
			for (x = 0; x < img->width; ++x) {
				CvScalar v = cvGet2D(img, y, x);
				NV_MAT3D_V(nv, y, x, 0) = (float)v.val[0];
			}
		}
	}

	return nv;
}

nv_matrix_t *nv_conv_cvsc2nv(const CvMat *mat)
{
	nv_matrix_t *nv = NULL;
	int x, y;
	int step = mat->step / sizeof(int);

	nv = nv_matrix3d_alloc(1, mat->rows, mat->cols);
	for (y = 0; y < mat->rows; ++y) {
		for (x = 0; x < mat->cols; ++x) {
			NV_MAT3D_V(nv, y, x, 0) = (float)mat->data.i[y * step + x];
		}
	}

	return nv;
}

#endif
