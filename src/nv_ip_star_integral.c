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

#include "nv_core.h"
#include "nv_ip_integral.h"
#include "nv_ip_star_integral.h"

/*
 * 画像上の半径rの円の積分を星型で近似した高速積分
 * ◆■の重なっている領域を1，重なっていない角の領域を0.5で重み付けして合計する．
 * rは正方形の対角線の長さの1/2
 */
float
nv_star_integral(const nv_matrix_t *integral, const nv_matrix_t *integral_tilted,
				 int row, int col, int r)
{
	float intl_norm, intl_tilt, response;

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

	response = intl_norm + intl_tilt;
	/* 合計を計算する(誤差によって負になることがあるので0.0に) */
	if (response < 0.0f) {
		response = 0.0f;
	}
	return response;
}
