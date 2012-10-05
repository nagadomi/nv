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
#include "nv_test.h"

int main(void)
{
	NV_BACKTRACE;
	nv_test_sha1();
	nv_test_io();
	nv_test_serialize();
	
	nv_test_keypoint();
	
	nv_test_knn_pca();
	nv_test_knn2();
	nv_test_lr();
	nv_test_arow();
	nv_test_pa();	
	nv_test_mlp();
	nv_test_nb();
	
	nv_test_knn();
	nv_test_munkres();
	
	nv_test_kmeans();
	nv_test_lbgu();	
	nv_test_klr();
	nv_test_knb();

	nv_test_kmeans_tree();
	nv_test_klr_tree();
	
	//nv_test_plsi();

	
	return 0;
}
