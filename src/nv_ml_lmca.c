/*
 * This file is part of libnv.
 *
 * Copyright (C) 2012 nagadomi@nurs.or.jp
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License,
 * or any later version.
 * 4
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
#include "nv_core.h"
#include "nv_num.h"
#include "nv_ml_lmca.h"

/**
 * Reference:
 *  Large Margin Component Analysis 
 *  Lorenzo Torresani, Kuang-chih Lee
 */

#define NV_LMCA_TERM2_SCALE 0.5f
#define NV_LMCA_MOMENTUM_W 0.5f
#define NV_LMCA_LEARNING_RATE(w, max_iteration, e) (w - (w * 0.5f * (1.0f - ((max_iteration - e) / (float)max_iteration))))

static int nv_lmca_progress_flag = 0;

#define NV_LMCA_SMOOTH_HINGE(s, margin)   \
	((s) < 0.0f ? 0.0f : ((s) > (margin) ? 1.0f : (((s) / (margin)) * ((s) / (margin)))))

#define NV_LMCA_D_SMOOTH_HINGE(s, margin) \
	((s) < 0.0f ? 0.0f : ((s) > (margin) ? 1.0f : (s) / (margin)))

#define NV_LMCA_DEFAULT_HINGE(s, margin)   \
	((s) < 0.0f ? 0.0f : (s) / (margin))

#define NV_LMCA_D_DEFAULT_HINGE(s, margin) \
	((s) < 0.0f ? 0.0f : 1.0f)

#define NV_LMCA_SMOOTH 0
#if NV_LMCA_SMOOTH
#  define NV_LMCA_HINGE(s, margin) NV_LMCA_SMOOTH_HINGE(s, margin)
#  define NV_LMCA_D_HINGE(s, margin) NV_LMCA_D_SMOOTH_HINGE(s, margin)
#else
#  define NV_LMCA_HINGE(s, margin) NV_LMCA_DEFAULT_HINGE(s, margin)
#  define NV_LMCA_D_HINGE(s, margin) NV_LMCA_D_DEFAULT_HINGE(s, margin)
#endif

void
nv_lmca_progress(int onoff)
{
	nv_lmca_progress_flag = onoff;
}

/* PCAで初期化 */
void
nv_lmca_init_pca(nv_matrix_t *ldm,
				 const nv_matrix_t *data)
{
	int i;
	nv_cov_t *cov;
	long t = nv_clock();
	
	NV_ASSERT(ldm->n == data->n);
	if (nv_lmca_progress_flag) {
		printf("nv_lmca_init_pca: ");
		fflush(stdout);
	}
	cov = nv_cov_alloc(data->n);
	
	/* 初期値用 */
	nv_cov_eigen_ex(cov, data, ldm->m, 50);
	
	/* 固有値が大きい方から使う  */
	for (i = 0; i < ldm->m; ++i) {
		nv_vector_copy(ldm, i, cov->eigen_vec, i);
	}
	nv_vector_normalize_all(ldm);
	
	nv_cov_free(&cov);
	
	if (nv_lmca_progress_flag) {
		printf("%ldms\n", nv_clock() - t);
		fflush(stdout);
	}
}

/* 分散共分散行列で初期化 */
void
nv_lmca_init_cov(nv_matrix_t *ldm,
				 const nv_matrix_t *data)
{
	int i;
	nv_cov_t *cov;
	nv_matrix_t *c;
	long t = nv_clock();
	
	if (nv_lmca_progress_flag) {
		printf("nv_lmca_init_cov: ");
		fflush(stdout);
	}
	NV_ASSERT(ldm->n == data->n);
	
	cov = nv_cov_alloc(data->n);
	nv_cov(cov->cov, cov->u, data);
	c = nv_matrix_alloc(cov->cov->n + 1, cov->cov->m);
	
#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (i = 0; i < cov->cov->m; ++i) {
		NV_MAT_V(c, i, 0) = nv_vector_norm(cov->cov, i);
		memmove(&NV_MAT_V(c, i, 1), &NV_MAT_V(cov->cov, i, 0), sizeof(float) * cov->cov->n);
	}
	nv_matrix_sort(c, 0, NV_SORT_DIR_DESC);
	
	/* ノルムが大きい方から使う */
	for (i = 0; i < ldm->m; ++i) {
		memmove(&NV_MAT_V(ldm, i, 0), &NV_MAT_V(c, i, 1), sizeof(float) * ldm->n);
	}
	/* L2 Normalize */	
	nv_vector_normalize_all(ldm);
	
	nv_matrix_free(&c);
	nv_cov_free(&cov);
	
	if (nv_lmca_progress_flag) {
		printf("%ldms\n", nv_clock() - t);
		fflush(stdout);
	}
}

/* 対角成分1, それ以外を0で初期化 */
void
nv_lmca_init_diag1(nv_matrix_t *ldm)
{
	int i;
	
	nv_matrix_zero(ldm);
	
	for (i = 0; i < ldm->n; ++i) {
		NV_MAT_V(ldm, i, i) = 1.0f;
	}
}

/* Random Projectionで初期化 */
void
nv_lmca_init_random_projection(nv_matrix_t *ldm)
{
	int j, i;
	
	for (i = 0; i < ldm->m; ++i) {
		for (j = 0; j < ldm->n; ++j) {
			NV_MAT_V(ldm, i, j) = nv_nrand(0.0f, 1.0f);
		}
	}
	nv_vector_normalize_all(ldm);
}

static float
nv_lmca_matrix_abs_max(const nv_matrix_t *mat)
{
	float maxv = 0.0f;
	int i;
	
	for (i = 0; i < mat->m; ++i) {
		int j;
		for (j = 0; j < mat->n; ++j) {
			float v = fabsf(NV_MAT_V(mat, i, j));
			if (maxv < v) {
				maxv = v;
			}
		}
	}
	return maxv;
}

/* 変換行列を正規化する */
static void
nv_lmca_normalize_l(nv_matrix_t *ldm, nv_lmca_type_e type)
{
	if (type == NV_LMCA_FULL) {
		nv_vector_normalize_all(ldm);
	} else if (type == NV_LMCA_DIAG) {
		/* DIAGの場合は対角成分のノルムを1にする  */
		int i;
		float dot = 0.0f;
		for (i = 0; i < ldm->n; ++i) {
			dot += NV_MAT_V(ldm, i, i) * NV_MAT_V(ldm, i, i);
		}
		if (dot > 0.0f) {
			float scale = 1.0f / sqrtf(dot);
			for (i = 0; i < ldm->n; ++i) {
				NV_MAT_V(ldm, i, i) *= scale;
			}
		}
	}
}

static void
nv_lmca_lx(nv_matrix_t *data2,
		   const nv_matrix_t *ldm,
		   const nv_matrix_t *data
	)
{
	int i;

	NV_ASSERT(data2->m == data->m && data2->n == ldm->m);
	
#ifdef _OPENMP
#pragma omp parallel for	
#endif
	for (i = 0; i < data->m; ++i) {
		nv_lmca_projection(data2, i, ldm, data, i);
	}
}

/* コスト関数 */
static float
nv_lmca_cost(const nv_matrix_t *lx,
			 const nv_matrix_t *labels,
			 nv_knn_result_t **eta,
			 int k,
			 nv_knn_result_t **eta_lx,
			 int k_n,
			 float margin,
			 float c,
			 int iteration
	)
{
	int i;
	float eps1, eps2;
	int correct = 0;
	
	eps1 = 0.0f;
#ifdef _OPENMP
#pragma omp parallel for reduction (+: eps1)
#endif
	for (i = 0; i < lx->m; ++i) {
		int j;
		int il = NV_MAT_VI(labels, i, 0); 
		for (j = 0; j < k; ++j) {
			int j_idx = eta[i][j].index;
			if (il == NV_MAT_VI(labels, j_idx, 0)) {
				eps1 += nv_euclidean2(lx, i, lx, j_idx);
			}
		}
	}
	eps2 = 0.0f;
#ifdef _OPENMP
#pragma omp parallel for reduction (+: eps2)
#endif
	for (i = 0; i < lx->m; ++i) {
		int j, l;
		int il = NV_MAT_VI(labels, i, 0); 
		for (j = 0; j < k; ++j) {
			int j_idx = eta[i][j].index;
			if (il == NV_MAT_V(labels, j_idx, 0)) {
				float d_ij = nv_euclidean2(lx, i, lx, j_idx);
				for (l = 0; l < k_n; ++l) {
					int l_idx = eta_lx[i][l].index;
					int yil = (il == NV_MAT_VI(labels, l_idx, 0)) ? 1 : 0;
					if (yil == 0) {
						float s = d_ij + margin - nv_euclidean2(lx, i, lx, l_idx);
						eps2 += NV_LMCA_HINGE(s, margin);
					}
				}
			}
		}
	}
#ifdef _OPENMP
#pragma omp parallel for reduction (+: correct)
#endif
	for (i = 0; i < lx->m; ++i) {
		int j;
		int il = NV_MAT_VI(labels, i, 0);
		for (j = 0; j < k; ++j) {
			int j_idx = eta_lx[i][j].index;
			correct += (il == NV_MAT_VI(labels, j_idx, 0)) ? 1 : 0;
		}
	}
	if (nv_lmca_progress_flag) {
		printf("nv_lmca: %3d: pull_cost: %E push_cost: %E, knn precision: %f(%d/%d/%d)\n", iteration, eps1, eps2, (float)correct / (lx->m * k), correct, k, lx->m);
		fflush(stdout);
	}
	return (1.0f - c) * eps1 + c * eps2;
}

static void
nv_lmca_momentum_add(nv_matrix_t *dl, nv_matrix_t *dl_old, float weight, int epoch)
{
	if (epoch == 0) {
		nv_matrix_copy_all(dl_old, dl);
	} else {
		nv_matrix_t *tmp = nv_matrix_dup(dl);
		//nv_matrix_muls(dl, dl, 1.0 - weight);
		nv_matrix_muls(dl_old, dl_old, weight);
		nv_matrix_add(dl, dl, dl_old);
		nv_matrix_copy_all(dl_old, tmp);
		nv_matrix_free(&tmp);
	}
}

void
nv_lmca_train_ex(nv_matrix_t *ldm,
				 nv_lmca_type_e type,
				 const nv_matrix_t *data, const nv_matrix_t *labels,
				 int k, int k_n,
				 float margin, float push_weight, float learning_rate,
				 int max_iteration
	)
{
	nv_matrix_t *lx = nv_matrix_alloc(ldm->m, data->m);
	nv_knn_result_t **eta = nv_alloc_type(nv_knn_result_t *, data->m);
	nv_knn_result_t **eta_lx = nv_alloc_type(nv_knn_result_t *, data->m);
	nv_matrix_t *dl, *dl_old, **term1, **term2;
	int e, i;
	int procs = nv_omp_procs();
	long t = nv_clock();
	float pull_ratio = 1.0f - push_weight;
	int knn_correct = 0;
	float delta_scale;

	NV_ASSERT(type == NV_LMCA_FULL || (type == NV_LMCA_DIAG && ldm->m == ldm->n));
	NV_ASSERT(data->m >= k);
	NV_ASSERT(data->m >= k_n);
	
	dl = nv_matrix_clone(ldm);
	dl_old = nv_matrix_clone(ldm);
	term1 = nv_alloc_type(nv_matrix_t *, procs);
	term2 = nv_alloc_type(nv_matrix_t *, procs);
	for (i = 0; i < procs; ++i) {
		term1[i] = nv_matrix_alloc(data->n, data->n);
		term2[i] = nv_matrix_alloc(data->n, data->n);
	}
	
	/* 元の空間でのk近傍にあるデータを保存 */
#ifdef _OPENMP
#pragma omp parallel for	
#endif
	for (i = 0; i < data->m; ++i) {
		eta[i] = nv_alloc_type(nv_knn_result_t, k);
		nv_knn(eta[i], k, data, data, i);

		/* 射影後の空間でのk+N近傍（あとで使う) */
		eta_lx[i] = nv_alloc_type(nv_knn_result_t, k_n);
	}
	if (nv_lmca_progress_flag) {
		printf("nv_lmca: eta_ij: %ldms\n", nv_clock() - t);
		fflush(stdout);
	}
	t = nv_clock();
	
	/* 第一項 */
	for (i = 0; i < procs; ++i) {
		nv_matrix_zero(term1[i]);
	}
#ifdef _OPENMP
#pragma omp parallel for num_threads(procs) schedule(dynamic, 1) reduction(+:knn_correct)
#endif
	for (i = 0; i < data->m; ++i) {
		const int thread_id = nv_omp_thread_id();
		const int il = NV_MAT_VI(labels, i, 0);
		nv_matrix_t *mat = term1[thread_id];
		nv_matrix_t *d = nv_matrix_alloc(data->n, 1);
		int j;
		
		for (j = 0; j < k; ++j) {
			int j_idx = eta[i][j].index;
			if (il == NV_MAT_VI(labels, j_idx, 0)) {
				++knn_correct;
				nv_vector_sub(d, 0, data, i, data, j_idx);
				if (type == NV_LMCA_FULL) {
					int ii;
#if NV_ENABLE_SSE
					int pk_lp = (data->n & 0xfffffffc);
				
					if (pk_lp == data->n) {
						for (ii = 0; ii < data->n; ii += 4) {
							const __m128 d0 = _mm_load_ps(&NV_MAT_V(d, 0, ii));
							const __m128 iv0 = _mm_shuffle_ps(d0, d0, _MM_SHUFFLE(0, 0, 0, 0));
							const __m128 iv1 = _mm_shuffle_ps(d0, d0, _MM_SHUFFLE(1, 1, 1, 1));
							const __m128 iv2 = _mm_shuffle_ps(d0, d0, _MM_SHUFFLE(2, 2, 2, 2));
							const __m128 iv3 = _mm_shuffle_ps(d0, d0, _MM_SHUFFLE(3, 3, 3, 3));
							int jj;
							
							for (jj = 0; jj < data->n; jj += 4) {
								__m128 jv = _mm_load_ps(&NV_MAT_V(d, 0, jj));
								_mm_store_ps(&NV_MAT_V(mat, ii, jj),
											 _mm_add_ps(_mm_mul_ps(jv, iv0),
														*(const __m128 *)&NV_MAT_V(mat, ii, jj)));
								_mm_store_ps(&NV_MAT_V(mat, ii + 1, jj),
											 _mm_add_ps(_mm_mul_ps(jv, iv1),
														*(const __m128 *)&NV_MAT_V(mat, ii + 1, jj)));
								_mm_store_ps(&NV_MAT_V(mat, ii + 2, jj),
											 _mm_add_ps(_mm_mul_ps(jv, iv2),
														*(const __m128 *)&NV_MAT_V(mat, ii + 2, jj)));
								_mm_store_ps(&NV_MAT_V(mat, ii + 3, jj),
											 _mm_add_ps(_mm_mul_ps(jv, iv3),
														*(const __m128 *)&NV_MAT_V(mat, ii + 3, jj)));
							}
						}
					} else {
						for (ii = 0; ii < d->n; ++ii) {
							const __m128 iv = _mm_set1_ps(NV_MAT_V(d, 0, ii));
							int jj;
							for (jj = 0; jj < pk_lp; jj += 4) {
								__m128 jv = _mm_load_ps(&NV_MAT_V(d, 0, jj));
								jv = _mm_mul_ps(jv, iv);
								jv = _mm_add_ps(jv, *(const __m128 *)&NV_MAT_V(mat, ii, jj));
								_mm_store_ps(&NV_MAT_V(mat, ii, jj), jv);
							}
							for (jj = pk_lp; jj < data->n; ++jj) {
								NV_MAT_V(mat, ii, jj) += NV_MAT_V(d, 0, jj) * NV_MAT_V(d, 0, ii);
							}
						}
					}
#else
					for (ii = 0; ii < d->n; ++ii) {
						int jj;
						NV_MAT_V(mat, ii, ii) += NV_MAT_V(d, 0, ii) * NV_MAT_V(d, 0, ii);;
						for (jj = ii + 1; jj < d->n; ++jj) {
							const float v = NV_MAT_V(d, 0, jj) * NV_MAT_V(d, 0, ii);
							NV_MAT_V(mat, ii, jj) += v;
							NV_MAT_V(mat, jj, ii) += v;
						}
					}
#endif
				} else if (type == NV_LMCA_DIAG) {
					int ii;
					for (ii = 0; ii < d->n; ++ii) {
						float v = NV_MAT_V(d, 0, ii) * NV_MAT_V(d, 0, ii);
						NV_MAT_V(mat, ii, ii) += v;
					}
				} else {
					NV_ASSERT("uknown type" == NULL);
				}
			}
		}
		nv_matrix_free(&d);
	}
	/* 並列計算の結果を統合 */
	for (i = 1; i < procs; ++i) {
		int j;
		for (j = 0; j < term1[0]->m; ++j) {
			nv_vector_add(term1[0], j, term1[0], j, term1[i], j);
		}
	}
	
    /*
	 * 最初の勾配の絶対値が最大の要素で正規化する
	 */
	delta_scale = nv_lmca_matrix_abs_max(term1[0]);
	if (delta_scale > 0.0f) {
		delta_scale = 1.0f / delta_scale * NV_LMCA_TERM2_SCALE;
	}
	nv_matrix_muls(term1[0], term1[0], delta_scale);
	
	if (nv_lmca_progress_flag) {
		printf("nv_lmca: term1: %ldms, input space knn precision: %f(%d/%d/%d)\n",
			   nv_clock() - t, (float)knn_correct / (data->m * k), knn_correct, k, data->m);
	}

	/* 変換行列を正則化して全データを変換する */
	nv_lmca_normalize_l(ldm, type);
	nv_lmca_lx(lx, ldm, data);

	/* 各データの射影後のk_n近傍にあるデータを保存 0回目 */
#ifdef _OPENMP
#pragma omp parallel for	
#endif
	for (i = 0; i < lx->m; ++i) {
		nv_knn(eta_lx[i], k_n, lx, lx, i);
	}
	
	/* 第二項 */
	for (e = 0; e < max_iteration; ++e) {
		float cost;
		int push_sum = 0;
		int push_count = 0;
		t = nv_clock();
		
		for (i = 0; i < procs; ++i) {
			nv_matrix_zero(term2[i]);
		}
#ifdef _OPENMP
#pragma omp parallel for num_threads(procs)  schedule(dynamic, 1) reduction(+: push_sum, push_count)
#endif
		for (i = 0; i < data->m; ++i) {
			int j;
			const int thread_id = nv_omp_thread_id();
			const int il = NV_MAT_VI(labels, i, 0);
			nv_matrix_t *mat = term2[thread_id];
			nv_matrix_t *d = nv_matrix_alloc(data->n, 2);
			
			for (j = 1; j < k; ++j) {
				const int j_idx = eta[i][j].index;
				/* k近傍のうち同一ラベルについて(自分自身は除く j=0) */
				if (il == NV_MAT_VI(labels, j_idx, 0)) {
					int l;
					int nc = 0;
					const float d_ij = nv_euclidean2(lx, i, lx, j_idx);
					
					for (l = 1; l < k_n; ++l) {
						const int l_idx = eta_lx[i][l].index;
						if (il != NV_MAT_VI(labels, l_idx, 0)) {
							const float d_il = eta_lx[i][l].dist;
							/* lからマージンまでの距離 */
							const float dist_margin = d_ij + margin - d_il;
							const float d_hinge = NV_LMCA_D_HINGE(dist_margin, margin);
							if (d_hinge > 0.0f) {
								/* k_n近傍のうちk近傍内の同一ラベルからmergin以内のデータについて  */
								nv_vector_sub(d, 0, data, i, data, j_idx);
								nv_vector_sub(d, 1, data, i, data, l_idx);
								if (type == NV_LMCA_FULL) {
									int ii;
#if NV_ENABLE_SSE
									const int pk_lp = (data->n & 0xfffffffc);
									const __m128 d_hinge_v = _mm_set1_ps(d_hinge);
									if (pk_lp == data->n) {
										for (ii = 0; ii < data->n; ii += 4) {
											const __m128 d0 = _mm_load_ps(&NV_MAT_V(d, 0, ii));
											const __m128 iv0 = _mm_shuffle_ps(d0, d0, _MM_SHUFFLE(0, 0, 0, 0));
											const __m128 iv1 = _mm_shuffle_ps(d0, d0, _MM_SHUFFLE(1, 1, 1, 1));
											const __m128 iv2 = _mm_shuffle_ps(d0, d0, _MM_SHUFFLE(2, 2, 2, 2));
											const __m128 iv3 = _mm_shuffle_ps(d0, d0, _MM_SHUFFLE(3, 3, 3, 3));
											const __m128 d1 = _mm_load_ps(&NV_MAT_V(d, 1, ii));
											const __m128 iv4 = _mm_shuffle_ps(d1, d1, _MM_SHUFFLE(0, 0, 0, 0));
											const __m128 iv5 = _mm_shuffle_ps(d1, d1, _MM_SHUFFLE(1, 1, 1, 1));
											const __m128 iv6 = _mm_shuffle_ps(d1, d1, _MM_SHUFFLE(2, 2, 2, 2));
											const __m128 iv7 = _mm_shuffle_ps(d1, d1, _MM_SHUFFLE(3, 3, 3, 3));
											int jj;
											
											for (jj = 0; jj < data->n; jj += 4) {
												const __m128 jv0 = _mm_load_ps(&NV_MAT_V(d, 0, jj));
												const __m128 jv1 = _mm_load_ps(&NV_MAT_V(d, 1, jj));
											
												_mm_store_ps(&NV_MAT_V(mat, ii, jj),
															 _mm_add_ps(_mm_mul_ps(_mm_sub_ps(_mm_mul_ps(iv0, jv0), _mm_mul_ps(iv4, jv1)), d_hinge_v),
																		*(const __m128 *)&NV_MAT_V(mat, ii, jj)));
												_mm_store_ps(&NV_MAT_V(mat, ii + 1, jj),
															 _mm_add_ps(_mm_mul_ps(_mm_sub_ps(_mm_mul_ps(iv1, jv0), _mm_mul_ps(iv5, jv1)), d_hinge_v),
																		*(const __m128 *)&NV_MAT_V(mat, ii + 1, jj)));
												_mm_store_ps(&NV_MAT_V(mat, ii + 2, jj),
															 _mm_add_ps(_mm_mul_ps(_mm_sub_ps(_mm_mul_ps(iv2, jv0), _mm_mul_ps(iv6, jv1)), d_hinge_v),
																		*(const __m128 *)&NV_MAT_V(mat, ii + 2, jj)));
												_mm_store_ps(&NV_MAT_V(mat, ii + 3, jj),
															 _mm_add_ps(_mm_mul_ps(_mm_sub_ps(_mm_mul_ps(iv3, jv0), _mm_mul_ps(iv7, jv1)), d_hinge_v),
																		*(const __m128 *)&NV_MAT_V(mat, ii + 3, jj)));
											}
										}
									} else {
										for (ii = 0; ii < data->n; ++ii) {
											const __m128 iv1 = _mm_set1_ps(NV_MAT_V(d, 0, ii));
											const __m128 iv2 = _mm_set1_ps(NV_MAT_V(d, 1, ii));
											int jj;
											for (jj = 0; jj < pk_lp; jj += 4) {
												const __m128 jv1 = _mm_load_ps(&NV_MAT_V(d, 0, jj));
												const __m128 jv2 = _mm_load_ps(&NV_MAT_V(d, 1, jj));
												_mm_store_ps(&NV_MAT_V(mat, ii, jj),
															 _mm_add_ps(_mm_mul_ps(_mm_sub_ps(_mm_mul_ps(iv1, jv1), _mm_mul_ps(iv2, jv2)), d_hinge_v),
																		*(const __m128 *)&NV_MAT_V(mat, ii, jj)));
											}
											for (jj = pk_lp; jj < data->n; ++jj) {
												NV_MAT_V(mat, ii, jj) +=
													(NV_MAT_V(d, 0, ii) * NV_MAT_V(d, 0, jj)
													 -
													 NV_MAT_V(d, 1, ii) * NV_MAT_V(d, 1, jj)) * d_hinge;
											}
										}
									}
#else
									for (ii = 0; ii < data->n; ++ii) {
										int jj;
										NV_MAT_V(mat, ii, ii) += (NV_MAT_V(d, 0, ii) *  NV_MAT_V(d, 0, ii)
																  -
																  NV_MAT_V(d, 1, ii) *  NV_MAT_V(d, 1, ii)) * d_hinge;
										for (jj = ii + 1; jj < data->n; ++jj) {
											const float v = (NV_MAT_V(d, 0, ii) *  NV_MAT_V(d, 0, jj)
															 -
															 NV_MAT_V(d, 1, ii) *  NV_MAT_V(d, 1, jj)) * d_hinge;
											NV_MAT_V(mat, ii, jj) += v;
											NV_MAT_V(mat, jj, ii) += v;
										}
									}
#endif
								} else if (type == NV_LMCA_DIAG) {
									int ii;
									for (ii = 0; ii < data->n; ++ii) {
										const float v =
											(NV_MAT_V(d, 0, ii) *  NV_MAT_V(d, 0, ii)
											 -
											 NV_MAT_V(d, 1, ii) *  NV_MAT_V(d, 1, ii)) * d_hinge;
										NV_MAT_V(mat, ii, ii) += v;
									}
								} else {
									NV_ASSERT("uknown type" == NULL);
								}
								++nc;
							} else {
								/* eta_lxは距離順なのでこれより近いデータはない */
								break;
							}
						}
					}
					++push_count;
					push_sum += nc;
				}
			}
			nv_matrix_free(&d);
		}
		/* 並列計算の結果を統合 */
		for (i = 1; i < procs; ++i) {
			int j;
			for (j = 0; j < term2[0]->m; ++j) {
				nv_vector_add(term2[0], j, term2[0], j, term2[i], j);
			}
		}
		nv_matrix_muls(term2[0], term2[0], delta_scale);
		
		/*
		 *　dL = 2 * L * term1 + 2 * c * L * term2
		 *
		 *  2はcに含まれるという事で省略する
		 *  cは比率でやるので
		 *   dL = (1.0 - push_weight) * L * term1 + push_weight * L * term2
		 *  を更新方向とする
		 */
#ifdef _OPENMP
#pragma omp parallel for
#endif
		for (i = 0; i < dl->n; ++i) {
			int j;
			for (j = 0; j < dl->m; ++j) {
				NV_MAT_V(dl, j, i) = pull_ratio * nv_vector_dot(ldm, j, term1[0], i)
					+ push_weight * nv_vector_dot(ldm, j, term2[0], i);
			}
		}
		/*
		 * L_new = L_old - learning_rate * dL
		 */
		nv_lmca_momentum_add(dl, dl_old, NV_LMCA_MOMENTUM_W, e);
#ifdef _OPENMP
#pragma omp parallel for
#endif
		for (i = 0; i < ldm->m; ++i) {
			int j;
			float w = NV_LMCA_LEARNING_RATE(learning_rate, max_iteration, e);
			for (j = 0; j < ldm->n; ++j) {
				NV_MAT_V(ldm, i, j) -= w * NV_MAT_V(dl, i, j);
			}
		}
		
		/* 結果を評価する */
		nv_lmca_normalize_l(ldm, type);
		nv_lmca_lx(lx, ldm, data);
		
		/* 各データの射影後のk_n近傍にあるデータを保存 */
#ifdef _OPENMP
#pragma omp parallel for	
#endif
		for (i = 0; i < lx->m; ++i) {
			nv_knn(eta_lx[i], k_n, lx, lx, i);
		}
		
		/* コスト */
		cost = nv_lmca_cost(lx, labels, eta, k, eta_lx, k_n, margin, push_weight, e);
		if (nv_lmca_progress_flag) {
			printf("nv_lmca: %3d: %ldms, cost: %f, push count avg: %f\n",
				   e, nv_clock() - t,
				   cost,
				   (float)push_sum / push_count
				);
			fflush(stdout);
		}
	}
	nv_lmca_normalize_l(ldm, type);
	
	for (i = 0; i < procs; ++i) {
		nv_matrix_free(&term1[i]);
		nv_matrix_free(&term2[i]);
	}
	nv_free(term1);
	nv_free(term2);
	nv_matrix_free(&lx);
	nv_matrix_free(&dl);
	nv_matrix_free(&dl_old);
	for (i = 0; i < data->m; ++i) {
		nv_free(eta[i]);
		nv_free(eta_lx[i]);
	}
	nv_free(eta);
	nv_free(eta_lx);
}

void
nv_lmca_train(nv_matrix_t *ldm,
			  const nv_matrix_t *data, const nv_matrix_t *labels,
			  int k, int k_n,
			  float margin, float push_weight, float learning_rate,
			  int max_iteration
	)
{
	if (max_iteration > 0) {
		nv_lmca_train_ex(ldm, NV_LMCA_FULL, data, labels, k, k_n, margin, push_weight, learning_rate, max_iteration);
	}
}

void
nv_lmca(nv_matrix_t *ldm,
		const nv_matrix_t *data, const nv_matrix_t *labels,
		int k, int k_n, float margin, float push_weight, float learning_rate,
		int max_iteration)
{
	NV_ASSERT(0 < k);	
	NV_ASSERT(k < k_n);
	NV_ASSERT(push_weight >= 0.0f);
	NV_ASSERT(push_weight <= 1.0f);
	NV_ASSERT(learning_rate > 0.0f);
	NV_ASSERT(margin >= 0.0f); 
	
	/* PCAで初期化 */
	nv_lmca_init_pca(ldm, data);
	/* ldmをkNNの結果がよくなるように更新 */
	nv_lmca_train(ldm, data, labels, k, k_n, margin, push_weight, learning_rate, max_iteration);
}

void
nv_lmca_projection(nv_matrix_t *y, int yj,
				   const nv_matrix_t *ldm,
				   const nv_matrix_t *x, int xj)
{
	int i;
	
	NV_ASSERT(ldm->m == y->n);
	NV_ASSERT(ldm->n == x->n);
	
	for (i = 0; i < ldm->m; ++i) {
		NV_MAT_V(y, yj, i) = nv_vector_dot(ldm, i, x, xj);
	}
}
void
nv_lmca_projection_all(nv_matrix_t *y,
					   const nv_matrix_t *ldm,
					   const nv_matrix_t *x)
{
	int i;
	NV_ASSERT(ldm->m == y->n);
	NV_ASSERT(ldm->n == x->n);
	NV_ASSERT(y->m >= x->m);
	
#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (i = 0; i < x->m; ++i) {
		nv_lmca_projection(y, i, ldm, x, i);
	}
}
