EXTRA_DIST = \
nv_internal.h \
nv_config_msvc.h \
3rd_party/sha1/sha.h \
3rd_party/tinymt/tinymt32.h \
nv.pc.in \
libnv-2.def

lib_LTLIBRARIES = libnv.la
libnv_la_LIBADD = 
libnv_la_CFLAGS = -I$(srcdir)/3rd_party/sha1 -I$(srcdir)/3rd_party/tinymt

libnv_la_CXXFLAGS = $(libnv_la_CFLAGS)
libnv_la_LDFLAGS = -no-undefined -version-number @NV_MAJOR@:@NV_MINOR@:@NV_REVISION@ -rpath "$(libdir)"

libnv_la_SOURCES = \
nv_portable.c \
nv_core_alloc.c \
nv_core_util.c \
nv_core_atomic.c \
nv_core_matrix.c \
nv_core_rand.c \
nv_util_map.cpp \
nv_util_sha1.c \
nv_util_clock.c \
nv_util_2d_index.cpp \
nv_util_array.cpp \
nv_num_cov.c \
nv_num_vector.c \
nv_num_munkres.c \
nv_num_matrix.c \
nv_num_standardize.c \
nv_num_zca.c \
nv_num_distance.c \
nv_num_eigen.c \
nv_num_prime_static.c \
nv_num_knn.cpp \
nv_ml_util.c \
nv_ml_xmeans.c \
nv_ml_kmeans.c \
nv_ml_lr.c \
nv_ml_arow.c \
nv_ml_nb.c \
nv_ml_lbgu.c \
nv_ml_plsi.c \
nv_ml_gaussian.c \
nv_ml_klr.c \
nv_ml_knb.c \
nv_ml_mlp.c \
nv_ml_dae.c \
nv_ml_som.c \
nv_ml_kmeans_tree.c \
nv_ml_pca_kmeans_tree.c \
nv_ml_klr_tree.c \
nv_ml_lmca.c \
nv_ip_gaussian.c \
nv_ip_gray.c \
nv_ip_integral.c \
nv_ip_star_integral.c \
nv_ip_keypoint.cpp \
nv_ip_laplacian.c \
nv_ip_resize.c \
nv_ip_pooling.c \
nv_ip_star_integral_static.c \
nv_ip_flip.c \
nv_ip_euclidean_color.c \
nv_ip_hsv.c \
nv_ip_ccv.c \
nv_ip_bgseg.c \
nv_ip_morphology.c \
nv_ip_patch.c \
nv_io_stream.c \
nv_io_ipl.c \
nv_io_mlp.c \
nv_io_dae.c \
nv_io_lr.c \
nv_io_nb.c \
nv_io_cov.c \
nv_io_matrix.cpp \
nv_io_image.c \
nv_io_libsvm.c \
nv_io_video.c \
nv_io_kmeans_tree.c \
nv_io_klr_tree.c \
nv_fv_rectangle_feature.c \
3rd_party/sha1/sha1.c \
3rd_party/tinymt/tinymt32.c \
3rd_party/getopt/getopt.c

nodist_include_HEADERS = nv_config_auto.h
include_HEADERS = \
nv_core_alloc.h \
nv_config.h \
nv_core_util.h \
nv_core_atomic.h \
nv_core.h \
nv_portable.h \
nv_core_matrix.h \
nv_core_rand.h \
nv_util.h \
nv_util_map.h \
nv_util_clock.h \
nv_util_sha1.h \
nv_util_getopt.h \
nv_util_2d_index.h \
nv_util_array.h \
nv_num_cov.h \
nv_num_vector.h \
nv_num_munkres.h \
nv_num_matrix.h \
nv_num_standardize.h \
nv_num_zca.h \
nv_num_prime.h \
nv_num.h \
nv_num_distance.h \
nv_num_eigen.h \
nv_num_knn.h \
nv_ml_util.h \
nv_ml_xmeans.h \
nv_ml_kmeans.h \
nv_ml_lr.h \
nv_ml_arow.h \
nv_ml_nb.h \
nv_ml_lbgu.h \
nv_ml_plsi.h \
nv_ml_gaussian.h \
nv_ml_klr.h \
nv_ml_knb.h \
nv_ml_mlp.h \
nv_ml_dae.h \
nv_ml_som.h \
nv_ml_kmeans_tree.h \
nv_ml_pca_kmeans_tree.h \
nv_ml_klr_tree.h \
nv_ml_lmca.h \
nv_ml.h \
nv_ip_gaussian.h \
nv_ip_gray.h \
nv_ip_integral.h \
nv_ip_star_integral.h \
nv_ip_keypoint.h \
nv_ip_laplacian.h \
nv_ip_resize.h \
nv_ip_pooling.h \
nv_ip_flip.h \
nv_ip.h \
nv_ip_star_integral_static.h \
nv_ip_euclidean_color.h \
nv_ip_hsv.h \
nv_ip_ccv.h \
nv_ip_bgseg.h \
nv_ip_morphology.h \
nv_ip_patch.h \
nv_io_stream.h \
nv_io_ipl.h \
nv_io_mlp.h \
nv_io_dae.h \
nv_io_lr.h \
nv_io_nb.h \
nv_io_cov.h \
nv_io_matrix.h \
nv_io_libsvm.h \
nv_io.h \
nv_io_image.h \
nv_io_video.h \
nv_io_kmeans_tree.h \
nv_io_klr_tree.h \
nv_fv_rectangle_feature.h

bin_PROGRAMS = nv_dump_lr nv_flags nv_matrix_text2bin nv_kmeans_tree_text2bin

nv_dump_lr_CFLAGS = $(libnv_la_CFLAGS)
nv_dump_lr_CXXFLAGS = $(libnv_la_CXXFLAGS)
nv_dump_lr_LDFLAGS = 
nv_dump_lr_LDADD = libnv.la
nv_dump_lr_SOURCES = util/nv_dump_lr.c

nv_matrix_text2bin_CFLAGS = $(libnv_la_CFLAGS)
nv_matrix_text2bin_CXXFLAGS = $(libnv_la_CXXFLAGS)
nv_matrix_text2bin_LDFLAGS = 
nv_matrix_text2bin_LDADD = libnv.la
nv_matrix_text2bin_SOURCES = util/nv_matrix_text2bin.c

nv_kmeans_tree_text2bin_CFLAGS = $(libnv_la_CFLAGS)
nv_kmeans_tree_text2bin_CXXFLAGS = $(libnv_la_CXXFLAGS)
nv_kmeans_tree_text2bin_LDFLAGS = 
nv_kmeans_tree_text2bin_LDADD = libnv.la
nv_kmeans_tree_text2bin_SOURCES = util/nv_kmeans_tree_text2bin.c

nv_flags_CFLAGS = $(libnv_la_CFLAGS)
nv_flags_LDADD = libnv.la
nv_flags_SOURCES = util/nv_flags.c 

pkgconfigdir = @pkgconfigdir@
pkgconfig_DATA = nv.pc

CLEANFILES = nv.pc
