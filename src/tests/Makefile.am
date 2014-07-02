EXTRA_DIST = data/iris_data.vec data/iris_labels.vec data/lena-768x768.jpg data/lena-affine.jpg data/lena-nega.jpg data/lena-rotate.jpg data/lena.jpg data/mnist_test_0-5.vec data/mnist_test_0-5_labels.vec data/mnist_test_14x14.vec data/mnist_test_14x14.vec data/mnist_test_labels.vec

check_PROGRAMS = nv_test
nv_test_SOURCES = nv_test.c nv_test.h \
nv_test_serialize.c \
nv_test_lr.c \
nv_test_arow.c \
nv_test_mlp.c \
nv_test_dae.c \
nv_test_nb.c \
nv_test_keypoint.c \
nv_test_kmeans.c \
nv_test_klr.c \
nv_test_klr_tree.c \
nv_test_kmeans_tree.c \
nv_test_pca_kmeans_tree.c \
nv_test_knb.c \
nv_test_plsi.c \
nv_test_lbgu.c \
nv_test_knn.c \
nv_test_io.c \
nv_test_munkres.c \
nv_test_sha1.c \
nv_test_knn2.c \
nv_test_knn_pca.c \
nv_test_knn_lmca.c \
nv_test_eigen.c \
nv_test_matrix.c

nv_test_CFLAGS= -Wall \
-I$(srcdir)/../ -I$(srcdir)/../3rd_party/sha1 -I$(srcdir)/../3rd_party/mt -I$(srcdir)/../3rd_party/avxintrin_emu \
-DNV_TEST_IMG=\"$(srcdir)/data/lena.jpg\" \
-DNV_TEST_NEGA=\"$(srcdir)/data/lena-nega.jpg\" \
-DNV_TEST_IMG_ROTATE=\"$(srcdir)/data/lena-rotate.jpg\" \
-DNV_TEST_AFFINE=\"$(srcdir)/data/lena-affine.jpg\" \
-DNV_TEST_DATA=\"$(srcdir)/data/mnist_test_14x14.vec\" \
-DNV_TEST_LABEL=\"$(srcdir)/data/mnist_test_labels.vec\" \
-DNV_TEST_DATA_K=10

nv_test_CXXFLAGS = $(nv_test_CFLAGS)
nv_test_LDADD = $(builddir)/../libnv.la


#-DNV_TEST_DATA=\"$(srcdir)/data/mnist_test_0-5.vec\" \
#-DNV_TEST_LABEL=\"$(srcdir)/data/mnist_test_0-5_labels.vec\" \
#

#TESTS = $(check_PROGRAMS) # does not work on automake >= 1.13
check-local:
	$(builddir)/nv_test$(EXEEXT)

CLEANFILES = test.mat test.matb
