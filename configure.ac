#                                               -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.

AC_PREREQ(2.61)
AC_INIT([nv],[2.2.0],[nagadomi@nurs.or.jp])
NV_MAJOR=2
NV_MINOR=2
NV_REVISION=0
AM_INIT_AUTOMAKE([foreign subdir-objects])
AC_CONFIG_SRCDIR([src])
AC_CONFIG_HEADER([src/config.h:src/config.h.in])
AC_CONFIG_MACRO_DIR([m4])

AC_SUBST(CFLAGS)
AC_SUBST(CXXFLAGS)
AC_SUBST(CPPFLAGS)
AC_SUBST(LDFLAGS)

CFLAGS="$CFLAGS "
CXXFLAGS="$CXXFLAGS "
LDFLAGS="$LDFLAGS "
CPPFLAGS="$CPPFLAGS "

CFLAGS=" $CFLAGS -DNV_DLL -DNV_INTERNAL -D__STDC_LIMIT_MACROS -D__STDC_FORMAT_MACROS "
CFLAGS=" $CFLAGS -Wall -Wextra -Wformat=2 -Werror=format-security -Wdisabled-optimization -Wno-unused-parameter "
CFLAGS=" $CFLAGS -D_FORTIFY_SOURCE=2 -fstack-protector --param=ssp-buffer-size=4 "

# Checks for programs.
AC_PROG_CXX
AC_PROG_AWK
AC_PROG_CC
AC_PROG_CPP
AC_PROG_INSTALL
AC_PROG_LN_S
AC_PROG_MAKE_SET
AC_PROG_MKDIR_P

# Checks for libraries.
LT_INIT([win32-dll])

#AC_PATH_PROG(PKG_CONFIG, pkg-config, no)
#if test x$PKG_CONFIG = xno ; then
#  AC_MSG_ERROR(*** pkg-config not found. Plese either install pkg-config.)
#fi

# Checks for header files.
AC_HEADER_STDC
AC_CHECK_HEADERS([float.h inttypes.h limits.h stdlib.h string.h sys/time.h unistd.h])

# Checks for typedefs, structures, and compiler characteristics.

AC_CHECK_LIB(m, main)
AC_CHECK_LIB(stdc++, main)

nv_glibc_backtrace=0
AC_CHECK_HEADERS([execinfo.h], [AC_CHECK_FUNCS(backtrace backtrace_symbols_fd,[nv_glibc_backtrace=1])])

AC_CANONICAL_HOST
case $host in
  *-*-mingw* | *-*-cygwin*)
    AC_CHECK_LIB(ssp, main)
    CFLAGS=" $CFLAGS -mstackrealign -mthreads "
    LDFLAGS=" $LDFLAGS -Wl,-no-undefined -Wl,--export-all-symbols -Wl,--stack,10485760 "
    ;;
  *-linux*)
    LDFLAGS=" $LDFLAGS -Wl,-z,relro "
    if test $nv_glibc_backtrace = 1; then
       CFLAGS="$CFLAGS -rdynamic "
    fi
    ;;
  *freebsd*)
    AC_CHECK_LIB(ssp, main)
    if test $nv_glibc_backtrace = 1; then
       CFLAGS="$CFLAGS -rdynamic "
    fi
    ;;
  *apple-darwin*)
    AC_CHECK_LIB(ssp, main)
    ;;
  *)
    ;;
esac

AX_ARG_ENABLE(debug,
        AS_HELP_STRING([--enable-debug],[enable debug symbol and assertion.]), [no],
	[nv_strict=1
	CFLAGS=" $CFLAGS -g -D_GLIBCXX_DEBUG -DGLIBCXX_FORCE_NEW "
    ],
    [nv_strict=0
	CFLAGS="$CFLAGS -DNDEBUG -O3 -funroll-loops -ffast-math -g0 "])

nv_arch=native
AC_ARG_WITH(arch,
	AS_HELP_STRING([--with-arch],
	[processor architecture. native/generic. (default is native)]),
	[nv_arch=${withval}],
	[])
case $nv_arch in
  yes|"")
    nv_arch=native
    ;;
  no)
    nv_arch=generic
    ;;
esac
if test "$nv_arch" != "generic"; then
   CFLAGS=" $CFLAGS -march=$nv_arch "
   AC_MSG_NOTICE(adding -march=$nv_arch)
fi

AX_ARG_ENABLE(ml-test,
        AS_HELP_STRING([--disable-ml-test],[disable machine learning tests]), [yes],
        [CFLAGS=" $CFLAGS -DNV_TEST_ML "],
        [])

AX_ARG_ENABLE(popcnt,
        AS_HELP_STRING([--enable-popcnt],[enable popcnt]), [no],
        [CFLAGS="$CFLAGS -mpopcnt -D__POPCNT__"],
        [])
        
#AX_ARG_ENABLE(opencv,
#        AS_HELP_STRING([--enable-opencv],[ enable nv_conv_ipl2nv and nv_conv_nv2ipl.]), [no],
#        [PKG_CHECK_MODULES(OPENCV, opencv >= 0.22,
#		[nv_opencv=1
#		CFLAGS="$CFLAGS $OPENCV_CFLAGS"
#		LIBS="$LIBS $OPENCV_LIBS"],
#		[AC_MSG_ERROR(opencv not found. please either install opencv or use the --disable-opencv option.)])
#	],
#	[nv_opencv=0])
nv_opencv=0
AX_ARG_ENABLE(opencv,
        AS_HELP_STRING([--enable-opencv],[enable libopencv support]), [no],
        [
                nv_opencv=1
                AC_CHECK_HEADERS([opencv/cv.h],	[], [AC_MSG_ERROR(opencv not found. please either install libopencv-dev or use the --disable-opencv option.)])
                AC_CHECK_HEADERS([opencv/highgui.h],	[], [AC_MSG_ERROR(opencv not found. please either install libopencv-dev or use the --disable-opencv option.)])
                AC_CHECK_HEADERS([opencv/cxcore.h],	[], [AC_MSG_ERROR(opencv not found. please either install libopencv-dev or use the --disable-opencv option.)])
                AC_CHECK_LIB(opencv_core, main, [], [AC_MSG_ERROR(opencv not found. please either install libopencv-dev or use the --disable-opencv option.)])
                AC_CHECK_LIB(opencv_imgproc, main, [], [AC_MSG_ERROR(opencv not found. please either install libopencv-dev or use the --disable-opencv option.)])
                AC_CHECK_LIB(opencv_highgui, main, [], [AC_MSG_ERROR(opencv not found. please either install libopencv-dev or use the --disable-opencv option.)])
	],
	[]
)
nv_openmp=0
AX_ARG_ENABLE(openmp,
        AS_HELP_STRING([--disable-openmp],[ disable OpenMP]), [yes],
        [nv_openmp=1
	AC_CHECK_HEADERS(
		[omp.h],
		[CFLAGS="$CFLAGS -fopenmp ";
		AC_CHECK_LIB(gomp, omp_get_num_procs)],
		[AC_MSG_ERROR(omp.h not found. please either install gcc version 4.2 or any later version or use the --disable-openmp options..)])
	],
	[]
)
nv_openssl=0
AX_ARG_ENABLE(openssl,
        AS_HELP_STRING([--disable-openssl],[ disable SHA1 with OpenSSL]), [yes],
        [AC_CHECK_HEADERS(
		[openssl/sha.h],
		[nv_openssl=1;
		AC_CHECK_LIB(crypto, SHA1_Init)],
		[AC_MSG_ERROR(libssl-dev not found. please either install libssl-dev or use the --disable-openssl options.)])
	],
	[]
)
nv_builtin_sha1=1
if test $nv_openssl = 1; then
  nv_builtin_sha1=0
else
  nv_builtin_sha1=1
fi

nv_video=0
AX_ARG_ENABLE(eiio,
        AS_HELP_STRING([--disable-eiio],[ disable libeiio support]), [yes],
        [
                nv_eiio=1
                AC_CHECK_HEADERS([eiio.h],	[], [AC_MSG_ERROR(eiio not found. please either install libeiio or use the --disable-eiio option.)])
                AC_CHECK_LIB(eiio, eiio_read_file, [], [AC_MSG_ERROR(eiio not found. please either install libeiio or use the --disable-eiio option.)])
                AC_CHECK_LIB(eiio, eiio_video_open, [nv_video=1], [nv_video=0])
	],
	[nv_eiio=0]
)

AC_ARG_WITH(pkgconfigdir,
	AS_HELP_STRING([--with-pkgconfigdir],[use the specified pkgconfig dir (default is libdir/pkgconfig)]),
	[pkgconfigdir=${withval}],
	[pkgconfigdir='${libdir}/pkgconfig'])
AC_MSG_NOTICE([pkgconfig directory is ${pkgconfigdir}])

AC_C_CONST
AC_TYPE_INT16_T
AC_TYPE_INT32_T
AC_TYPE_INT64_T
AC_TYPE_INT8_T
AC_TYPE_SIZE_T
AC_TYPE_UINT16_T
AC_TYPE_UINT32_T
AC_TYPE_UINT64_T
AC_TYPE_UINT8_T
AC_HEADER_TIME
AC_STRUCT_TM
AC_CHECK_HEADERS([float.h inttypes.h limits.h stdlib.h string.h sys/time.h unistd.h wchar.h])

AC_FUNC_MEMCMP
AC_FUNC_VPRINTF
AC_CHECK_FUNCS([gettimeofday memset strchr])

CXXFLAGS=$CFLAGS

AC_SUBST(nv_arch)
AC_SUBST(nv_strict)
AC_SUBST(nv_glibc_backtrace)
AC_SUBST(nv_openssl)
AC_SUBST(nv_eiio)
AC_SUBST(nv_video)
AC_SUBST(nv_builtin_sha1)
AC_SUBST(nv_openmp)
AC_SUBST(nv_opencv)
AC_SUBST(pkgconfigdir)
AC_SUBST(NV_MAJOR)
AC_SUBST(NV_MINOR)
AC_SUBST(NV_REVISION)

AC_CONFIG_FILES([
src/nv.pc
src/nv_config_auto.h
src/Makefile
src/tests/Makefile
vcproj/Makefile
nv_cuda/Makefile
Makefile
])

AC_OUTPUT
echo "libnv $NV_MAJOR.$NV_MINOR.$NV_REVISION:"
echo "   Processor Architecture ............ : $nv_arch"
echo "   Compiling with OpenMP ............. : $nv_openmp"
echo "   Compiling with Debug .............. : $nv_strict"
echo "   Building with OpenCV support....... : $nv_opencv"
echo "   Building with builtin SHA1 function : $nv_builtin_sha1"

