#                                               -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.

AC_PREREQ(2.65)
AC_INIT([nv],[2.1.0],[nagadomi@nurs.or.jp])
NV_MAJOR=2
NV_MINOR=1
AM_INIT_AUTOMAKE([foreign])
AC_CONFIG_SRCDIR([src])
AC_CONFIG_HEADER([src/config.h:src/config.h.in])
AC_CONFIG_MACRO_DIR([m4])

AC_SUBST(CFLAGS)
AC_SUBST(CXXFLAGS)
AC_SUBST(CPPFLAGS)
AC_SUBST(LDFLAGS)
if test "" = "$CFLAGS"; then
    CFLAGS=" "
fi
if test "" = "$CXXFLAGS"; then
    CXXFLAGS=" "
fi
if test "" = "$CPPFLAGS"; then
    CPPFLAGS=" "
fi
if test "" = "$LDFLAGS"; then
    LDFLAGS=" "
fi

CFLAGS=" $CFLAGS -DNV_DLL -DNV_INTERNAL -D__STDC_LIMIT_MACROS -D__STDC_FORMAT_MACROS "
CFLAGS=" $CFLAGS -D_FORTIFY_SOURCE=2 -fstack-protector --param=ssp-buffer-size=4 -Wformat -Werror=format-security "

# Checks for programs.
AC_PROG_CXX
AC_PROG_AWK
AC_PROG_CC
AC_PROG_CPP
AC_PROG_INSTALL
AC_PROG_LN_S
AC_PROG_MAKE_SET
AC_PROG_MKDIR_P

AC_PATH_PROG([DEBUILD], [debuild], [notfound])
AC_PATH_PROG([TEST], [test], [notfound])
AC_PATH_PROG([ECHO], [echo], [notfound])
AC_PATH_PROG([CP], [cp], [notfound])
AC_PATH_PROG([MV], [mv], [notfound])
AC_PATH_PROG([TAR], [tar], [notfound])

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

AC_CHECK_HEADERS([execinfo.h], [AC_CHECK_FUNCS(backtrace backtrace_symbols_fd,[nv_glibc_backtrace=1;CFLAGS="$CFLAGS -rdynamic "])])

nv_deb=0
nv_debsign=0
DPKG_FLAGS=""
AX_ARG_ENABLE(deb,
        AC_HELP_STRING([--enable-deb], [build debian package. ]), [no],
	[nv_deb=1],
        [])
AX_ARG_ENABLE(debsrc,
        AC_HELP_STRING([--enable-debsrc], [build debian source package. ]), [no],
	[nv_deb=1;DPKG_FLAGS=" $DPKG_FLAGS -S -sa "],
        [])
AX_ARG_ENABLE(sign,
        AC_HELP_STRING([--enable-sign], [build signed debian package. ]), [no],
	[nv_deb=1],
        [DPKG_FLAGS=" $DPKG_FLAGS -uc -us "])

AX_ARG_ENABLE(debug,
        AS_HELP_STRING([--enable-debug],[enable assert -g and boundary check.]), [no],
	[nv_strict=1
	CFLAGS=" $CFLAGS -g -D_GLIBCXX_DEBUG -DGLIBCXX_FORCE_NEW "
    ],
    [nv_strict=0
	CFLAGS="$CFLAGS -DNDEBUG -O3 -funroll-loops -ffast-math -g "])

AX_ARG_ENABLE(native,
        AS_HELP_STRING([--disable-native],[disable -march=native]), [yes],
        [CFLAGS=" $CFLAGS -march=native "],
        [])

AX_ARG_ENABLE(ml-test,
        AS_HELP_STRING([--disable-ml-test],[disable machine learning tests]), [yes],
        [CFLAGS=" $CFLAGS -DNV_TEST_ML "],
        [])

AX_ARG_ENABLE(popcnt,
        AS_HELP_STRING([--enable-popcnt],[enable popcnt]), [no],
        [CFLAGS="$CFLAGS -mpopcnt -D__POPCNT__"],
        [])

AX_ARG_ENABLE(profile-generate,
        AS_HELP_STRING([--enable-profile-generate],[enable -fprofile-generate]), [no],
        [CFLAGS=" $CFLAGS -fprofile-generate "
        LDFLAGS=" $LDFLAGS -fprofile-generate "
        ],
        [])
AX_ARG_ENABLE(profile-use,
        AS_HELP_STRING([--enable-profile-use],[enable -fprofile-use]), [no],
        [CFLAGS=" $CFLAGS -fprofile-use "
        LDFLAGS=" $LDFLAGS -fprofile-use "
        ],
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

AX_ARG_ENABLE(openmp,
        AS_HELP_STRING([--disable-openmp],[ disable OpenMP]), [yes],
        [AC_CHECK_HEADERS(
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

AX_ARG_ENABLE(clock,
        AS_HELP_STRING([--disable-clock],[ disable nv_clock]), [yes],
	[nv_clock=1],
	[nv_clock=0]
)
	
AX_ARG_ENABLE(sleep,
        AS_HELP_STRING([--disable-sleep],[ disable nv_sleep]), [yes],
	[nv_sleep=1],
	[nv_sleep=0])

AX_ARG_ENABLE(avxemu,
        AS_HELP_STRING([--enable-avxemu],[ enable avx emulation]), [no],
	[nv_avxemu=1],
	[nv_avxemu=0]
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

# host
AC_CANONICAL_HOST
case $host in
  *-*-mingw* | *-*-cygwin*)
    CFLAGS="$CFLAGS -mstackrealign -mthreads "
    LDFLAGS="$LDFLAGS -Wl,-no-undefined -Wl,--export-all-symbols -Wl,--stack,10485760 "
    ;;
  *-linux*)
    LDFLAGS=" $LDFLAGS -Wl,-z,relro "
    ;;
  *)
    ;;
esac

CPPFLAGS="$CFLAGS"
CXXFLAGS="$CFLAGS"
CFLAGS="$CFLAGS "

AM_CONDITIONAL([BUILDING_DEB], [$TEST $nv_deb = 1])
AC_SUBST(DPKG_FLAGS)
AC_SUBST(nv_avxemu)
AC_SUBST(nv_strict)
AC_SUBST(nv_glibc_backtrace)
AC_SUBST(nv_openssl)
AC_SUBST(nv_eiio)
AC_SUBST(nv_video)
AC_SUBST(nv_opencv)
AC_SUBST(nv_clock)
AC_SUBST(nv_sleep)
AC_SUBST(pkgconfigdir)
AC_SUBST(NV_MAJOR)
AC_SUBST(NV_MINOR)

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
