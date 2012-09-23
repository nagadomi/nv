ACLOCAL_AMFLAGS = -I m4
EXTRA_DIST = README.md autogen.sh ChangeLog COPYING m4 .gitignore 
SUBDIRS    = src nv_cuda src/tests vcproj

valgrind:
	cd src/tests; LD_LIBRARY_PATH=../.libs valgrind -v --tool=memcheck --leak-check=full --show-reachable=yes --run-libc-freeres=no --log-file=./valgrind.log ./.libs/nv_test ; cat valgrind.log; cd -

