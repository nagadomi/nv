ACLOCAL_AMFLAGS = -I m4
EXTRA_DIST = README.md autogen.sh ChangeLog COPYING m4 .gitignore 
SUBDIRS    = src nv_cuda src/tests vcproj

valgrind: all nv_test
	cd src/tests; LD_LIBRARY_PATH=../.libs valgrind -v --tool=memcheck --leak-check=full --show-reachable=yes --run-libc-freeres=no --track-origins=yes --log-file=./valgrind.log ./.libs/nv_test ; cat valgrind.log; cd -

nv_test:
	cd src/tests; make nv_test

gdb: nv_test
	cd $(abs_top_builddir)/src/tests && LD_LIBRARY_PATH="$(LD_LIBRARY_PATH):../.libs" gdb ./.libs/nv_test
