ACLOCAL_AMFLAGS = -I m4
EXTRA_DIST = README.md autogen.sh ChangeLog COPYING m4 .gitignore 
SUBDIRS    = src nv_cuda src/tests vcproj

valgrind:
	cd src/tests; LD_LIBRARY_PATH=../.libs valgrind -v --tool=memcheck --leak-check=full --show-reachable=yes --run-libc-freeres=no --track-origins=yes --log-file=./valgrind.log ./.libs/nv_test ; cat valgrind.log; cd -

nv_test:
	cd src/tests; make nv_test

check-human: all nv_test
	cd src/tests; LD_LIBRARY_PATH=../.libs ./.libs/nv_test

if BUILDING_DEB
DEBSOURCEPKG=$(PACKAGE)_$(VERSION).orig.tar.gz
DEBSOURCEDIR=$(PACKAGE)-$(VERSION)
DEBIANDIR=packages/debian
deb: dist
	@$(ECHO) "Building Debian package..."; \
	if $(TEST) ! -d $(DEBIANDIR); then \
		$(ECHO) "The files needed for building a Debian package are not" \
		"included by default in the distribution. To build a package, check" \
		"out the project from source control."; \
		exit 1; \
	fi; \
	$(MV) $(PACKAGE)-$(VERSION).tar.gz $(DEBSOURCEPKG) && \
	$(TAR) --extract --gunzip --file $(DEBSOURCEPKG) && \
	$(MKDIR_P) $(DEBSOURCEDIR)/debian && \
	$(CP) -r $(DEBIANDIR) $(DEBSOURCEDIR) && \
	cd $(DEBSOURCEDIR) && \
	$(DEBUILD) $(DPKG_FLAGS); \
	cd ..;
else
deb:
	@$(ECHO) \
	"To build a Debian package, you must run configure with --enable-deb."; \
	exit 1
endif
clean-local:
	-rm -rf $(DEBSOURCEDIR)

CLEANFILES = *.deb *.debian.tar.gz *.dsc *.changes *.build *.upload $(DEBSOURCEPKG) 