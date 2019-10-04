# $Id: Makefile.test_pcre.app 184574 2010-03-02 17:06:58Z gouriano $

REQUIRES = LocalPCRE

SRC = pcretest
APP = test_pcre
PROJ_TAG = test

CPPFLAGS = -I$(includedir)/util/regexp -I$(srcdir)/.. -DHAVE_CONFIG_H $(ORIG_CPPFLAGS)

LIB = $(PCRE_LIB)
LIBS = $(PCRE_LIBS)

CHECK_CMD = test_pcre.sh
CHECK_COPY = testdata test_pcre.sh

WATCHERS = ivanov
