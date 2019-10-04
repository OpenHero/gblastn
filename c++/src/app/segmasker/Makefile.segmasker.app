# $Id: Makefile.segmasker.app 208957 2010-10-21 19:23:10Z camacho $

REQUIRES = objects algo

ASN_DEP = seq

APP = segmasker
SRC = segmasker

LIB_ = xobjsimple seqmasks_io xalgosegmask $(BLAST_LIBS) $(OBJMGR_LIBS)
LIB = $(LIB_:%=%$(STATIC))

LIBS = $(CMPRS_LIBS) $(NETWORK_LIBS) $(DL_LIBS) $(ORIG_LIBS)

CXXFLAGS = $(FAST_CXXFLAGS)
LDFLAGS  = $(FAST_LDFLAGS)


WATCHERS = camacho
