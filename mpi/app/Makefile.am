AUTOMAKE_OPTIONS = subdir-objects

NVCC = nvcc
NVCFLAGS = -cuda -maxrregcount 32 -ccbin $(CXX)
SUFFIXES = .cu .cpp
.cu.cpp:
	$(NVCC) $(NVCFLAGS) $(INCLUDES) $(CPPFLAGS) --output-file $@ $<

appdir = $(pkglibexecdir)/mpi/app
app_PROGRAMS = osu_matrix_mult

AM_CFLAGS = -I../../util

UTILITIES = ../../util/osu_util.c osu_util.h
if CUDA_KERNELS
UTILITIES += ../../util/kernel.cu
endif

osu_matrix_mult_SOURCES = osu_matrix_mult.c $(UTILITIES)

if EMBEDDED_BUILD
    AM_LDFLAGS =
    AM_CPPFLAGS = -I$(top_builddir)/../src/include \
          -I${top_srcdir}/util \
		  -I${top_srcdir}/../src/include
if BUILD_PROFILING_LIB
    AM_LDFLAGS += $(top_builddir)/../lib/lib@PMPILIBNAME@.la
endif
    AM_LDFLAGS += $(top_builddir)/../lib/lib@MPILIBNAME@.la
endif

if OPENACC
    AM_CFLAGS += -acc -ta=tesla:nordc
    AM_CXXFLAGS = -acc -ta=tesla:nordc
endif
