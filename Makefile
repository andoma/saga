-include local.mk

include mkglue/cuda.mk

PREFIX ?= /usr/local

O ?= build

PROG=${O}/saga
SONAME=libsaga.so.0

PKG_CONFIG ?= pkg-config

CPPFLAGS += -g -O2 -Wall -Werror -I. -I$(O) -Isrc -fPIC
CXXFLAGS += --std=c++17 -march=native -fno-exceptions
CXXFLAGS += -Wno-deprecated-declarations

###########################################
# Lib

SRCS-lib += \
	src/tensor.cpp \
	src/graph.cpp \
	src/node.cpp \
	src/context.cpp \


###########################################
# Dnnl

HAVE_DNNL ?= no

SRCS-lib-$(HAVE_DNNL) += \
	src/dnnl/dnnl.cpp \
	src/dnnl/dnnl_tensor.cpp \


CPPFLAGS-$(HAVE_DNNL) += -I${DNNL_PATH}/include
LDFLAGS-$(HAVE_DNNL)  += -L${DNNL_PATH}/lib -ldnnl

###########################################
# Cuda


NVCCFLAGS := --std=c++14 -O2 -g -I. -arch sm_53

SRCS-lib-$(HAVE_CUDA) += \
	src/cuda/cuda_common.cpp \
	src/cuda/cuda_analysis.cpp \
	src/cuda/cuda_dnn.cpp \
	src/cuda/cuda_tensor.cpp \
	src/cuda/cuda_kernels.cu \

CPPFLAGS += ${CUDA_CPPFLAGS}
LDFLAGS += ${CUDA_LDFLAGS}

LDFLAGS-$(HAVE_CUDA) += -lcudnn -lcublas

###########################################
# Onnx & Protobuf

HAVE_PROTOBUF := $(subst 0,yes,$(shell $(PKG_CONFIG) protobuf; echo $$?))

SRCS-lib-$(HAVE_PROTOBUF) += src/onnx.cpp onnx/onnx.proto3

PKGS-$(HAVE_PROTOBUF) += protobuf

###########################################
# Program

SRCS-prog += \
	main.cpp \
	test/mnist.cpp \
	test/cifar.cpp \
	test/minimal.cpp \
	test/test_classifier.cpp \
	test/test_ops.cpp \
	test/test_jpeg_decoder.cpp \
	test/test_resnode.cpp \

SRCS-prog-$(HAVE_PROTOBUF) += test/test_onnx.cpp

###########################################

SRCS += $(SRCS-lib) $(SRCS-lib-yes)

CPPFLAGS += $(CPPFLAGS-yes)
LDFLAGS  += $(LDFLAGS-yes)

CPPFLAGS += $(shell ${PKG_CONFIG} --cflags ${PKGS} ${PKGS-yes} 2>/dev/null)
LDFLAGS  += $(shell ${PKG_CONFIG} --libs   ${PKGS} ${PKGS-yes} 2>/dev/null)

OBJS := ${SRCS:%.cpp=${O}/%.o}
OBJS := ${OBJS:%.proto3=${O}/%.proto3.o}
OBJS := ${OBJS:%.cu=${O}/%.o}

OBJS-prog := ${SRCS-prog:%.cpp=${O}/%.o} ${SRCS-prog-yes:%.cpp=${O}/%.o}

DEPS := ${OBJS:%.o=%.d} ${OBJS-prog:%.o=%.d}
SRCDEPS := $(patsubst %,$(O)/%.pb.cc,$(filter %.proto3,$(SRCS)))

ALLDEPS += Makefile

${PROG}: ${OBJS} ${OBJS-prog} ${ALLDEPS}
	@mkdir -p $(dir $@)
	${CXX} -o $@ ${OBJS} ${OBJS-prog} ${LDFLAGS}

${O}/${SONAME}: ${OBJS} ${ALLDEPS}
	@mkdir -p $(dir $@)
	${CXX} -shared -Wl,-soname,${SONAME} -Wl,--no-undefined -o $@ ${OBJS} ${LDFLAGS}
	strip --strip-all --discard-all $@

solib: ${O}/${SONAME}

${O}/%.o: %.cu ${ALLDEPS}
	@mkdir -p $(dir $@)
	${NVCC} ${NVCCFLAGS} -o $@ -c $<
	${NVCC} -M ${NVCCFLAGS} -o ${@:%.o=%.d} -c $<
	@sed -itmp "s:^$(notdir $@) :$@ :" ${@:%.o=%.d}

${O}/%.o: %.cpp ${ALLDEPS} | $(SRCDEPS)
	@mkdir -p $(dir $@)
	${CXX} -MD -MP ${CPPFLAGS} ${CXXFLAGS} -o $@ -c $<

${O}/%.o: ${O}/%.pb.cc ${ALLDEPS}
	@mkdir -p $(dir $@)
	${CXX} -MD -MP ${CPPFLAGS} ${CXXFLAGS} -o $@ -c $<

${O}/%.proto3.pb.cc: %.proto3 ${ALLDEPS}
	@mkdir -p $(dir $@)
	protoc --cpp_out=$(O) $<

clean:
	rm -rf "${O}" "${PROG}"

install: ${O}/${SONAME}
	@mkdir -p "${PREFIX}/include/" "${PREFIX}/lib/"
	cp saga.hpp "${PREFIX}/include/saga.hpp"
	cp "${O}/${SONAME}" "${PREFIX}/lib/"
	ln -srf "${PREFIX}/lib/${SONAME}" "${PREFIX}/lib/libsaga.so"
	if [ "x`id -u $$USER`" = "x0" ]; then ldconfig ; fi


-include ${DEPS}

.PRECIOUS: ${O}/%.proto3.pb.cc
