O=build

PROG=saga
LIB=libsaga.a

CUDA_VERSION ?= 10.2

PKG_CONFIG ?= pkg-config

CPPFLAGS += -g -O2 -Wall -Werror -I. -I$(O)
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
# Cuda

HAVE_CUDA := $(subst 0,yes,$(subst 1,no,$(shell $(PKG_CONFIG) cuda-${CUDA_VERSION} cudart-${CUDA_VERSION}; echo $$?)))

SRCS-lib-$(HAVE_CUDA) += \
	src/cuda_dnn.cpp \
	src/cuda_tensor.cpp \
	src/cuda_kernels.cu \

CPPFLAGS-$(HAVE_CUDA) += $(shell pkg-config --cflags cuda-${CUDA_VERSION} cudart-${CUDA_VERSION})
LDFLAGS-$(HAVE_CUDA)  += $(shell pkg-config --libs   cuda-${CUDA_VERSION} cudart-${CUDA_VERSION})
LDFLAGS-$(HAVE_CUDA)  += -lnvidia-ml -lcudnn -lcublas

NVCCFLAGS := --std=c++14 -O2 -g -I. -arch sm_53
NVCC := /usr/local/cuda-${CUDA_VERSION}/bin/nvcc


###########################################
# Onnx & Protobuf

HAVE_PROTOBUF := $(subst 0,yes,$(subst 1,no,$(shell $(PKG_CONFIG) protobuf; echo $$?)))

SRCS-lib-$(HAVE_PROTOBUF) += src/onnx.cpp onnx/onnx.proto3

CPPFLAGS-$(HAVE_PROTOBUF) += $(shell pkg-config --cflags protobuf)
LDFLAGS-$(HAVE_PROTOBUF)  += $(shell pkg-config --libs protobuf)

###########################################
# Program

SRCS-prog += \
	main.cpp \
	test/test_onnx.cpp \
	test/mnist.cpp \
	test/cifar.cpp \
	test/minimal.cpp \
	test/test_classifier.cpp \


###########################################

SRCS += $(SRCS-lib) $(SRCS-lib-yes)

CPPFLAGS += $(CPPFLAGS-yes)
LDFLAGS  += $(LDFLAGS-yes)

OBJS := ${SRCS:%.cpp=${O}/%.o}
OBJS := ${OBJS:%.proto3=${O}/%.proto3.o}
OBJS := ${OBJS:%.cu=${O}/%.o}

OBJS-prog := ${SRCS-prog:%.cpp=${O}/%.o}

DEPS := ${OBJS:%.o=%.d} ${OBJS-prog:%.o=%.d}
SRCDEPS := $(patsubst %,$(O)/%.pb.cc,$(filter %.proto3,$(SRCS)))

ALLDEPS += Makefile

${PROG}: ${OBJS} ${OBJS-prog} ${ALLDEPS}
	@mkdir -p $(dir $@)
	${CXX} -o $@ ${OBJS} ${OBJS-prog} ${LDFLAGS}

${LIB}:  ${OBJS} ${ALLDEPS}
	@mkdir -p $(dir $@)
	ar rcs $@ ${OBJS}
	ranlib $@

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

lib: ${LIB}

-include ${DEPS}

.PRECIOUS: ${O}/%.proto3.pb.cc
