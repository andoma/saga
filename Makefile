O=build

PROG=saga

PKG_CONFIG ?= pkg-config

CPPFLAGS += -g -O2 -Wall -Werror -I. -I$(O)
CXXFLAGS += --std=c++17 -march=native -fno-exceptions

###########################################
# Lib

SRCS-lib += \
	src/tensor.cpp \
	src/graph.cpp \
	src/node.cpp \

###########################################
# Cuda

HAVE_CUDA := $(subst 0,yes,$(subst 1,no,$(shell $(PKG_CONFIG) cuda-10.1 cudart-10.1; echo $$?)))

SRCS-lib-$(HAVE_CUDA) += \
	src/cuda_dnn.cpp \
	src/cuda_tensor.cpp \
	src/cuda_kernels.cu \

CPPFLAGS-$(HAVE_CUDA) += $(shell pkg-config --cflags cuda-10.1 cudart-10.1)
LDFLAGS-$(HAVE_CUDA)  += $(shell pkg-config --libs   cuda-10.1 cudart-10.1)
LDFLAGS-$(HAVE_CUDA)  += -lnvidia-ml -lcudnn -lcublas

NVCCFLAGS := --std=c++14 -O2 -g -I. -arch sm_53
NVCC := /usr/local/cuda-10.1/bin/nvcc


###########################################
# Onnx & Protobuf

HAVE_PROTOBUF := $(subst 0,yes,$(subst 1,no,$(shell $(PKG_CONFIG) protobuf; echo $$?)))

SRCS-lib-$(HAVE_PROTOBUF) += src/onnx.cpp onnx/onnx.proto3

CPPFLAGS-$(HAVE_PROTOBUF) += $(shell pkg-config --cflags protobuf)
LDFLAGS-$(HAVE_PROTOBUF)  += $(shell pkg-config --libs protobuf)

###########################################
# Program

SRCS += main.cpp \
	test/test_onnx.cpp \
	test/mnist.cpp \
	test/minimal.cpp


###########################################

SRCS += $(SRCS-lib) $(SRCS-lib-yes)

CPPFLAGS += $(CPPFLAGS-yes)
LDFLAGS  += $(LDFLAGS-yes)

OBJS := ${SRCS:%.cpp=${O}/%.o}
OBJS := ${OBJS:%.proto3=${O}/%.proto3.o}
OBJS := ${OBJS:%.cu=${O}/%.o}
DEPS := ${OBJS:%.o=%.d}
SRCDEPS := $(patsubst %,$(O)/%.pb.cc,$(filter %.proto3,$(SRCS)))

ALLDEPS += Makefile

${PROG}: ${OBJS} ${ALLDEPS}
	@mkdir -p $(dir $@)
	${CXX} -o $@ ${OBJS} ${LDFLAGS}

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

-include ${DEPS}

.PRECIOUS: ${O}/%.proto3.pb.cc
