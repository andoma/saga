include saga.mk

PROG=saga

CPPFLAGS += -g -O2 -Wall -Werror -I.
CPPFLAGS += $(shell pkg-config --cflags cuda-10.1 cudart-10.1)
LDFLAGS += $(shell pkg-config --libs cuda-10.1 cudart-10.1)

CXXFLAGS += --std=c++17 -march=native -fno-exceptions

NVCCFLAGS := --std=c++14 -O2 -g -I. -arch sm_53

NVCC := /usr/local/cuda-10.1/bin/nvcc

SRCS += main.cpp \
	test/test_onnx.cpp

SRCS += ${SAGA_SRCS}

LDFLAGS += -lnvidia-ml -lcublas -lcudnn -lprotobuf -lpthread


O=build
OBJS := ${SRCS:%.cpp=${O}/%.o}
OBJS := ${OBJS:%.cu=${O}/%.o}
DEPS := ${OBJS:%.o=%.d}

ALLDEPS = Makefile

${PROG}: ${OBJS} ${ALLDEPS}
	@mkdir -p $(dir $@)
	${CXX} -o $@ ${OBJS} ${LDFLAGS}

${O}/%.o: %.cu ${ALLDEPS}
	@mkdir -p $(dir $@)
	${NVCC} ${NVCCFLAGS} -o $@ -c $<
	${NVCC} -M ${NVCCFLAGS} -o ${@:%.o=%.d} -c $<
	@sed -itmp "s:^$(notdir $@) :$@ :" ${@:%.o=%.d}

${O}/%.o: %.cpp ${ALLDEPS}
	@mkdir -p $(dir $@)
	${CXX} -MD -MP ${CPPFLAGS} ${CXXFLAGS} -o $@ -c $<

clean: rm -rf "${O}" "${PROG}"

-include ${DEPS}
