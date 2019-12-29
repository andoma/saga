
SAGA_SRCS += \
	src/onnx.cpp \
	src/graph.cpp \
	src/tensor.cpp \
	src/node.cpp \
	src/onnx.proto3.pb.cpp \

SAGA_SRCS += \
	src/cudnn.cpp \
	src/cuda_tensor.cpp \


OLDS_SRCS += \
	src/network.cpp \
	src/tensor.cpp \
	src/image.cpp \
	src/conv.cpp \
	src/batchnorm.cpp \
	src/fc.cpp \
	src/dropout.cpp \
	src/activation.cpp \
	src/pooling.cpp \
	src/softmax.cpp \
	src/category_classifier.cu \
	src/mathop.cpp \
	src/concat.cpp \
	src/sum.cpp \
	src/gd.cpp \
	src/adam.cu \
