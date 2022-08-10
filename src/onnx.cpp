/*
 * Copyright (c) 2019, Andreas Smas
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include <sstream>
#include <iostream>
#include <memory>
#include <map>

#include <inttypes.h>

#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>

#include "onnx/onnx.proto3.pb.h"

#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/io/coded_stream.h>

#include "saga.hpp"

using namespace google::protobuf::io;
using namespace std;

namespace saga {

static const char *
DataType_str(int dt)
{
    switch((onnx::TensorProto_DataType)dt) {
    case onnx::TensorProto_DataType_UNDEFINED:
        return "UNDEFINED";
    case onnx::TensorProto_DataType_FLOAT:
        return "FLOAT";
    case onnx::TensorProto_DataType_UINT8:
        return "UINT8";
    case onnx::TensorProto_DataType_INT8:
        return "INT8";
    case onnx::TensorProto_DataType_UINT16:
        return "UINT16";
    case onnx::TensorProto_DataType_INT16:
        return "INT16";
    case onnx::TensorProto_DataType_INT32:
        return "INT32";
    case onnx::TensorProto_DataType_INT64:
        return "INT64";
    case onnx::TensorProto_DataType_STRING:
        return "STRING";
    case onnx::TensorProto_DataType_BOOL:
        return "BOOL";
    case onnx::TensorProto_DataType_FLOAT16:
        return "FLOAT16";
    case onnx::TensorProto_DataType_DOUBLE:
        return "DOUBLE";
    case onnx::TensorProto_DataType_UINT32:
        return "UINT32";
    case onnx::TensorProto_DataType_UINT64:
        return "UINT64";
    case onnx::TensorProto_DataType_COMPLEX64:
        return "COMPLEX64";
    case onnx::TensorProto_DataType_COMPLEX128:
        return "COMPLEX128";
    case onnx::TensorProto_DataType_BFLOAT16:
        return "BFLOAT16";
    default:
        return "???";
    }
}

static const char *
AttributeType_str(int at)
{
    switch((onnx::AttributeProto_AttributeType)at) {
    case onnx::AttributeProto_AttributeType_UNDEFINED:
        return "UNDEFINED";
    case onnx::AttributeProto_AttributeType_FLOAT:
        return "FLOAT";
    case onnx::AttributeProto_AttributeType_INT:
        return "INT";
    case onnx::AttributeProto_AttributeType_STRING:
        return "STRING";
    case onnx::AttributeProto_AttributeType_TENSOR:
        return "TENSOR";
    case onnx::AttributeProto_AttributeType_GRAPH:
        return "GRAPH";
    case onnx::AttributeProto_AttributeType_FLOATS:
        return "FLOATS";
    case onnx::AttributeProto_AttributeType_INTS:
        return "INTS";
    case onnx::AttributeProto_AttributeType_STRINGS:
        return "STRINGS";
    case onnx::AttributeProto_AttributeType_TENSORS:
        return "TENSORS";
    case onnx::AttributeProto_AttributeType_GRAPHS:
        return "GRAPHS";
    default:
        return "???";
    }
}

static std::string
TensorShapeProto_str(const onnx::TensorShapeProto &tsp)
{
    std::ostringstream b;
    for(int i = 0; i < tsp.dim_size(); i++) {
        b << (i ? ", " : "{") << tsp.dim(i).dim_value();
    }
    b << "}";
    return b.str();
}

static std::string
TypeProto_str(const onnx::TypeProto &tp)
{
    std::ostringstream b;
    b << DataType_str(tp.tensor_type().elem_type()) << ":";
    b << TensorShapeProto_str(tp.tensor_type().shape());
    return b.str();
}

static std::string
ValueInfoProto_str(const onnx::ValueInfoProto &vip)
{
    std::ostringstream b;
    b << vip.name();
    b << ":[" << TypeProto_str(vip.type()) << "] ";
    return b.str();
}

static void
NodeProto_print(const onnx::NodeProto &np)
{
    printf("  Node %s\n", np.op_type().c_str());
    for(int j = 0; j < np.input_size(); j++) {
        printf("      Input %3d: %s\n", j, np.input(j).c_str());
    }
    for(int j = 0; j < np.output_size(); j++) {
        printf("     Output %3d: %s\n", j, np.output(j).c_str());
    }
    for(int j = 0; j < np.attribute_size(); j++) {
        const auto &a = np.attribute(j);
        assert(a.ref_attr_name()[0] == 0);
        printf("    Attribute %d: [Type:%s] %s = ", j,
               AttributeType_str(a.type()), a.name().c_str());
        switch(a.type()) {
        case onnx::AttributeProto_AttributeType_INT:
            printf("%d\n", (int)a.i());
            break;
        case onnx::AttributeProto_AttributeType_FLOAT:
            printf("%f\n", a.f());
            break;
        case onnx::AttributeProto_AttributeType_INTS:
            for(const auto &i : a.ints()) {
                printf("%d ", (int)i);
            }
            printf("\n");
            break;
        default:
            printf("?\n");
        }
    }
}

void
print_onnx_graph_info(const onnx::GraphProto &gp)
{
    printf("Graph name: %s\n", gp.name().c_str());

    for(int i = 0; i < gp.node_size(); i++) {
        NodeProto_print(gp.node(i));
    }

    for(int i = 0; i < gp.initializer_size(); i++) {
        const auto &ini = gp.initializer(i);
        printf("  Initializer %d: %s\n", i, ini.name().c_str());
        printf("    %s: ", DataType_str(ini.data_type()));
        if(ini.dims_size() == 0) {
            printf("<scalar> ");
        } else {
            for(int j = 0; j < ini.dims_size(); j++) {
                printf("%s%" PRId64, j ? ", " : "{", ini.dims(j));
            }
            printf("} ");
        }
        assert(ini.segment().begin() == 0);
        assert(ini.segment().end() == 0);
        printf(" size:%zd\n", ini.raw_data().size());
    }

    for(int i = 0; i < gp.input_size(); i++) {
        printf("  Input %d: %s\n", i, ValueInfoProto_str(gp.input(i)).c_str());
    }

    for(int i = 0; i < gp.output_size(); i++) {
        printf("  Output %d: %s\n", i,
               ValueInfoProto_str(gp.output(i)).c_str());
    }

    for(int i = 0; i < gp.value_info_size(); i++) {
        printf("  ValueInfo %d: %s\n", i,
               ValueInfoProto_str(gp.value_info(i)).c_str());
    }
}

struct MappedPBFile : public CodedInputStream {
public:
    MappedPBFile(void *data, size_t size)
      : CodedInputStream((const uint8_t *)data, size), data_(data), size_(size)
    {
#if GOOGLE_PROTOBUF_VERSION >= 3002000
        SetTotalBytesLimit(size);
#else
        SetTotalBytesLimit(size, size);
#endif
    }

    ~MappedPBFile() { munmap(data_, size_); }

private:
    void *data_;
    size_t size_;
};

static unique_ptr<MappedPBFile>
mapPBfile(const char *path)
{
    const int fd = open(path, O_RDONLY);
    if(fd == -1) {
        fprintf(stderr, "Failed to open protobuf file %s: %s\n", path,
                strerror(errno));
        return nullptr;
    }

    struct stat st;
    if(fstat(fd, &st) == -1) {
        fprintf(stderr, "Failed to stat protobuf file %s: %s\n", path,
                strerror(errno));
        close(fd);
        return nullptr;
    }

    void *mem = mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if(mem == MAP_FAILED) {
        fprintf(stderr, "Failed to map protobuf file %s: %s\n", path,
                strerror(errno));
        close(fd);
        return nullptr;
    }

    close(fd);

    return make_unique<MappedPBFile>(mem, st.st_size);
}

//------------------------------------------------------------------------
//

static Tensor::DataType
DataType_map(int dt)
{
    switch((onnx::TensorProto_DataType)dt) {
    case onnx::TensorProto_DataType_FLOAT:
        return Tensor::DataType::FLOAT;
    case onnx::TensorProto_DataType_UINT8:
        return Tensor::DataType::U8;
    case onnx::TensorProto_DataType_INT64:
        return Tensor::DataType::INT64;
    case onnx::TensorProto_DataType_FLOAT16:
        return Tensor::DataType::HALF;
    default:
        fprintf(stderr, "ONNX: No mapping for data-type %s\n",
                DataType_str(dt));
        abort();
    }
}

static shared_ptr<Tensor>
make_tensor(const onnx::ValueInfoProto &vip)
{
    const auto &tp = vip.type();
    if(!tp.has_tensor_type())
        return nullptr;

    const auto &shape = tp.tensor_type().shape();

    Dims dims;
    for(const auto &dim : shape.dim()) {
        int d;
        switch(dim.value_case()) {
        case onnx::TensorShapeProto_Dimension::kDimValue:
            d = dim.dim_value();
            break;
        case onnx::TensorShapeProto_Dimension::kDimParam:
            d = 1;
            break;
        default:
            return nullptr;
        }
        dims.push_back(d);
    }
    return makeTensor(DataType_map(tp.tensor_type().elem_type()), dims,
                      vip.name());
}

static shared_ptr<Tensor>
make_tensor(const onnx::TensorProto &tp)
{
    Dims dims;
    for(const auto &dim : tp.dims()) {
        dims.push_back(dim);
    }

    auto t = makeCPUTensor(DataType_map(tp.data_type()), dims, tp.name());

    auto ta = t->access();

    if(tp.raw_data().size()) {
        memcpy(ta->data(), (const void *)&tp.raw_data()[0],
               tp.raw_data().size());
    } else if(tp.float_data_size()) {
        std::vector<float> v(tp.float_data().begin(), tp.float_data().end());
        memcpy(ta->data(), (const void *)&v[0], v.size() * sizeof(float));
    } else if(tp.int64_data_size()) {
        std::vector<int64_t> v(tp.int64_data().begin(), tp.int64_data().end());
        memcpy(ta->data(), (const void *)&v[0], v.size() * sizeof(int64_t));
    } else {
        fprintf(stderr,
                "Unable to load %s: Can't load data format (please fix)\n",
                tp.name().c_str());
        abort();
    }
    return t;
}

static shared_ptr<Tensor>
find_tensor(Graph &g, const std::string &name)
{
    auto it = g.m_named_tensors->find(name);
    if(it != g.m_named_tensors->end())
        return it->second;
    return nullptr;
}

static void
make_tensor_y(Graph &g, Node &n, const std::string &name)
{
    auto t = n.inferTensor_y(name);
    (*g.m_named_tensors)[name] = t;
    n.outputs_["y"] = t;
}

static shared_ptr<Node>
make_conv(Graph &g, const onnx::NodeProto &np, const Attributes &attribs)
{
    assert(np.input_size() > 1);
    assert(np.output_size() == 1);

    auto n = std::make_shared<Node>("conv");
    n->inputs_["x"] = find_tensor(g, np.input(0));

    n->inputs_["w"] = find_tensor(g, np.input(1));
    if(np.input_size() > 2)
        n->inputs_["b"] = find_tensor(g, np.input(2));

    assert(attribs.find("auto_pad") == attribs.end());  // Can't deal with this

    int group = attribs.get("group", 1);
    assert(group == 1);

    for(auto d : attribs.get("dilations", std::vector<int>({}))) {
        assert(d == 1);
    }

    int pad = 0;
    auto pads = attribs.get("pads", std::vector<int>({}));

    if(pads.size()) {
        pad = pads[0];
        for(auto p : pads) {
            assert(p == pad);
        }
    }

    n->attributes_["pad"] = pad;

    int stride = 1;
    auto strides = attribs.get("strides", std::vector<int>({}));

    if(strides.size()) {
        stride = strides[0];
        for(auto s : strides) {
            assert(s == stride);
        }
    }
    n->attributes_["stride"] = stride;

    make_tensor_y(g, *n, np.output(0));
    return n;
}

static shared_ptr<Node>
make_batchnorm(Graph &g, const onnx::NodeProto &np, const Attributes &attribs)
{
    assert(np.input_size() == 5);
    assert(np.output_size() == 1);

    auto n = std::make_shared<Node>("batchnorm");
    n->inputs_["x"] = find_tensor(g, np.input(0));

    n->inputs_["s"] = find_tensor(g, np.input(1));
    n->inputs_["b"] = find_tensor(g, np.input(2));
    n->inputs_["m"] = find_tensor(g, np.input(3));
    n->inputs_["v"] = find_tensor(g, np.input(4));

    float epsilon = attribs.get("epsilon", 1e-05f);
    n->attributes_["epsilon"] = epsilon;

    make_tensor_y(g, *n, np.output(0));
    return n;
}

static shared_ptr<Node>
make_relu(Graph &g, const onnx::NodeProto &np, const Attributes &attribs)
{
    assert(np.input_size() == 1);
    assert(np.output_size() == 1);
    auto n = std::make_shared<Node>("relu");
    n->inputs_["x"] = find_tensor(g, np.input(0));
    make_tensor_y(g, *n, np.output(0));
    return n;
}

static shared_ptr<Node>
make_leakyrelu(Graph &g, const onnx::NodeProto &np, const Attributes &attribs)
{
    assert(np.input_size() == 1);
    assert(np.output_size() == 1);
    auto n = std::make_shared<Node>("leakyrelu");
    n->inputs_["x"] = find_tensor(g, np.input(0));
    float alpha = attribs.get("alpha", 0.01f);
    n->attributes_["alpha"] = alpha;
    make_tensor_y(g, *n, np.output(0));
    return n;
}

static shared_ptr<Node>
make_pooling(Graph &g, const onnx::NodeProto &np, const Attributes &attribs,
             const std::string &type, bool global)
{
    assert(np.input_size() == 1);
    assert(np.output_size() == 1);
    auto n = std::make_shared<Node>(type);
    n->inputs_["x"] = find_tensor(g, np.input(0));

    assert(attribs.find("auto_pad") == attribs.end());  // Can't deal with this
    assert(attribs.get("ceil_mode", 0) == 0);
    for(auto d : attribs.get("dilations", std::vector<int>({}))) {
        assert(d == 1);
    }

    if(global) {
        n->attributes_["global"] = true;
    } else {
        int size = 0;
        auto kernel_shape = attribs.get("kernel_shape", std::vector<int>({}));
        if(kernel_shape.size()) {
            size = kernel_shape[0];
            for(auto x : kernel_shape) {
                assert(x == size);
            }
        }
        n->attributes_["size"] = size;
    }

    int pad = 0;
    auto pads = attribs.get("pads", std::vector<int>({}));
    if(pads.size()) {
        pad = pads[0];
        for(auto d : pads) {
            assert(d == pad);
        }
    }

    assert(attribs.get("storage_order", 0) == 0);

    int stride = 1;
    auto strides = attribs.get("strides", std::vector<int>({}));

    if(strides.size()) {
        stride = strides[0];
        for(auto s : strides) {
            assert(s == stride);
        }
    }

    n->attributes_["pad"] = pad;
    n->attributes_["stride"] = stride;
    make_tensor_y(g, *n, np.output(0));
    return n;
}

static shared_ptr<Node>
make_mathop(Graph &g, const onnx::NodeProto &np, const Attributes &attribs,
            const std::string &type)
{
    assert(np.input_size() == 2);
    assert(np.output_size() == 1);
    auto n = std::make_shared<Node>(type);
    n->inputs_["x"] = find_tensor(g, np.input(0));
    n->inputs_["b"] = find_tensor(g, np.input(1));
    make_tensor_y(g, *n, np.output(0));
    return n;
}

static shared_ptr<Node>
make_sum(Graph &g, const onnx::NodeProto &np, const Attributes &attribs)
{
    assert(np.input_size() >= 1);
    assert(np.output_size() == 1);
    auto n = std::make_shared<Node>("sum");

    for(int i = 0; i < np.input_size(); i++) {
        char name[20];
        snprintf(name, sizeof(name), "x%u", i);
        n->inputs_[name] = find_tensor(g, np.input(i));
    }
    make_tensor_y(g, *n, np.output(0));
    return n;
}

static shared_ptr<Node>
make_concat(Graph &g, const onnx::NodeProto &np, const Attributes &attribs)
{
    assert(np.input_size() >= 1);
    assert(np.output_size() == 1);
    auto n = std::make_shared<Node>("concat");

    for(int i = 0; i < np.input_size(); i++) {
        char name[20];
        snprintf(name, sizeof(name), "x%u", i);
        n->inputs_[name] = find_tensor(g, np.input(i));
    }
    make_tensor_y(g, *n, np.output(0));
    return n;
}

static shared_ptr<Node>
make_reshape(Graph &g, const onnx::NodeProto &np, const Attributes &attribs)
{
    assert(np.input_size() == 2);
    assert(np.output_size() == 1);
    auto n = std::make_shared<Node>("reshape");
    n->inputs_["x"] = find_tensor(g, np.input(0));

    auto shape = find_tensor(g, np.input(1));
    auto ta = shape->access();
    Dims a;
    for(int i = 0; i < shape->dims_[0]; i++) {
        const int64_t v = ta->get({i});
        if(v == 0) {
            a.push_back(DimParam::UNCHANGED);
        } else if(v == -1) {
            a.push_back(DimParam::REDUCE);
        } else if(v > 0) {
            a.push_back(v);
        }
    }

    n->attributes_["shape"] = a;
    make_tensor_y(g, *n, np.output(0));
    return n;
}

static shared_ptr<Node>
make_flatten(Graph &g, const onnx::NodeProto &np, const Attributes &attribs)
{
    assert(np.input_size() == 1);
    assert(np.output_size() == 1);
    auto n = std::make_shared<Node>("reshape");
    auto x = find_tensor(g, np.input(0));
    n->inputs_["x"] = x;
    const int rank = x->dims_.size();
    int axis = attribs.get("axis", 1);
    if(axis < 0)
        axis = rank + axis;

    Dims shape(axis + 1, DimParam::UNCHANGED);
    shape[axis] = DimParam::REDUCE;
    n->attributes_["shape"] = shape;

    make_tensor_y(g, *n, np.output(0));
    return n;
}

static shared_ptr<Node>
make_gemm(Graph &g, const onnx::NodeProto &np, const Attributes &attribs)
{
    assert(np.input_size() == 3);
    assert(np.output_size() == 1);
    auto n = std::make_shared<Node>("fc");
    n->inputs_["x"] = find_tensor(g, np.input(0));
    n->inputs_["w"] = find_tensor(g, np.input(1));
    n->inputs_["b"] = find_tensor(g, np.input(2));
    // Confusing but transB maps to second tensor which is Weight for us
    n->attributes_["transW"] = attribs.get("transB", 0) ? true : false;
    make_tensor_y(g, *n, np.output(0));
    return n;
}

static shared_ptr<Node>
make_matmul(Graph &g, const onnx::NodeProto &np, const Attributes &attribs)
{
    assert(np.input_size() == 2);
    assert(np.output_size() == 1);
    auto n = std::make_shared<Node>("fc");
    n->inputs_["x"] = find_tensor(g, np.input(0));
    n->inputs_["w"] = find_tensor(g, np.input(1));
    make_tensor_y(g, *n, np.output(0));
    return n;
}

static shared_ptr<Node>
make_softmax(Graph &g, const onnx::NodeProto &np, const Attributes &attribs)
{
    assert(np.input_size() == 1);
    assert(np.output_size() == 1);
    auto n = std::make_shared<Node>("softmax");
    n->inputs_["x"] = find_tensor(g, np.input(0));
    make_tensor_y(g, *n, np.output(0));
    return n;
}

static shared_ptr<Node>
make_dropout(Graph &g, const onnx::NodeProto &np, const Attributes &attribs)
{
    assert(np.input_size() == 1);
    assert(np.output_size() == 1);
    auto n = std::make_shared<Node>("dropout");
    n->inputs_["x"] = find_tensor(g, np.input(0));
    make_tensor_y(g, *n, np.output(0));
    return n;
}

static bool
loadgraph(Graph &g, const onnx::GraphProto &gp)
{
    for(const auto &vip : gp.input()) {
        auto t = make_tensor(vip);
        (*g.m_named_tensors)[vip.name()] = t;
        g.inputs_.insert(t);
    }

    for(const auto &tp : gp.initializer()) {
        auto it = g.m_named_tensors->find(tp.name());
        if(it != g.m_named_tensors->end()) {
            g.inputs_.erase(it->second);
        }
        (*g.m_named_tensors)[tp.name()] = make_tensor(tp);
    }

    for(const auto &np : gp.node()) {
        assert(np.output_size() == 1);

        Attributes attribs;

        for(const auto &a : np.attribute()) {
            switch(a.type()) {
            case onnx::AttributeProto_AttributeType_INT:
                attribs[a.name()] = (int)a.i();
                break;
            case onnx::AttributeProto_AttributeType_FLOAT:
                attribs[a.name()] = a.f();
                break;
            case onnx::AttributeProto_AttributeType_INTS:
                attribs[a.name()] =
                    vector<int>(a.ints().begin(), a.ints().end());
                break;
            default:
                break;
            }
        }

        shared_ptr<Node> n;
        const auto &node_type = np.op_type();
        if(node_type == "Conv") {
            n = make_conv(g, np, attribs);
        } else if(node_type == "BatchNormalization") {
            n = make_batchnorm(g, np, attribs);
        } else if(node_type == "Relu") {
            n = make_relu(g, np, attribs);
        } else if(node_type == "LeakyRelu") {
            n = make_leakyrelu(g, np, attribs);
        } else if(node_type == "MaxPool") {
            n = make_pooling(g, np, attribs, "maxpool", false);
        } else if(node_type == "AveragePool") {
            n = make_pooling(g, np, attribs, "avgpool", false);
        } else if(node_type == "GlobalAveragePool") {
            n = make_pooling(g, np, attribs, "avgpool", true);
        } else if(node_type == "Add") {
            n = make_mathop(g, np, attribs, "add");
        } else if(node_type == "Mul") {
            n = make_mathop(g, np, attribs, "mul");
        } else if(node_type == "MatMul") {
            n = make_matmul(g, np, attribs);
        } else if(node_type == "Sum") {
            n = make_sum(g, np, attribs);
        } else if(node_type == "Concat") {
            n = make_concat(g, np, attribs);
        } else if(node_type == "Reshape") {
            n = make_reshape(g, np, attribs);
        } else if(node_type == "Flatten") {
            n = make_flatten(g, np, attribs);
        } else if(node_type == "Gemm") {
            n = make_gemm(g, np, attribs);
        } else if(node_type == "Softmax") {
            n = make_softmax(g, np, attribs);
        } else if(node_type == "Dropout") {
            n = make_dropout(g, np, attribs);
        } else {
            fprintf(stderr, "Can't handle node type %s\n", node_type.c_str());
            NodeProto_print(np);
            abort();
            return false;
        }
        g.nodes_.push_back(n);
    }

    for(const auto &vip : gp.output()) {
        g.outputs_.insert((*g.m_named_tensors)[vip.name()]);
    }

    return true;
}

std::shared_ptr<Graph>
Graph::load(const char *path)
{
    auto pb = mapPBfile(path);
    if(pb == NULL)
        return nullptr;

    onnx::ModelProto mp;
    if(!mp.ParseFromCodedStream(pb.get()))
        return nullptr;

    const auto &gp = mp.graph();

    if(0)
        print_onnx_graph_info(gp);

    auto g = make_shared<Graph>();

    if(!loadgraph(*g.get(), gp))
        return nullptr;

    return g;
}

std::shared_ptr<Tensor>
Tensor::loadProtoBuf(const char *path)
{
    auto pb = mapPBfile(path);
    if(pb == NULL)
        return nullptr;

    onnx::TensorProto tp;
    if(!tp.ParseFromCodedStream(pb.get()))
        return nullptr;

    return make_tensor(tp);
}

}  // namespace saga
