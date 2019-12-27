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

#include <iostream>
#include <memory>
#include <map>

#include <inttypes.h>

#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>

#include "common.h"

#include "onnx.proto3.pb.h"

#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/io/coded_stream.h>

using namespace google::protobuf::io;
using namespace std;

namespace saga {


static const char *
DataType_str(int dt)
{
  switch((onnx::TensorProto_DataType)dt) {
  case onnx::TensorProto_DataType_UNDEFINED: return "UNDEFINED";
  case onnx::TensorProto_DataType_FLOAT: return "FLOAT";
  case onnx::TensorProto_DataType_UINT8: return "UINT8";
  case onnx::TensorProto_DataType_INT8: return "INT8";
  case onnx::TensorProto_DataType_UINT16: return "UINT16";
  case onnx::TensorProto_DataType_INT16: return "INT16";
  case onnx::TensorProto_DataType_INT32: return "INT32";
  case onnx::TensorProto_DataType_INT64: return "INT64";
  case onnx::TensorProto_DataType_STRING: return "STRING";
  case onnx::TensorProto_DataType_BOOL: return "BOOL";
  case onnx::TensorProto_DataType_FLOAT16: return "FLOAT16";
  case onnx::TensorProto_DataType_DOUBLE: return "DOUBLE";
  case onnx::TensorProto_DataType_UINT32: return "UINT32";
  case onnx::TensorProto_DataType_UINT64: return "UINT64";
  case onnx::TensorProto_DataType_COMPLEX64: return "COMPLEX64";
  case onnx::TensorProto_DataType_COMPLEX128: return "COMPLEX128";
  case onnx::TensorProto_DataType_BFLOAT16: return "BFLOAT16";
  default:
    return "???";
  }
}


static const char *
AttributeType_str(int at)
{
  switch((onnx::AttributeProto_AttributeType)at) {
  case onnx::AttributeProto_AttributeType_UNDEFINED: return "UNDEFINED";
  case onnx::AttributeProto_AttributeType_FLOAT: return "FLOAT";
  case onnx::AttributeProto_AttributeType_INT: return "INT";
  case onnx::AttributeProto_AttributeType_STRING: return "STRING";
  case onnx::AttributeProto_AttributeType_TENSOR: return "TENSOR";
  case onnx::AttributeProto_AttributeType_GRAPH: return "GRAPH";
  case onnx::AttributeProto_AttributeType_FLOATS: return "FLOATS";
  case onnx::AttributeProto_AttributeType_INTS: return "INTS";
  case onnx::AttributeProto_AttributeType_STRINGS: return "STRINGS";
  case onnx::AttributeProto_AttributeType_TENSORS: return "TENSORS";
  case onnx::AttributeProto_AttributeType_GRAPHS: return "GRAPHS";
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
    printf("    Attribute %d: [Type:%s] %s\n", j,
           AttributeType_str(a.type()), a.name().c_str());
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
    for(int j = 0; j < ini.dims_size(); j++) {
      printf("%s%" PRId64, j ? ", " : "{", ini.dims(j));
    }
    printf("} ");
    assert(ini.segment().begin() == 0);
    assert(ini.segment().end() == 0);
    printf(" size:%zd\n", ini.raw_data().size());
  }

  for(int i = 0; i < gp.input_size(); i++) {
    printf("  Input %d: %s\n",
           i, ValueInfoProto_str(gp.input(i)).c_str());
  }

  for(int i = 0; i < gp.output_size(); i++) {
    printf("  Output %d: %s\n",
           i, ValueInfoProto_str(gp.output(i)).c_str());
  }

  for(int i = 0; i < gp.value_info_size(); i++) {
    printf("  ValueInfo %d: %s\n",
           i, ValueInfoProto_str(gp.value_info(i)).c_str());
  }
}


#if 0

class AttributeMap : public unordered_map<string, const onnx::AttributeProto *>
{
public:
  float getFloat(const string& name, float def) const {
    auto r = find(name);
    if(r == AttributeMap::end() ||
       r->second->type() != onnx::AttributeProto_AttributeType_FLOAT)
      return def;
    return r->second->f();
  }

  int getInt(const string& name, int def) const {
    auto r = find(name);
    if(r == AttributeMap::end() ||
       r->second->type() != onnx::AttributeProto_AttributeType_INT)
      return def;
    return r->second->i();
  }

  vector<int> getInts(const string& name) const {
    auto r = find(name);
    if(r == AttributeMap::end() ||
       r->second->type() != onnx::AttributeProto_AttributeType_INTS)
      return {};
    return vector<int>(r->second->ints().begin(), r->second->ints().end());
  }
};

#endif




struct MappedPBFile : public CodedInputStream {
public:
  MappedPBFile(void *data, size_t size)
    : CodedInputStream((const uint8_t *)data, size)
    , data_(data)
    , size_(size)
  {
    SetTotalBytesLimit(size, size);
  }

  ~MappedPBFile() {
    munmap(data_, size_);
  }

private:
  void *data_;
  size_t size_;
};



static unique_ptr<MappedPBFile>
mapPBfile(const char *path)
{
  const int fd = open(path, O_RDONLY);
  if(fd == -1) {
    fprintf(stderr, "Failed to open protobuf file %s: %s\n",
            path, strerror(errno));
    return nullptr;
  }

  struct stat st;
  fstat(fd, &st);

  void *mem = mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
  close(fd);

  return make_unique<MappedPBFile>(mem, st.st_size);
}










#if 0

typedef unordered_map<string, const onnx::TensorProto *> TensorMap;

static void
loadNCHW(const float *src, Tensor &t)
{
  for(size_t i = 0; i < t.n; i++) {
    for(size_t j = 0; j < t.c; j++) {
      for(size_t y = 0; y < t.h; y++) {
        for(size_t x = 0; x < t.w; x++) {
          t.set(i, j, y, x, *src++);
        }
      }
    }
  }
}

static shared_ptr<Tensor>
tensor_from_TensorProto(const onnx::TensorProto &tp, size_t rank)
{
  if(tp.dims_size() > 4) {
    fprintf(stderr, "Unable to load %s: Tensor of dimension %d not supported\n",
            tp.name().c_str(), tp.dims_size());
    return nullptr;
  }

  assert(tp.data_type() == onnx::TensorProto_DataType_FLOAT);


  auto dims = std::vector<unsigned int>(tp.dims().begin(), tp.dims().end());

  while(rank && dims.size() < rank)
    dims.insert(dims.begin(), 1);

  auto t = make_shared<Tensor>(Size(dims), Tensor::Type::FLOAT);

  t->allocate(CUDNN_TENSOR_NHWC);

  if(tp.raw_data().size()) {
    loadNCHW((const float *)&tp.raw_data()[0], *t);
  } else if(tp.float_data_size()) {
    std::vector<float> v(tp.float_data().begin(), tp.float_data().end());
    loadNCHW(&v[0], *t);
  } else {
    fprintf(stderr, "Unable to load %s: Can't load data format (please fix)\n",
            tp.name().c_str());
    return nullptr;
  }
  return t;
}




static std::shared_ptr<Tensor>
make_initializer(const TensorMap &initializers, const string &initializername,
                 Network &n, size_t rank)
{
  auto x = initializers.find(initializername);
  if(x == initializers.end()) {
    fprintf(stderr, "Unable to find %s\n", initializername.c_str());
    exit(1);
  }
  auto t = tensor_from_TensorProto(*x->second, rank);

  n.named_tensors_[initializername] = t;
  return t;
}


static shared_ptr<Layer>
onnx_add_batchnorm(Network &n,
                   const onnx::NodeProto &np,
                   const AttributeMap &attribs,
                   const TensorMap &initializers)
{
  assert(np.input_size() == 5);

  auto x = n.findLayer(np.input(0));
  if(!x) {
    fprintf(stderr, "Can't find input layer %s\n", np.input(0).c_str());
    return NULL;
  }

  make_initializer(initializers, np.input(1), n, 2);
  make_initializer(initializers, np.input(2), n, 2);
  make_initializer(initializers, np.input(3), n, 2);
  make_initializer(initializers, np.input(4), n, 2);

  float epsilon = attribs.getFloat("epsilon", 1e-05);

  return n.addLayer(makeBatchNorm(epsilon, *x.get(), n, 0.5,
                                  np.input(1).c_str(),
                                  np.input(2).c_str(),
                                  np.input(3).c_str(),
                                  np.input(4).c_str()));
}


static shared_ptr<Layer>
onnx_add_conv(Network &n,
              const onnx::NodeProto &np,
              const AttributeMap &attribs,
              const TensorMap &initializers)
{
  assert(np.input_size() == 2 || np.input_size() == 3);
  bool with_bias = np.input_size() == 3;
  auto x = n.findLayer(np.input(0));
  if(!x) {
    fprintf(stderr, "Can't find input layer %s\n", np.input(0).c_str());
    return NULL;
  }

  assert(attribs.find("auto_pad") == attribs.end()); // Can't deal with this
  assert(attribs.getInt("group", 1) == 1);
  for(auto d : attribs.getInts("dilations")) {
    assert(d == 1);
  }

  auto weights = make_initializer(initializers, np.input(1), n, 4);
  const int feature_maps = weights->n;
  const int kernel_size = weights->w;
  assert(weights->w == weights->h); // Only square kernels for now

  auto bias = with_bias ?
    make_initializer(initializers, np.input(2), n, 2) : NULL;

  int pad = 0;
  auto pads = attribs.getInts("pads");

  if(pads.size()) {
    pad = pads[0];
    for(auto p : pads) {
      assert(p == pad);
    }
  }


  int stride = 0;
  auto strides = attribs.getInts("strides");

  if(strides.size()) {
    stride = strides[0];
    for(auto s: strides) {
      assert(s == stride);
    }
  }

  return n.addLayer(makeConvolution(feature_maps, kernel_size, stride, pad,
                                    *x.get(), n, with_bias,
                                    np.input(1).c_str(),
                                    with_bias ? np.input(2).c_str() : NULL));
}



static shared_ptr<Layer>
onnx_add_gemm(Network &n,
              const onnx::NodeProto &np,
              const AttributeMap &attribs,
              const TensorMap &initializers)
{
  assert(np.input_size() == 3);
  auto x = n.findLayer(np.input(0));
  if(!x) {
    fprintf(stderr, "Can't find input layer %s\n", np.input(0).c_str());
    return NULL;
  }

  auto weights = make_initializer(initializers, np.input(1), n, 2);
  printf("initialized weights: %s\n", weights->name().c_str());

  auto bias    = make_initializer(initializers, np.input(2), n, 2);
  printf("   initialized bias: %s\n", bias->name().c_str());

  return n.addLayer(makeFullyConnected(bias->c, *x.get(),
                                       n,
                                       np.input(1).c_str(),
                                       np.input(2).c_str()));
}


static shared_ptr<Layer>
onnx_add_relu(Network &n,
              const onnx::NodeProto &np,
              const AttributeMap &attribs,
              const TensorMap &initializers)
{
  assert(np.input_size() == 1);
  auto x = n.findLayer(np.input(0));
  if(!x) {
    fprintf(stderr, "Can't find input layer %s\n", np.input(0).c_str());
    return NULL;
  }
  return n.addLayer(makeActivation(ActivationMode::RELU, 0, *x.get(), n));
}

static shared_ptr<Layer>
onnx_add_softmax(Network &n,
                 const onnx::NodeProto &np,
                 const AttributeMap &attribs)
{
  assert(np.input_size() == 1);
  auto x = n.findLayer(np.input(0));
  if(!x) {
    fprintf(stderr, "Can't find input layer %s\n", np.input(0).c_str());
    return NULL;
  }
  return n.addLayer(makeSoftmax(*x.get(), n));
}


static shared_ptr<Layer>
onnx_add_pool(Network &n,
              const onnx::NodeProto &np,
              const AttributeMap &attribs,
              const TensorMap &initializers,
              PoolingMode mode)
{
  auto x = n.findLayer(np.input(0));
  if(!x) {
    fprintf(stderr, "Can't find input layer %s\n", np.input(0).c_str());
    return NULL;
  }

  assert(attribs.find("auto_pad") == attribs.end()); // Can't deal with this
  assert(attribs.getInt("ceil_mode", 0) == 0);
  for(auto d : attribs.getInts("dilations")) {
    assert(d == 1);
  }

  int size = 0;
  auto kernel_shape = attribs.getInts("kernel_shape");
  if(kernel_shape.size()) {
    size = kernel_shape[0];
    for(auto x : kernel_shape) {
      assert(x == size);
    }
  }

  int pad = 0;
  auto pads = attribs.getInts("pads");
  if(pads.size()) {
    pad = pads[0];
    for(auto d : pads) {
      assert(d == pad);
    }
  }

  assert(attribs.getInt("storage_order", 0) == 0);

  int stride = 0;
  auto strides = attribs.getInts("strides");

  if(strides.size()) {
    stride = strides[0];
    for(auto s: strides) {
      assert(s == stride);
    }
  }

  return n.addLayer(makePooling(mode, size, pad, stride, *x.get(), n));
}


static shared_ptr<Layer>
onnx_add_concat(Network &n,
                const onnx::NodeProto &np,
                const AttributeMap &attribs,
                const TensorMap &initializers)
{
  assert(attribs.getInt("axis", 0) == 1); // c-axis

  vector<const Layer *>inputs;
  for(const auto &i : np.input()) {
    auto l = n.findLayer(i);
    if(!l) {
      fprintf(stderr, "Can't find input layer %s\n", i.c_str());
      return NULL;
    }
    inputs.push_back(l.get());
  }
  return n.addLayer(makeConcat(inputs, n));
}


static shared_ptr<Layer>
onnx_add_sum(Network &n,
             const onnx::NodeProto &np,
             const AttributeMap &attribs,
             const TensorMap &initializers)
{
  vector<const Layer *>inputs;
  for(const auto &i : np.input()) {
    auto l = n.findLayer(i);
    if(!l) {
      fprintf(stderr, "Can't find input layer %s\n", i.c_str());
      return NULL;
    }
    inputs.push_back(l.get());
  }
  return n.addLayer(makeSum(inputs, n));
}


static shared_ptr<Layer>
onnx_add_dropout(Network &n,
                 const onnx::NodeProto &np,
                 const AttributeMap &attribs,
                 const TensorMap &initializers)
{
  float ratio = attribs.getFloat("ratio", 0.5);

  assert(np.input_size() == 1);
  auto x = n.findLayer(np.input(0));
  if(!x) {
    fprintf(stderr, "Can't find input layer %s\n", np.input(0).c_str());
    return NULL;
  }

  if(n.backprop_)
    return n.addLayer(makeDropout(ratio, x, n));
  else
    return x;
}


static shared_ptr<Layer>
onnx_add_reshape(Network &n,
                 const onnx::NodeProto &np,
                 const AttributeMap &attribs)
{
  auto x = n.findLayer(np.input(0));
  if(!x) {
    fprintf(stderr, "Can't find input layer %s\n", np.input(0).c_str());
    return NULL;
  }
  return x;
}


static bool
loadgraph(Network &n, const onnx::GraphProto &gp)
{
  TensorMap initializers;

  for(int i = 0; i < gp.initializer_size(); i++) {
    const auto &ini = gp.initializer(i);
    initializers[ini.name()] = &ini;
  }

  for(int i = 0; i < gp.node_size(); i++) {
    const auto &np = gp.node(i);

    NodeProto_print(np);
    assert(np.output_size() == 1);

    AttributeMap attribs;
    for(int j = 0; j < np.attribute_size(); j++) {
      const auto &a = np.attribute(j);
      attribs[a.name()] = &a;
    }
    shared_ptr<Layer> l;
    const auto& node_type = np.op_type();
    if(node_type == "Conv") {
      l = onnx_add_conv(n, np, attribs, initializers);
    } else if(node_type == "BatchNormalization") {
      l = onnx_add_batchnorm(n, np, attribs, initializers);
    } else if(node_type == "Relu") {
      l = onnx_add_relu(n, np, attribs, initializers);
    } else if(node_type == "Softmax") {
      l = onnx_add_softmax(n, np, attribs);
    } else if(node_type == "MaxPool") {
      l = onnx_add_pool(n, np, attribs, initializers,
                           PoolingMode::MAX);
    } else if(node_type == "AveragePool") {
      l = onnx_add_pool(n, np, attribs, initializers,
                           PoolingMode::AVERAGE);
    } else if(node_type == "Concat") {
      l = onnx_add_concat(n, np, attribs, initializers);
    } else if(node_type == "Sum") {
      l = onnx_add_sum(n, np, attribs, initializers);
    } else if(node_type == "Dropout") {
      l = onnx_add_dropout(n, np, attribs, initializers);
    } else if(node_type == "Reshape") {
      l = onnx_add_reshape(n, np, attribs);
    } else if(node_type == "Gemm") {
      l = onnx_add_gemm(n, np, attribs, initializers);
    } else {
      fprintf(stderr, "Can't handle node type %s\n", node_type.c_str());
      return false;
    }
    if(l == NULL)
      return false;

    n.nameLayer(l, np.output(0));
  }

  return true;
}








shared_ptr<Tensor>
Tensor::createFromPB(const char *path)
{
  auto pb = mapPBfile(path);
  if(pb == NULL)
    return nullptr;

  onnx::TensorProto tp;
  if(!tp.ParseFromCodedStream(pb.get()))
    return nullptr;

  return tensor_from_TensorProto(tp, 0);
}


bool
Network::load(const char *path)
{
  auto pb = mapPBfile(path);
  if(pb == NULL)
    return false;

  onnx::ModelProto mp;
  if(!mp.ParseFromCodedStream(pb.get()))
    return false;

  cout << "IR version: " << mp.ir_version() << endl;
  const auto &gp = mp.graph();

  if(0)
    print_onnx_graph_info(gp);

  return loadgraph(*this, gp);
}
#endif

//------------------------------------------------------------------------
//


static Tensor::DataType
DataType_map(int dt)
{
  switch((onnx::TensorProto_DataType)dt) {
  case onnx::TensorProto_DataType_FLOAT: return Tensor::DataType::FLOAT;
  case onnx::TensorProto_DataType_UINT8: return Tensor::DataType::U8;
  case onnx::TensorProto_DataType_INT64: return Tensor::DataType::INT64;
  case onnx::TensorProto_DataType_FLOAT16: return Tensor::DataType::HALF;
  default:
    fprintf(stderr, "ONNX: No mapping for data-type %s\n",  DataType_str(dt));
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
  for(const auto dim : shape.dim()) {
    int d;
    switch(dim.value_case()) {
    case onnx::TensorShapeProto_Dimension::kDimValue:
      d = dim.dim_value();
      break;
    default:
      return nullptr;
    }
    dims.push_back(d);
  }
  return make_shared<Tensor>(vip.name(), DataType_map(tp.tensor_type().elem_type()), dims);
}


static shared_ptr<Tensor>
make_tensor(const onnx::TensorProto &tp)
{
  Dims dims;
  for(const auto &dim : tp.dims()) {
    dims.push_back(dim);
  }

  auto t = makeCPUTensor(tp.name(), DataType_map(tp.data_type()), dims);

  auto ta = t->access();

  if(tp.raw_data().size()) {
    printf("Loading %s from %zd bytes\n", t->info().c_str(), tp.raw_data().size());
    memcpy(ta->data(), (const void *)&tp.raw_data()[0], tp.raw_data().size());
  } else {
    fprintf(stderr, "Unable to load %s: Can't load data format (please fix)\n",
            tp.name().c_str());
  }
  return t;
}



static shared_ptr<Tensor>
find_tensor(Graph &g, const std::string &name)
{
  auto it = g.tensors_.find(name);
  if(it != g.tensors_.end())
    return it->second;
  fprintf(stderr, "Unable to find tensor %s\n", name.c_str());
  return nullptr;
}


static shared_ptr<Node>
make_conv(Graph &g, const onnx::NodeProto &np, const Attributes &attribs)
{
  assert(np.input_size() > 1);
  assert(np.output_size() == 1);

  auto n = std::make_shared<Node>(Node::Type::CONV);
  n->inputs_["x"] = find_tensor(g, np.input(0));

  n->inputs_["w"] = find_tensor(g, np.input(1));
  if(np.input_size() > 2)
    n->inputs_["b"] = find_tensor(g, np.input(2));


  assert(attribs.find("auto_pad") == attribs.end()); // Can't deal with this

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
    for(auto s: strides) {
      assert(s == stride);
    }
  }
  n->attributes_["stride"] = stride;

  g.tensors_[np.output(0)] = n->makeOutputTensor(np.output(0));
  return n;
}



static shared_ptr<Node>
make_batchnorm(Graph &g, const onnx::NodeProto &np, const Attributes &attribs)
{
  assert(np.input_size() == 5);
  assert(np.output_size() == 1);

  auto n = std::make_shared<Node>(Node::Type::BATCHNORM);
  n->inputs_["x"] = find_tensor(g, np.input(0));

  n->inputs_["s"] = find_tensor(g, np.input(1));
  n->inputs_["b"] = find_tensor(g, np.input(2));
  n->inputs_["rm"] = find_tensor(g, np.input(3));
  n->inputs_["riv"] = find_tensor(g, np.input(4));

  float epsilon = attribs.get("epsilon", 1e-05f);
  n->attributes_["epsilon"] = epsilon;

  g.tensors_[np.output(0)] = n->makeOutputTensor(np.output(0));
  return n;
}


static shared_ptr<Node>
make_relu(Graph &g, const onnx::NodeProto &np, const Attributes &attribs)
{
  assert(np.input_size() == 1);
  assert(np.output_size() == 1);
  auto n = std::make_shared<Node>(Node::Type::RELU);
  n->inputs_["x"] = find_tensor(g, np.input(0));
  g.tensors_[np.output(0)] = n->makeOutputTensor(np.output(0));
  return n;
}

static shared_ptr<Node>
make_pooling(Graph &g, const onnx::NodeProto &np, const Attributes &attribs,
             Node::Type type)
{
  assert(np.input_size() == 1);
  assert(np.output_size() == 1);
  auto n = std::make_shared<Node>(type);
  n->inputs_["x"] = find_tensor(g, np.input(0));

  assert(attribs.find("auto_pad") == attribs.end()); // Can't deal with this
  assert(attribs.get("ceil_mode", 0) == 0);
  for(auto d : attribs.get("dilations", std::vector<int>({}))) {
    assert(d == 1);
  }

  int size = 0;
  auto kernel_shape = attribs.get("kernel_shape", std::vector<int>({}));
  if(kernel_shape.size()) {
    size = kernel_shape[0];
    for(auto x : kernel_shape) {
      assert(x == size);
    }
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

  int stride = 0;
  auto strides = attribs.get("strides", std::vector<int>({}));

  if(strides.size()) {
    stride = strides[0];
    for(auto s: strides) {
      assert(s == stride);
    }
  }

  n->attributes_["size"] = size;
  n->attributes_["pad"] = pad;
  n->attributes_["stride"] = stride;
  g.tensors_[np.output(0)] = n->makeOutputTensor(np.output(0));

  return n;
}

static shared_ptr<Node>
make_sum(Graph &g, const onnx::NodeProto &np, const Attributes &attribs)
{
  assert(np.input_size() >= 1);
  assert(np.output_size() == 1);
  auto n = std::make_shared<Node>(Node::Type::SUM);

  for(int i = 0; i < np.input_size(); i++) {
    char name[20];
    snprintf(name, sizeof(name), "x%u", i);
    n->inputs_[name] = find_tensor(g, np.input(i));
  }
  g.tensors_[np.output(0)] = n->makeOutputTensor(np.output(0));
  return n;
}


static shared_ptr<Node>
make_reshape(Graph &g, const onnx::NodeProto &np, const Attributes &attribs)
{
  assert(np.input_size() == 2);
  assert(np.output_size() == 1);
  auto n = std::make_shared<Node>(Node::Type::RESHAPE);
  n->inputs_["x"] = find_tensor(g, np.input(0));
  n->inputs_["shape"] = find_tensor(g, np.input(1));
  g.tensors_[np.output(0)] = n->makeOutputTensor(np.output(0));
  return n;
}

static shared_ptr<Node>
make_gemm(Graph &g, const onnx::NodeProto &np, const Attributes &attribs)
{
  assert(np.input_size() == 3);
  assert(np.output_size() == 1);
  auto n = std::make_shared<Node>(Node::Type::FC);
  n->inputs_["x"] = find_tensor(g, np.input(0));
  n->inputs_["w"] = find_tensor(g, np.input(1));
  n->inputs_["b"] = find_tensor(g, np.input(2));
  g.tensors_[np.output(0)] = n->makeOutputTensor(np.output(0));
  return n;
}


static shared_ptr<Node>
make_softmax(Graph &g, const onnx::NodeProto &np, const Attributes &attribs)
{
  assert(np.input_size() == 1);
  assert(np.output_size() == 1);
  auto n = std::make_shared<Node>(Node::Type::SOFTMAX);
  n->inputs_["x"] = find_tensor(g, np.input(0));
  g.tensors_[np.output(0)] = n->makeOutputTensor(np.output(0));
  return n;
}


static bool
loadgraph(Graph &g, const onnx::GraphProto &gp)
{
  for(const auto &vip : gp.input()) {
    g.tensors_[vip.name()] = make_tensor(vip);
  }

  for(const auto &vip : gp.output()) {
    g.tensors_[vip.name()] = make_tensor(vip);
  }

  for(const auto &tp : gp.initializer()) {
    g.tensors_[tp.name()] = make_tensor(tp);
  }

  for(const auto &np : gp.node()) {

    NodeProto_print(np);
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
        attribs[a.name()] = vector<int>(a.ints().begin(), a.ints().end());
        break;
      default:
        break;
      }
    }

    shared_ptr<Node> n;
    const auto& node_type = np.op_type();
    if(node_type == "Conv") {
      n = make_conv(g, np, attribs);
    } else if(node_type == "BatchNormalization") {
      n = make_batchnorm(g, np, attribs);
    } else if(node_type == "Relu") {
      n = make_relu(g, np, attribs);
    } else if(node_type == "MaxPool") {
      n = make_pooling(g, np, attribs, Node::Type::MAXPOOL);
    } else if(node_type == "AveragePool") {
      n = make_pooling(g, np, attribs, Node::Type::AVGPOOL);
    } else if(node_type == "Sum") {
      n = make_sum(g, np, attribs);
    } else if(node_type == "Reshape") {
      n = make_reshape(g, np, attribs);
    } else if(node_type == "Gemm") {
      n = make_gemm(g, np, attribs);
    } else if(node_type == "Softmax") {
      n = make_softmax(g, np, attribs);
    } else {
      fprintf(stderr, "Can't handle node type %s\n", node_type.c_str());
      return false;
    }
    g.nodes_.push_back(n);


    for(const auto &i : n->inputs_) {
      printf("%15s: %s\n", i.first.c_str(), i.second->info().c_str());
    }
    for(const auto &i : n->outputs_) {
      printf("%15s: %s\n", i.first.c_str(), i.second->info().c_str());
    }
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

  cout << "IR version: " << mp.ir_version() << endl;
  const auto &gp = mp.graph();

  if(1)
    print_onnx_graph_info(gp);


  auto g = make_shared<Graph>();

  if(!loadgraph(*g.get(), gp))
    return nullptr;

  return g;
}




std::shared_ptr<Tensor>
Graph::loadTensor(const char *path)
{
  auto pb = mapPBfile(path);
  if(pb == NULL)
    return nullptr;

  onnx::TensorProto tp;
  if(!tp.ParseFromCodedStream(pb.get()))
    return nullptr;

  auto t = make_tensor(tp);

  printf("Loaded named tensor %s from %s\n", tp.name().c_str(), path);
  tensors_[tp.name()] = t;
  return t;

}



}
