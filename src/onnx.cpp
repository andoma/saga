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


typedef unordered_map<string, const onnx::TensorProto *> TensorMap;

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



static vector<int>
make_initializer(InitData &id, const string &idname,
                 const TensorMap &initializers, const string &initializername)
{
  auto x = initializers.find(initializername);
  if(x == initializers.end()) {
    fprintf(stderr, "Unable to find %s", initializername.c_str());
    exit(1);
  }

  const auto &ini = *x->second;
  assert(ini.data_type() == onnx::TensorProto_DataType_FLOAT);

  shared_ptr<TensorValues> tv;

  if(ini.float_data_size()) {
    tv = make_shared<TensorValues>(vector<float>(ini.float_data().begin(),
                                                 ini.float_data().end()));
  } else if(ini.raw_data().size()) {
    tv = make_shared<TensorValues>((const void *)&ini.raw_data()[0],
                                   ini.raw_data().size());
  } else {
    fprintf(stderr, "%s have no data we can parse\n",
            initializername.c_str());
    exit(1);
  }
  id[idname] = tv;
  return vector<int>(ini.dims().begin(), ini.dims().end());
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

  InitData id;
  auto ws = make_initializer(id, "weights", initializers, np.input(1));
  const int feature_maps = ws[0];
  assert(ws[2] == ws[3]); // Only square filters are supported
  const int kernel_size = ws[2];

  if(with_bias)
    make_initializer(id, "bias", initializers, np.input(2));

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
                                    *x.get(), id, n, with_bias));
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

  for(auto d : attribs.getInts("pads")) {
    assert(d == 0);
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

  return n.addLayer(makePooling(mode, size, stride, *x.get(), n));
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
#if 0
  Tensor input(TensorDescriptor(CUDNN_DATA_FLOAT,
                                CUDNN_TENSOR_NCHW,
                                Size(n.batch_size_, 3, 224, 224)));

  auto inputLayer = n.addLayer(makeInput(&input));
  layers["data"] = inputLayer;
#endif

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
    } else if(node_type == "Relu") {
      l = onnx_add_relu(n, np, attribs, initializers);
    } else if(node_type == "MaxPool") {
      l = onnx_add_pool(n, np, attribs, initializers,
                           PoolingMode::MAX);
    } else if(node_type == "AveragePool") {
      l = onnx_add_pool(n, np, attribs, initializers,
                           PoolingMode::AVERAGE);
    } else if(node_type == "Concat") {
      l = onnx_add_concat(n, np, attribs, initializers);
    } else if(node_type == "Dropout") {
      l = onnx_add_dropout(n, np, attribs, initializers);
    } else if(node_type == "Reshape") {
      l = onnx_add_reshape(n, np, attribs);
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







shared_ptr<Tensor>
Tensor::loadFromPB(const char *path, bool hackit)
{
  auto pb = mapPBfile(path);
  if(pb == NULL)
    return nullptr;

  onnx::TensorProto tp;
  if(!tp.ParseFromCodedStream(pb.get()))
    return nullptr;

  if(tp.dims_size() > 4) {
    fprintf(stderr, "Unable to load %s: Tensor of dimension %d not supported\n",
            path, tp.dims_size());
    return nullptr;
  }

  Size s = std::vector<int64_t>(tp.dims().begin(), tp.dims().end());
  assert(tp.data_type() == onnx::TensorProto_DataType_FLOAT);

  auto t = make_shared<Tensor>(TensorDescriptor(CUDNN_DATA_FLOAT,
                                                CUDNN_TENSOR_NCHW,
                                                s));

  if(tp.raw_data().size()) {

    if(hackit) {
      float *copy = (float *)malloc(tp.raw_data().size());
      size_t values = tp.raw_data().size() / sizeof(float);
      memcpy((void *)copy,
             (const void *)&tp.raw_data()[0], tp.raw_data().size());

      size_t c = values / 3;
      assert(c * 3 == values);
      size_t o = 0;
      for(size_t i = 0; i < c; i++) {
        copy[i + o] = ((copy[i + o] / 255.0f) - 0.485f) / 0.229f;
      }
      o += c;
      for(size_t i = 0; i < c; i++) {
        copy[i + o] = ((copy[i + o] / 255.0f) - 0.456f) / 0.224f;
      }
      o += c;
      for(size_t i = 0; i < c; i++) {
        copy[i + o] = ((copy[i + o] / 255.0f) - 0.406f) / 0.225f;
      }

      t->load((const void *)copy, tp.raw_data().size());

    } else {
      t->load((const void *)&tp.raw_data()[0], tp.raw_data().size());
    }

  } else {
    fprintf(stderr, "Unable to load %s: Protobuf contains data we cant parse\n",
            path);
    return nullptr;

  }
  fprintf(stderr, "Loaded tensor %s from %s\n", s.name().c_str(), path);
  return t;
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


}
