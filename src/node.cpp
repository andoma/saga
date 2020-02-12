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

#include <math.h>
#include "saga.h"

namespace saga {


static std::optional<const std::string>
node_tensor_name(const std::optional<const std::string> &node_name,
                 const std::string &tensor_name)
{
  if(!node_name)
    return std::nullopt;

  return *node_name + "-" + tensor_name;
}




//------------------------------------------------------------------------

static std::shared_ptr<Tensor>
conv_y(const Node &n, const std::optional<const std::string> &name)
{
  // Should make this more generic for n-dimensions
  const int stride = n.attributes_.get("stride", 1);
  const int pad = n.attributes_.get("pad", 0);
  const int dilation = n.attributes_.get("dilation", 1);

  auto w = n.inputs_.get("w");
  if(w == nullptr)
    return nullptr;
  const int features = w->dims_[0];
  const int filterdim_h = w->dims_[2];
  const int filterdim_w = w->dims_[3];

  auto x = n.inputs_.get("x");
  if(x == nullptr)
    return nullptr;

  const int inputdim_h = x->dims_[2];
  const int inputdim_w = x->dims_[3];

  const int outputdim_w =
    1 + (inputdim_w + 2 * pad - (((filterdim_w - 1) * dilation) + 1))/stride;

  const int outputdim_h =
    1 + (inputdim_h + 2 * pad - (((filterdim_h - 1) * dilation) + 1))/stride;

  return std::make_shared<Tensor>(x->data_type_,
                                  Dims({1, features,
                                        outputdim_h, outputdim_w}), name);
}

static std::vector<std::shared_ptr<Node>>
conv_setup(std::shared_ptr<Node> n, Tensors &named_tensors)
{
  auto x = n->inputs_.get("x");
  if(!x)
    return {};

  auto w = n->inputs_.get("w");
  auto b = n->inputs_.get("b");

  if(!w) {
    const int activations = n->attributes_.get("activations", 1);
    const int size = n->attributes_.get("size", 1);

    n->inputs_["w"] = w =
      Tensor::find(x->data_type_, {activations, x->dims_[1], size, size},
                   0, sqrt(2.0 / (x->dims_[1] * size * size)),
                   named_tensors, node_tensor_name(n->name_, "w"));
  }

  if(!b && n->attributes_.get("bias", false)) {
    n->inputs_["b"] =
      Tensor::find(x->data_type_, {1, w->dims_[0]},
                   0, 0,
                   named_tensors, node_tensor_name(n->name_, "b"));
  }

  return {n};
}

//------------------------------------------------------------------------

static std::vector<std::shared_ptr<Node>>
batchnorm_setup(std::shared_ptr<Node> n, Tensors &named_tensors)
{
  auto x = n->inputs_.get("x");
  if(!x)
    return {};

  Dims dims{1, x->dims_[1]};

  if(!n->inputs_.get("s")) {
    n->inputs_["s"] =
      Tensor::find(Tensor::DataType::FLOAT, dims,
                   1.0, 0, named_tensors, node_tensor_name(n->name_, "s"));
  }

  if(!n->inputs_.get("b")) {
    n->inputs_["b"] =
      Tensor::find(Tensor::DataType::FLOAT, dims,
                   0.0, 0, named_tensors, node_tensor_name(n->name_, "b"));
  }

  if(!n->inputs_.get("m")) {
    n->inputs_["m"] =
      Tensor::find(Tensor::DataType::FLOAT, dims,
                   0.0, 0, named_tensors, node_tensor_name(n->name_, "m"));
  }

  if(!n->inputs_.get("v")) {
    n->inputs_["v"] =
      Tensor::find(Tensor::DataType::FLOAT, dims,
                   1.0, 0, named_tensors, node_tensor_name(n->name_, "v"));
  }
  return {n};
}


//------------------------------------------------------------------------

static std::shared_ptr<Tensor>
pooling_y(const Node &n, const std::optional<const std::string> &name)
{
  // Should make this more generic for n-dimensions

  const int pad = n.attributes_.get("pad", 0);
  const int stride = n.attributes_.get("stride", 1);
  auto x = n.inputs_.get("x");
  if(x == nullptr)
    return nullptr;

  int size;

  if(n.attributes_.get("global", false)) {
    size = x->dims_[2];
    assert(x->dims_[3] == size);
  } else {
    size = n.attributes_.get("size", 1);
  }

  const int channels   = x->dims_[1];
  const int inputdim_h = x->dims_[2];
  const int inputdim_w = x->dims_[3];

  const int outputdim_h =
    1 + (inputdim_h + 2 * pad - size) / stride;
  const int outputdim_w =
    1 + (inputdim_w + 2 * pad - size) / stride;

  return std::make_shared<Tensor>(x->data_type_,
                                  Dims({1, channels,
                                        outputdim_h, outputdim_w}), name);
}

//------------------------------------------------------------------------

static std::shared_ptr<Tensor>
reshape_y(const Node &n, const std::optional<const std::string> &name)
{
  auto x = n.inputs_.get("x");
  if(x == nullptr)
    return nullptr;
  auto shape = n.inputs_.get("shape");
  if(shape == nullptr)
    return nullptr;

  if(shape->dims_.size() != 1) {
    fprintf(stderr, "Shape tensor is not 1d\n");
    return nullptr;
  }

  auto ta = shape->access();

  Dims dims;
  for(int i = 0; i < shape->dims_[0]; i++) {
    const int64_t v = ta->get({i});
    if(v == 0) {
      dims.push_back(x->dims_[i]);
    } else if(v == -1) {
      int64_t s = 1;
      for(; i < (int64_t)x->dims_.size(); i++) {
        s *= x->dims_[i];
      }
      dims.push_back(s);
      break;
    } else {
      assert(v > 0);
      dims.push_back(v);
    }
  }

  return std::make_shared<Tensor>(x->data_type_, dims, name);
}


//------------------------------------------------------------------------

static std::shared_ptr<Tensor>
concat_y(const Node &n, const std::optional<const std::string> &name)
{
  int i = 0;
  int axis = 1;
  Dims dims;
  Tensor::DataType data_type = Tensor::DataType::U8;
  while(1) {
    auto x = n.inputs_.get("x" + std::to_string(i));
    if(x == nullptr)
      break;
    if(i == 0) {
      dims = x->dims_;
      data_type = x->data_type_;
    } else {
      dims[axis] += x->dims_[axis];
    }
    i++;
  }
  return std::make_shared<Tensor>(data_type, dims, name);
}

//------------------------------------------------------------------------

static std::shared_ptr<Tensor>
fc_y(const Node &n, const std::optional<const std::string> &name)
{
  auto w = n.inputs_.get("w");
  if(w == nullptr)
   return nullptr;
  const int transW = n.attributes_.get("transW", 0);

  return std::make_shared<Tensor>(w->data_type_,
                                  Dims({1, w->dims_[transW ? 0 : 1]}),
                                  name);
}


static std::vector<std::shared_ptr<Node>>
fc_setup(std::shared_ptr<Node> n,
         Tensors &named_tensors)
{
  std::vector<std::shared_ptr<Node>> nodes;

  auto x = n->inputs_.get("x");
  if(!x)
    return nodes;

  if(x->dims_.size() != 2) {
    // Auto-insert a reshape node
    auto shape = makeCPUTensor(Tensor::DataType::INT64, Dims({2}));
    auto a = shape->access();
    a->set({0}, 1);
    a->set({1}, x->elements_);

    auto r = std::make_shared<Node>("reshape");
    r->inputs_["x"] = x;
    r->inputs_["shape"] = shape;
    x = r->inferTensor_y();
    r->outputs_["y"] = x;
    n->inputs_["x"] = x;
    nodes.push_back(r);
  }

  auto w = n->inputs_.get("w");
  auto b = n->inputs_.get("b");

  if(!w) {
    const int outputs = n->attributes_.get("outputs", 1);
    n->attributes_["transB"] = 1;


    n->inputs_["w"] = w =
      Tensor::find(x->data_type_, {x->dims_[1], outputs},
                   0, sqrt(2.0 / x->dims_[1]),
                   named_tensors, node_tensor_name(n->name_, "w"));
  }

  if(!b && n->attributes_.get("bias", false)) {
    n->inputs_["b"] =
      Tensor::find(x->data_type_, {1, w->dims_[1]},
                   0, 0,
                   named_tensors, node_tensor_name(n->name_, "b"));
  }
  nodes.push_back(n);
  return nodes;
}

//------------------------------------------------------------------------

static std::shared_ptr<Tensor>
catclassifier_y(const Node &n, const std::optional<const std::string> &name)
{
  return std::make_shared<Tensor>(Tensor::DataType::I32, Dims({1, 1}), name);
}

static std::vector<std::shared_ptr<Node>>
catclassifier_setup(std::shared_ptr<Node> n, Tensors &named_tensors)
{
  n->outputs_["loss"] =
    std::make_shared<Tensor>(Tensor::DataType::FLOAT, Dims({1, 1}), "loss");
  return {n};
}
//------------------------------------------------------------------------

static std::shared_ptr<Tensor>
passthru_y(const Node &n, const std::optional<const std::string> &name)
{
  auto o = n.inputs_.get("x");
  if(o == nullptr)
    return nullptr;
  return std::make_shared<Tensor>(o->data_type_, o->dims_, name);
}

//------------------------------------------------------------------------

static std::shared_ptr<Tensor>
sum_y(const Node &n, const std::optional<const std::string> &name)
{
  auto o = n.inputs_.get("x0");
  if(o == nullptr)
    return nullptr;
  return std::make_shared<Tensor>(o->data_type_, o->dims_, name);
}

//------------------------------------------------------------------------

static std::shared_ptr<Tensor>
convert_y(const Node &n, const std::optional<const std::string> &name)
{
  auto x = n.inputs_.get("x");
  if(x == nullptr)
    return nullptr;

  auto datatype = n.attributes_.get("datatype", -1);
  if(datatype == -1)
    return nullptr;

  return std::make_shared<Tensor>((Tensor::DataType)datatype, x->dims_, name);
}


//------------------------------------------------------------------------
//------------------------------------------------------------------------

static const struct {
  const char *name;

  std::shared_ptr<Tensor>(*infer_y)(const Node &n,
                                    const std::optional<const std::string> &name);

  std::vector<std::shared_ptr<Node>>(*setup)(std::shared_ptr<Node> node,
                                             Tensors &named_tensors);

} nodetypes[] = {
  { "add",               passthru_y },
  { "avgpool",           pooling_y },
  { "batchnorm",         passthru_y, batchnorm_setup },
  { "catclassifier",     catclassifier_y, catclassifier_setup },
  { "concat",            concat_y },
  { "conv",              conv_y, conv_setup },
  { "dropout",           passthru_y },
  { "fc",                fc_y, fc_setup },
  { "maxpool",           pooling_y },
  { "mul",               passthru_y },
  { "relu",              passthru_y },
  { "reshape",           reshape_y },
  { "softmax",           passthru_y },
  { "spatialtransform",  passthru_y },
  { "sum",               sum_y },
  { "convert",           convert_y },
};


std::shared_ptr<Tensor>
Node::inferTensor_y(const std::optional<const std::string> &name)
{
  for(size_t i = 0; i < sizeof(nodetypes) / sizeof(nodetypes[0]); i++) {
    if(type_ == nodetypes[i].name) {
      return nodetypes[i].infer_y(*this, name);
    }
  }

  fprintf(stderr, "Failed to compute output tensor for type %s\n",
          type_.c_str());
  print();
  abort();
}


void
Node::print() const
{
  printf("%s:\n", type_.c_str());

  for(const auto &t : inputs_) {
    printf("\t Input: %s: %s\n",
           t.first.c_str(), t.second->info().c_str());
  }

  for(const auto &t : outputs_) {
    printf("\tOutput: %s: %s\n",
           t.first.c_str(), t.second->info().c_str());
  }

  for(const auto &a : attributes_) {
    std::string value;

    if(auto v = std::get_if<int>(&a.second)) {
      value = std::to_string(*v);
    } else if(auto v = std::get_if<float>(&a.second)) {
      value = std::to_string(*v);
    } else if(auto v = std::get_if<bool>(&a.second)) {
      value = *v ? "true" : "false";
    } else if(std::get_if<std::vector<int>>(&a.second)) {
      value = "<vector>";
    } else {
      value = "?";
    }

    printf("\tAttrib: %s: %s\n",
           a.first.c_str(), value.c_str());
  }

}


std::vector<std::shared_ptr<Node>>
Node::make(const std::string &type,
           const Tensors &inputs,
           const Attributes &attributes,
           Tensors &named_tensors,
           const std::optional<const std::string> &name)
{
  auto n = std::make_shared<Node>(type, name);
  n->inputs_ = inputs;
  n->attributes_ = attributes;

  std::vector<std::shared_ptr<Node>> nodes({n});

  for(size_t i = 0; i < sizeof(nodetypes) / sizeof(nodetypes[0]); i++) {
    if(type == nodetypes[i].name && nodetypes[i].setup) {
      nodes = nodetypes[i].setup(n, named_tensors);
    }
  }

  if(!nodes.empty()) {
    auto &last = nodes.back();
    last->outputs_["y"] = last->inferTensor_y();
  }
  return nodes;
}

std::shared_ptr<Tensor>
Node::y()
{
  auto it = outputs_.find("y");
  if(it == outputs_.end())
    return nullptr;
  return it->second;
}
}
