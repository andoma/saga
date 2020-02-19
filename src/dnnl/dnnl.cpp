/*
 * Copyright (c) 2020, Andreas Smas
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

#include <unistd.h>
#include <string.h>

#include "context.h"

#include "dnnl_common.h"
#include "dnnl_tensor.h"

namespace saga {


DnnlContext::DnnlContext()
{
  chkDNNL(dnnl_engine_create(&engine_, dnnl_cpu, 0));
  chkDNNL(dnnl_stream_create(&stream_, engine_, dnnl_stream_default_flags));
}


DnnlContext::~DnnlContext()
{
  chkDNNL(dnnl_stream_wait(stream_));
  dnnl_stream_destroy(stream_);
  dnnl_engine_destroy(engine_);

}


static
std::shared_ptr<Context> createDnnlContext()
{
  return std::make_shared<DnnlContext>();
}



static void registerDnnlContext(void) __attribute__((constructor(110)));
static void registerDnnlContext(void)
{
  registerContextFactory(&createDnnlContext);
}






//------------------------------------------------------------------------

class DnnlOperation;


class DnnlProgram : public Program {
public:

  DnnlProgram(std::shared_ptr<DnnlContext> ctx,
              TensorLayout tensor_layout,
              int batch_size,
              float learning_rate)
    : ctx_(ctx)
    , tensor_layout_(tensor_layout)
    , batch_size_(batch_size)
    , learning_rate_(learning_rate)
    , debug_(false)
    , workspace_(NULL)
    , workspace_size_(0)
    , workspace_requested_(0)
  {
  }

  ~DnnlProgram()
  {
  }

  std::shared_ptr<Tensor> resolveTensor(std::shared_ptr<Tensor> t) override;
  void infer() override;
  void train() override;
  void print() const override;
  void debug(bool) override;

  const std::shared_ptr<DnnlContext> ctx_;
  const TensorLayout tensor_layout_;
  const int batch_size_;
  const float learning_rate_;
  bool debug_;

  void *workspace_;
  size_t workspace_size_;
  size_t workspace_requested_;

  std::vector<std::shared_ptr<DnnlOperation>> infer_operations_;


  std::unordered_map<std::shared_ptr<Tensor>,
                     std::shared_ptr<DnnlTensor>> tensors_;

  dnnl_memory_desc_t dnnl_desc_from_tensor_any(std::shared_ptr<Tensor> t);

  dnnl_memory_desc_t dnnl_desc_from_tensor(std::shared_ptr<Tensor> t);

  std::shared_ptr<DnnlTensor> lower_tensor(std::shared_ptr<Tensor> src);

  std::shared_ptr<DnnlTensor> lower_tensor(std::shared_ptr<Tensor> src,
                                           const dnnl_memory_desc_t *desc);

  std::shared_ptr<DnnlTensor> lower_tensor(std::shared_ptr<Tensor> src,
                                           dnnl_primitive_desc_t desc,
                                           dnnl_query_t what);

  std::shared_ptr<DnnlTensor> lower_tensor_batch(std::shared_ptr<Tensor> src);


  void infer(const std::shared_ptr<DnnlOperation> &op)
  {
    infer_operations_.push_back(op);

    chkDNNL(dnnl_stream_wait(ctx_->stream_));


  }

};

//------------------------------------------------------------------------

static dnnl_data_type_t
dnnlDataType_from_dataType(Tensor::DataType data_type)
{
  switch(data_type) {
  case Tensor::DataType::FLOAT:
    return dnnl_f32;
  case Tensor::DataType::HALF:
    return dnnl_f16;
  case Tensor::DataType::U8:
    return dnnl_u8;
  case Tensor::DataType::I32:
    return dnnl_s32;
  default:
    fprintf(stderr, "Unsupported data_type %d for dnnl tensor\n",
            (int)data_type);
    abort();
  }
}


class DnnlOperation {
public:
  virtual ~DnnlOperation() {}
  virtual void exec(DnnlProgram &p) = 0;
  virtual void print() const = 0;
};


std::shared_ptr<Tensor>
DnnlProgram::resolveTensor(std::shared_ptr<Tensor> src)
{
  if(src == nullptr)
    return nullptr;

  auto it = tensors_.find(src);
  if(it != tensors_.end()) {
    return it->second;
  }
  return nullptr;
}


void
DnnlProgram::infer()
{
  for(const auto &op : infer_operations_) {
    op->exec(*this);
  }
}

void
DnnlProgram::train()
{
  fprintf(stderr, "%s not implemented\n", __FUNCTION__);
  usleep(100000);
}


void
DnnlProgram::print() const
{
  for(const auto &op : infer_operations_) {
    op->print();
  }
}

void
DnnlProgram::debug(bool debug)
{
  debug_ = debug;
}


std::shared_ptr<DnnlTensor>
DnnlProgram::lower_tensor(std::shared_ptr<Tensor> src)
{
  if(src == nullptr)
    return nullptr;

  auto it = tensors_.find(src);
  if(it != tensors_.end()) {
    return it->second;
  }

  dnnl_memory_desc_t desc;

  dnnl_format_tag_t ft = (dnnl_format_tag_t)(dnnl_a + src->dims_.size() - 1);

  chkDNNL(dnnl_memory_desc_init_by_tag(&desc, src->dims_.size(),
                                       &src->dims_.i64()[0],
                                       dnnlDataType_from_dataType(src->data_type_),
                                       ft));

  auto t = std::make_shared<DnnlTensor>(&desc, ctx_, src->name_);
  t->copyFromLocked(*src);
  tensors_[src] = t;
  return t;
}

std::shared_ptr<DnnlTensor>
DnnlProgram::lower_tensor(std::shared_ptr<Tensor> src,
                          const dnnl_memory_desc_t *desc)
{
  if(src == nullptr)
    return nullptr;

  auto it = tensors_.find(src);
  if(it != tensors_.end()) {
    return it->second;
  }

  auto t = std::make_shared<DnnlTensor>(desc, ctx_, src->name_);
  t->copyFromLocked(*src);
  tensors_[src] = t;
  return t;
}


std::shared_ptr<DnnlTensor>
DnnlProgram::lower_tensor(std::shared_ptr<Tensor> src,
                          dnnl_primitive_desc_t desc,
                          dnnl_query_t what)
{
  return lower_tensor(src, dnnl_primitive_desc_query_md(desc, what, 0));
}


std::shared_ptr<DnnlTensor>
DnnlProgram::lower_tensor_batch(std::shared_ptr<Tensor> src)
{
  if(src == nullptr)
    return nullptr;

  auto it = tensors_.find(src);
  if(it != tensors_.end()) {
    return it->second;
  }

  dnnl_format_tag_t ft = (dnnl_format_tag_t)(dnnl_a + src->dims_.size() - 1);

  dnnl_memory_desc_t desc;

  chkDNNL(dnnl_memory_desc_init_by_tag(&desc, src->dims_.size(),
                                       &src->dims_.n(batch_size_).i64()[0],
                                       dnnlDataType_from_dataType(src->data_type_),
                                       ft));

  auto t = std::make_shared<DnnlTensor>(&desc, ctx_, src->name_);
  t->copyFromLocked(*src);
  tensors_[src] = t;
  return t;
}





dnnl_memory_desc_t
dnnl_desc_from_tensor(std::shared_ptr<Tensor> t, size_t dimensions,
                      dnnl_format_tag_t format_tag)
{
  dnnl_memory_desc_t desc;

  if(t == nullptr) {
    memset(&desc, 0, sizeof(desc));
    return desc;
  }
  dnnl_dims_t dims;

  assert(t->dims_.size() <= DNNL_MAX_NDIMS);

  int j = 0;
  size_t k = 0;
  if(dimensions) {
    if(dimensions > t->dims_.size()) {
      for(size_t i = 0; i < dimensions - t->dims_.size(); i++)
        dims[j++] = 1;
    } else if(dimensions < t->dims_.size()) {
      k = t->dims_.size() - dimensions;
    }
  }

  for(;k < t->dims_.size(); k++)
    dims[j++] = t->dims_[k];

  chkDNNL(dnnl_memory_desc_init_by_tag(&desc, j, dims,
                                     dnnlDataType_from_dataType(t->data_type_),
                                       format_tag));
  return desc;
}



dnnl_memory_desc_t
DnnlProgram::dnnl_desc_from_tensor_any(std::shared_ptr<Tensor> t)
{
  dnnl_memory_desc_t desc;
  dnnl_dims_t dims;

  assert(t->dims_.size() <= DNNL_MAX_NDIMS);

  for(size_t i = 0; i < t->dims_.size(); i++)
    dims[i] = t->dims_[i];

  dims[0] = batch_size_;

  chkDNNL(dnnl_memory_desc_init_by_tag(&desc, t->dims_.size(), dims,
                                     dnnlDataType_from_dataType(t->data_type_),
                                       dnnl_format_tag_any));
  return desc;
}


dnnl_memory_desc_t
DnnlProgram::dnnl_desc_from_tensor(std::shared_ptr<Tensor> t)
{
  dnnl_memory_desc_t desc;
  dnnl_dims_t dims;

  assert(t->dims_.size() <= DNNL_MAX_NDIMS);

  for(size_t i = 0; i < t->dims_.size(); i++)
    dims[i] = t->dims_[i];

  dims[0] = batch_size_;

  dnnl_format_tag_t ft = (dnnl_format_tag_t)(dnnl_a + t->dims_.size() - 1);

  chkDNNL(dnnl_memory_desc_init_by_tag(&desc, t->dims_.size(), dims,
                                     dnnlDataType_from_dataType(t->data_type_),
                                       ft));
  return desc;
}






//------------------------------------------------------------------------




//------------------------------------------------------------------------


struct DnnlPrimitive : public DnnlOperation {

  DnnlPrimitive(dnnl_primitive_desc_t desc,
                const std::vector<dnnl_exec_arg_t> &args)
    : desc_(desc)
    , args_(args)
  {
    chkDNNL(dnnl_primitive_create(&prim_, desc_));
  }

  ~DnnlPrimitive()
  {
    chkDNNL(dnnl_primitive_destroy(prim_));
    chkDNNL(dnnl_primitive_desc_destroy(desc_));
  }

  void print() const {

    dnnl_primitive_kind_t prim_kind = dnnl_undefined_primitive;
    dnnl_prop_kind_t prop_kind = dnnl_prop_kind_undef;
    const char *impl_info = "?";
    dnnl_primitive_desc_query(desc_, dnnl_query_primitive_kind, 0, &prim_kind);
    dnnl_primitive_desc_query(desc_, dnnl_query_prop_kind, 0, &prop_kind);
    dnnl_primitive_desc_query(desc_, dnnl_query_impl_info_str, 0, &impl_info);
    printf("%-20s %-20s %-20s\n",
           dnnl_prim_kind2str(prim_kind),
           dnnl_prop_kind2str(prop_kind),
           impl_info);
  }


  void exec(DnnlProgram &p) {
    chkDNNL(dnnl_primitive_execute(prim_, p.ctx_->stream_,
                                   args_.size(), &args_[0]));
  }

  dnnl_primitive_desc_t desc_;
  dnnl_primitive_t prim_;
  std::vector<dnnl_exec_arg_t> args_;
};


static void __attribute__((unused))
print_desc(const char *prefix, const dnnl_memory_desc_t *desc)
{
  char tmp1[512];
  char tmp2[512];
  dnnl_md2fmt_str(tmp1, sizeof(tmp1), desc);
  dnnl_md2dim_str(tmp2, sizeof(tmp2), desc);

  printf("%s: %s [%s]\n", prefix, tmp1, tmp2);
}

//------------------------------------------------------------------------


static void
conv_infer(DnnlProgram &p, const Node &n)
{
  auto xh = n.inputs_.get("x");
  auto wh = n.inputs_.get("w");
  auto bh = n.inputs_.get("b");
  auto yh = n.outputs_.get("y");

  auto x_desc = p.dnnl_desc_from_tensor_any(xh);
  auto w_desc = dnnl_desc_from_tensor(wh, 0, dnnl_format_tag_any);
  auto b_desc = dnnl_desc_from_tensor(bh, 1, dnnl_a);
  auto y_desc = p.dnnl_desc_from_tensor_any(yh);

  const int pad = n.attributes_.get("pad", 0);
  const int stride = n.attributes_.get("stride", 1);

  dnnl_dims_t strides = {stride, stride, stride};
  dnnl_dims_t padding = {pad,    pad,    pad};

  dnnl_convolution_desc_t conv_desc;
  chkDNNL(dnnl_convolution_forward_desc_init(&conv_desc,
                                             dnnl_forward_inference,
                                             dnnl_convolution_auto, &x_desc,
                                             &w_desc, &b_desc, &y_desc,
                                             strides, padding, NULL));

  dnnl_primitive_desc_t pd;
  chkDNNL(dnnl_primitive_desc_create(&pd, &conv_desc, NULL,
                                     p.ctx_->engine_, NULL));

  auto x = p.lower_tensor(xh, pd, dnnl_query_src_md);
  auto w = p.lower_tensor(wh, pd, dnnl_query_weights_md);
  auto y = p.lower_tensor(yh, pd, dnnl_query_dst_md);
  auto b = p.lower_tensor(bh, &b_desc);

  std::vector<dnnl_exec_arg_t> args;
  args.push_back({DNNL_ARG_SRC,     x->memory_});
  args.push_back({DNNL_ARG_WEIGHTS, w->memory_});
  args.push_back({DNNL_ARG_BIAS,    b->memory_});
  args.push_back({DNNL_ARG_DST,     y->memory_});

  p.infer(std::make_shared<DnnlPrimitive>(pd, args));
}

//------------------------------------------------------------------------


static void
relu_infer(DnnlProgram &p, const Node &n)
{
  auto xh = n.inputs_.get("x");
  auto yh = n.outputs_.get("y");

  auto x = p.lower_tensor_batch(xh);
  auto y = p.lower_tensor(yh, &x->desc_);

  dnnl_eltwise_desc_t relu_desc;
  chkDNNL(dnnl_eltwise_forward_desc_init(&relu_desc, dnnl_forward,
                                         dnnl_eltwise_relu, &x->desc_,
                                         0.0f, 0));

  dnnl_primitive_desc_t pd;
  chkDNNL(dnnl_primitive_desc_create(&pd, &relu_desc, NULL,
                                     p.ctx_->engine_, NULL));


  std::vector<dnnl_exec_arg_t> args;
  args.push_back({DNNL_ARG_SRC,     x->memory_});
  args.push_back({DNNL_ARG_DST,     y->memory_});

  p.infer(std::make_shared<DnnlPrimitive>(pd, args));
}

//------------------------------------------------------------------------

static void
maxpool_infer(DnnlProgram &p, const Node &n)
{
  auto xh = n.inputs_.get("x");
  auto yh = n.outputs_.get("y");
  auto x = p.lower_tensor_batch(xh);
  auto y_desc = p.dnnl_desc_from_tensor_any(yh);

  int size;
  if(n.attributes_.get("global", false)) {
    size = x->dims_[2];
  } else {
    size = n.attributes_.get("size", 1);
  }
  const int pad    = n.attributes_.get("pad", 0);
  const int stride = n.attributes_.get("stride", 1);

  dnnl_dims_t kernel  = {size,   size,   size};
  dnnl_dims_t strides = {stride, stride, stride};
  dnnl_dims_t padding = {pad,    pad,    pad};

  dnnl_pooling_desc_t desc;
  chkDNNL(dnnl_pooling_forward_desc_init(&desc, dnnl_forward_inference,
                                         dnnl_pooling_max,
                                         &x->desc_, &y_desc,
                                         strides, kernel, padding, NULL));

  dnnl_primitive_desc_t pd;
  chkDNNL(dnnl_primitive_desc_create(&pd, &desc, NULL,
                                     p.ctx_->engine_, NULL));

  auto y = p.lower_tensor(yh, pd, dnnl_query_dst_md);

  std::vector<dnnl_exec_arg_t> args;
  args.push_back({DNNL_ARG_SRC,     x->memory_});
  args.push_back({DNNL_ARG_DST,     y->memory_});

  p.infer(std::make_shared<DnnlPrimitive>(pd, args));
}

//------------------------------------------------------------------------

static void
reshape_infer(DnnlProgram &p, const Node &n)
{
  auto xh = n.inputs_.get("x");
  auto yh = n.outputs_.get("y");

  auto x = p.lower_tensor_batch(xh);
  auto y = p.lower_tensor_batch(yh);

  dnnl_memory_desc_t dst_desc;

  dnnl_format_tag_t ft = (dnnl_format_tag_t)(dnnl_a + x->dims_.size() - 1);

  chkDNNL(dnnl_memory_desc_init_by_tag(&dst_desc, x->dims_.size(),
                                       &x->dims_.i64()[0],
                                       dnnlDataType_from_dataType(x->data_type_),
                                       ft));

  dnnl_primitive_desc_t pd;
  chkDNNL(dnnl_reorder_primitive_desc_create(&pd,
                                             &x->desc_, p.ctx_->engine_,
                                             &dst_desc, p.ctx_->engine_,
                                             NULL));


  std::vector<dnnl_exec_arg_t> args;
  args.push_back({DNNL_ARG_SRC,     x->memory_});
  args.push_back({DNNL_ARG_DST,     y->memory_});

  p.infer(std::make_shared<DnnlPrimitive>(pd, args));
}

//------------------------------------------------------------------------


static void
fc_infer(DnnlProgram &p, const Node &n)
{
  auto xh = n.inputs_.get("x");
  auto wh = n.inputs_.get("w");
  auto bh = n.inputs_.get("b");
  auto yh = n.outputs_.get("y");

  bool transW = n.attributes_.get("transW", false);

  dnnl_memory_desc_t w_desc;
  if(transW) {
    dnnl_dims_t w_dims = {wh->dims_[0], wh->dims_[1]};
    chkDNNL(dnnl_memory_desc_init_by_tag(&w_desc, 2,
                                         w_dims,
                                         dnnlDataType_from_dataType(wh->data_type_),
                                         dnnl_ab));

  } else {
    dnnl_dims_t w_dims = {wh->dims_[1], wh->dims_[0]};
    chkDNNL(dnnl_memory_desc_init_by_tag(&w_desc, 2,
                                         w_dims,
                                         dnnlDataType_from_dataType(wh->data_type_),
                                         dnnl_ba));
  }

  dnnl_memory_desc_t *b_desc = NULL, b_desc0;

  if(bh) {
    dnnl_dims_t b_dims = {yh->dims_[1]};
    chkDNNL(dnnl_memory_desc_init_by_tag(&b_desc0, 1,
                                         b_dims,
                                         dnnlDataType_from_dataType(bh->data_type_),
                                         dnnl_a));
    b_desc = &b_desc0;
  }


  auto x = p.lower_tensor_batch(xh);
  auto w = p.lower_tensor(wh);
  auto b = bh ? p.lower_tensor(bh) : nullptr;
  auto y = p.lower_tensor_batch(yh);

  dnnl_inner_product_desc_t desc;
  chkDNNL(dnnl_inner_product_forward_desc_init(&desc,
                                               dnnl_forward_inference,
                                               &x->desc_,
                                               &w_desc,
                                               b_desc,
                                               &y->desc_));

  dnnl_primitive_desc_t pd;
  chkDNNL(dnnl_primitive_desc_create(&pd, &desc, NULL,
                                     p.ctx_->engine_, NULL));

  std::vector<dnnl_exec_arg_t> args;
  args.push_back({DNNL_ARG_SRC,     x->memory_});
  args.push_back({DNNL_ARG_WEIGHTS, w->memory_});
  args.push_back({DNNL_ARG_DST,     y->memory_});
  if(b)
    args.push_back({DNNL_ARG_BIAS,    b->memory_});

  p.infer(std::make_shared<DnnlPrimitive>(pd, args));
}

//------------------------------------------------------------------------

static void
concat_infer(DnnlProgram &p, const Node &n)
{
  const int axis = 1;

  auto xhv = n.inputs_.getv("x");
  auto yh = n.outputs_.get("y");
  auto y_desc = p.dnnl_desc_from_tensor_any(yh);

  dnnl_memory_desc_t src_descs[xhv.size()];
  std::shared_ptr<DnnlTensor> xv[xhv.size()];

  for(size_t i = 0; i < xhv.size(); i++) {
    xv[i] = p.lower_tensor_batch(xhv[i]);
    src_descs[i] = xv[i]->desc_;
  }

  dnnl_primitive_desc_t pd;
  chkDNNL(dnnl_concat_primitive_desc_create(&pd, &y_desc, xhv.size(),
                                            axis, src_descs, NULL,
                                            p.ctx_->engine_));

  std::vector<dnnl_exec_arg_t> args;

  auto y = p.lower_tensor(yh, pd, dnnl_query_dst_md);

  args.push_back({DNNL_ARG_DST, y->memory_});
  for(size_t i = 0; i < xhv.size(); i++) {
    args.push_back({DNNL_ARG_MULTIPLE_SRC + (int)i, xv[i]->memory_});
  }

  p.infer(std::make_shared<DnnlPrimitive>(pd, args));

}




//------------------------------------------------------------------------

static const struct Operation {
  const char *name;
  void (*create_infer)(DnnlProgram &p, const Node &n);
  void (*create_train)(DnnlProgram &p, const Node &n);
} nodetypes[] = {
  { "conv",             conv_infer},
  { "relu",             relu_infer},
  { "maxpool",          maxpool_infer},
  { "reshape",          reshape_infer},
  { "fc",               fc_infer},
  { "dropout",          reshape_infer},
  { "concat",           concat_infer},
};

static const Operation *
find_operation(const Node &n)
{
  for(size_t i = 0; i < sizeof(nodetypes) / sizeof(nodetypes[0]); i++) {
    if(n.type_ == nodetypes[i].name) {
      return &nodetypes[i];
    }
  }
  return NULL;
}



std::shared_ptr<Program>
DnnlContext::createProgram(const Graph &g,
                           const ProgramConfig &pc)
{

  auto p = std::make_shared<DnnlProgram>(shared_from_this(),
                                         pc.tensor_layout,
                                         pc.batch_size,
                                         pc.initial_learning_rate);
  if(pc.training) {

    fprintf(stderr, "DNNL Training not implemented yet\n");
  }

  if(pc.inference) {
    for(const auto &n : g.nodes_) {
      auto op = find_operation(*n);
      if(op != NULL && op->create_infer) {
        op->create_infer(*p, *n);
      } else {
        fprintf(stderr, "Unable to create inference operation for node %s\n",
                n->type_.c_str());
        n->print();
        exit(1);
      }
    }
  }
  return p;

}


}














