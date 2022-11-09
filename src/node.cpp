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
#include "saga.hpp"

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
    const int stride = n.m_attributes.get("stride", 1);
    const int pad = n.m_attributes.get("pad", 0);
    const int dilation = n.m_attributes.get("dilation", 1);
    const int group = n.m_attributes.get("group", 1);
    const int transpose = n.m_attributes.get("transpose", false);

    auto w = n.m_inputs.get("w");
    if(w == nullptr)
        return nullptr;
    const int features = w->m_dims[transpose ? 1 : 0] * group;
    const int filterdim_h = w->m_dims[2];
    const int filterdim_w = w->m_dims[3];

    auto x = n.m_inputs.get("x");
    if(x == nullptr)
        return nullptr;

    const int inputdim_h = x->m_dims[2];
    const int inputdim_w = x->m_dims[3];

    int odw;
    int odh;

    if(transpose) {
        const int opad = n.m_attributes.get("outputpad", 0);

        odw = stride * (inputdim_w - 1) + opad +
              ((filterdim_w - 1) * dilation + 1) - 2 * pad;

        odh = stride * (inputdim_h - 1) + opad +
              ((filterdim_h - 1) * dilation + 1) - 2 * pad;
    } else {
        odw =
            1 + (inputdim_w + 2 * pad - (((filterdim_w - 1) * dilation) + 1)) /
                    stride;

        odh =
            1 + (inputdim_h + 2 * pad - (((filterdim_h - 1) * dilation) + 1)) /
                    stride;
    }

    return makeTensor(x->m_data_type, Dims({x->m_dims[0], features, odh, odw}),
                      name);
}

static std::vector<std::shared_ptr<Node>>
conv_setup(std::shared_ptr<Node> n, Tensors &named_tensors)
{
    auto x = n->m_inputs.get("x");
    if(!x)
        return {};

    auto w = n->m_inputs.get("w");
    auto b = n->m_inputs.get("b");

    if(!w) {
        const int activations = n->m_attributes.get("activations", 1);
        const int size = n->m_attributes.get("size", 1);
        const int transpose = n->m_attributes.get("transpose", false);
        const int group = n->m_attributes.get("group", 1);

        const int in_features =
            (transpose ? activations : (int)x->m_dims[1]) / group;
        const int out_features = transpose ? (int)x->m_dims[1] : activations;

        n->m_inputs["w"] = w = Tensor::find(
            x->m_data_type, {out_features, in_features, size, size}, 0,
            sqrt(2.0 / (in_features * size * size)), named_tensors,
            node_tensor_name(n->m_name, "w"));
    }

    if(!b && n->m_attributes.get("bias", false)) {
        const int transpose = n->m_attributes.get("transpose", false);
        const int group = n->m_attributes.get("group", 1);

        n->m_inputs["b"] = Tensor::find(
            x->m_data_type, {1, w->m_dims[!!transpose] * group}, 0, 0,
            named_tensors, node_tensor_name(n->m_name, transpose ? "b" : "bt"));
    }

    return {n};
}

//------------------------------------------------------------------------

static std::vector<std::shared_ptr<Node>>
batchnorm_setup(std::shared_ptr<Node> n, Tensors &named_tensors)
{
    auto x = n->m_inputs.get("x");
    if(!x)
        return {};

    Dims dims{1, x->m_dims[1]};

    if(!n->m_inputs.get("s")) {
        n->m_inputs["s"] =
            Tensor::find(Tensor::DataType::FLOAT, dims, 1.0, 0, named_tensors,
                         node_tensor_name(n->m_name, "s"));
    }

    if(!n->m_inputs.get("b")) {
        n->m_inputs["b"] =
            Tensor::find(Tensor::DataType::FLOAT, dims, 0.0, 0, named_tensors,
                         node_tensor_name(n->m_name, "b"));
    }

    if(!n->m_inputs.get("m")) {
        n->m_inputs["m"] =
            Tensor::find(Tensor::DataType::FLOAT, dims, 0.0, 0, named_tensors,
                         node_tensor_name(n->m_name, "m"));
    }

    if(!n->m_inputs.get("v")) {
        n->m_inputs["v"] =
            Tensor::find(Tensor::DataType::FLOAT, dims, 1.0, 0, named_tensors,
                         node_tensor_name(n->m_name, "v"));
    }
    return {n};
}

//------------------------------------------------------------------------

static std::shared_ptr<Tensor>
pooling_y(const Node &n, const std::optional<const std::string> &name)
{
    // Should make this more generic for n-dimensions

    const int pad = n.m_attributes.get("pad", 0);
    const int stride = n.m_attributes.get("stride", 1);
    auto x = n.m_inputs.get("x");
    if(x == nullptr)
        return nullptr;

    int size;

    if(n.m_attributes.get("global", false)) {
        size = x->m_dims[2];
        assert(x->m_dims[3] == size);
    } else {
        size = n.m_attributes.get("size", 1);
    }

    const int channels = x->m_dims[1];
    const int inputdim_h = x->m_dims[2];
    const int inputdim_w = x->m_dims[3];

    const int outputdim_h = 1 + (inputdim_h + 2 * pad - size) / stride;
    const int outputdim_w = 1 + (inputdim_w + 2 * pad - size) / stride;

    return makeTensor(x->m_data_type,
                      Dims({x->m_dims[0], channels, outputdim_h, outputdim_w}),
                      name);
}

//------------------------------------------------------------------------

static std::shared_ptr<Tensor>
reshape_y(const Node &n, const std::optional<const std::string> &name)
{
    auto x = n.m_inputs.get("x");
    if(x == nullptr)
        return nullptr;
    auto it = n.m_attributes.find("shape");
    if(it == n.m_attributes.end()) {
        fprintf(stderr, "Shape attribute not found\n");
        abort();
        return nullptr;
    }

    auto shapep = std::get_if<Dims>(&it->second);
    if(!shapep) {
        fprintf(stderr, "Shape attribute not Dimension object\n");
        return nullptr;
    }

    const auto xshape = x->m_dims;

    const auto shape = (*shapep).transform([&](auto dp, size_t i) {
        switch(dp) {
        case DimParam::UNCHANGED:
            return xshape[i];
        case DimParam::REDUCE:
            return (Dim)(int64_t)xshape.elements(i);
        default:
            return (Dim)dp;
        }
    });

    return makeTensor(x->m_data_type, shape, name);
}

//------------------------------------------------------------------------

static std::shared_ptr<Tensor>
concat_y(const Node &n, const std::optional<const std::string> &name)
{
    int i = 0;
    int axis = n.m_attributes.get("axis", 1);
    Dims dims;
    Tensor::DataType data_type = Tensor::DataType::U8;
    while(1) {
        auto x = n.m_inputs.get("x" + std::to_string(i));
        if(x == nullptr)
            break;
        if(i == 0) {
            dims = x->m_dims;
            data_type = x->m_data_type;
        } else {
            dims[axis] += x->m_dims[axis];
        }
        i++;
    }
    return makeTensor(data_type, dims, name);
}

//------------------------------------------------------------------------

static std::shared_ptr<Tensor>
window_y(const Node &n, const std::optional<const std::string> &name)
{
    auto o = n.m_inputs.get("x");
    if(o == nullptr)
        return nullptr;

    auto shape = n.m_attributes.get("shape", std::vector<int>{});
    if(shape.size() != o->m_dims.size())
        return nullptr;
    return makeTensor(o->m_data_type, shape, name);
}

//------------------------------------------------------------------------

static std::shared_ptr<Tensor>
fc_y(const Node &n, const std::optional<const std::string> &name)
{
    auto w = n.m_inputs.get("w");
    if(w == nullptr)
        return nullptr;
    auto x = n.m_inputs.get("x");
    const bool transW = n.m_attributes.get("transW", false);

    return makeTensor(w->m_data_type,
                      Dims({x->m_dims[0], w->m_dims[transW ? 0 : 1]}), name);
}

static std::vector<std::shared_ptr<Node>>
fc_setup(std::shared_ptr<Node> n, Tensors &named_tensors)
{
    std::vector<std::shared_ptr<Node>> nodes;

    auto x = n->m_inputs.get("x");
    if(!x)
        return nodes;

    if(x->m_dims.size() > 2) {
        // Auto-insert a reshape node
        auto r = std::make_shared<Node>("reshape");
        r->m_inputs["x"] = x;
        r->m_attributes["shape"] =
            Dims({DimParam::UNCHANGED, DimParam::REDUCE});
        x = r->inferTensor_y();
        r->m_outputs["y"] = x;
        n->m_inputs["x"] = x;
        nodes.push_back(r);
    }

    auto w = n->m_inputs.get("w");
    auto b = n->m_inputs.get("b");

    const bool transW = n->m_attributes.get("transW", false);

    if(!w) {
        const int outputs = n->m_attributes.get("outputs", 1);

        Dims d;
        if(transW) {
            d = Dims({outputs, x->m_dims[1]});
        } else {
            d = Dims({x->m_dims[1], outputs});
        }

        n->m_inputs["w"] = w =
            Tensor::find(x->m_data_type, d, 0, sqrt(2.0 / x->m_dims[1]),
                         named_tensors, node_tensor_name(n->m_name, "w"));
    }

    if(!b && n->m_attributes.get("bias", false)) {
        n->m_inputs["b"] = Tensor::find(
            x->m_data_type, {transW ? w->m_dims[0] : w->m_dims[1]}, 0, 0,
            named_tensors, node_tensor_name(n->m_name, transW ? "bt" : "b"));
    }
    nodes.push_back(n);
    return nodes;
}

//------------------------------------------------------------------------

static std::shared_ptr<Tensor>
catclassifier_y(const Node &n, const std::optional<const std::string> &name)
{
    auto x = n.m_inputs.get("x");
    if(x == nullptr)
        return nullptr;
    return makeTensor(Tensor::DataType::I32, Dims({x->m_dims[0], 1}), name);
}

static std::vector<std::shared_ptr<Node>>
catclassifier_setup(std::shared_ptr<Node> n, Tensors &named_tensors)
{
    n->m_outputs["loss"] =
        makeTensor(Tensor::DataType::FLOAT, Dims({1, 1}), "loss");
    return {n};
}

//------------------------------------------------------------------------

static std::shared_ptr<Tensor>
loss_y(const Node &n, const std::optional<const std::string> &name)
{
    auto o = n.m_inputs.get("x");
    if(o == nullptr)
        return nullptr;

    auto target = n.m_inputs.get("target");
    if(target == nullptr)
        return nullptr;

    return makeTensor(target->m_data_type, target->m_dims, name);
}

static std::vector<std::shared_ptr<Node>>
loss_setup(std::shared_ptr<Node> n, Tensors &named_tensors)
{
    auto x = n->m_inputs.get("x");
    n->m_outputs["mmss"] =
        makeTensor(Tensor::DataType::FLOAT, Dims({x->m_dims[0], 4}));
    return {n};
}

//------------------------------------------------------------------------

static std::shared_ptr<Tensor>
passthru_y(const Node &n, const std::optional<const std::string> &name)
{
    auto o = n.m_inputs.get("x");
    if(o == nullptr)
        return nullptr;
    return makeTensor(o->m_data_type, o->m_dims, name);
}

//------------------------------------------------------------------------

static std::shared_ptr<Tensor>
sum_y(const Node &n, const std::optional<const std::string> &name)
{
    auto x0 = n.m_inputs.get("x0");
    auto x1 = n.m_inputs.get("x0");
    if(x0 == nullptr || x1 == nullptr)
        return nullptr;

    if(x0->m_dims != x1->m_dims || x0->m_data_type != x1->m_data_type)
        return nullptr;

    return makeTensor(x0->m_data_type, x0->m_dims, name);
}

//------------------------------------------------------------------------

static std::shared_ptr<Tensor>
convert_y(const Node &n, const std::optional<const std::string> &name)
{
    auto x = n.m_inputs.get("x");
    if(x == nullptr)
        return nullptr;

    auto datatype = n.m_attributes.get("datatype", -1);
    if(datatype == -1)
        return nullptr;

    return makeTensor((Tensor::DataType)datatype, x->m_dims, name);
}

//------------------------------------------------------------------------

static std::shared_ptr<Tensor>
jpegdecoder_y(const Node &n, const std::optional<const std::string> &name)
{
    const int width = n.m_attributes.get("width", 0);
    if(width < 1)
        return nullptr;
    const int height = n.m_attributes.get("height", 0);
    if(height < 1)
        return nullptr;
    const int channels = n.m_attributes.get("channels", 3);

    return makeTensor(Tensor::DataType::U8, Dims({1, channels, width, height}),
                      name);
}

//------------------------------------------------------------------------

static std::shared_ptr<Tensor>
spatialtransform_y(const Node &n, const std::optional<const std::string> &name)
{
    auto o = n.m_inputs.get("x");
    if(o == nullptr)
        return nullptr;

    const int height = n.m_attributes.get("height", (int)o->m_dims[2]);
    const int width = n.m_attributes.get("width", (int)o->m_dims[3]);

    return makeTensor(o->m_data_type,
                      Dims({o->m_dims[0], o->m_dims[1], height, width}));
}

//------------------------------------------------------------------------

static std::shared_ptr<Tensor>
stats_y(const Node &n, const std::optional<const std::string> &name)
{
    return makeTensor(Tensor::DataType::FLOAT, Dims{4}, "stats");
}

//------------------------------------------------------------------------
//------------------------------------------------------------------------

static const struct {
    const char *name;

    std::shared_ptr<Tensor> (*infer_y)(
        const Node &n, const std::optional<const std::string> &name);

    std::vector<std::shared_ptr<Node>> (*setup)(std::shared_ptr<Node> node,
                                                Tensors &named_tensors);

} nodetypes[] = {
    {"add", passthru_y},
    {"avgpool", pooling_y},
    {"batchnorm", passthru_y, batchnorm_setup},
    {"catclassifier", catclassifier_y, catclassifier_setup},
    {"concat", concat_y},
    {"conv", conv_y, conv_setup},
    {"dropout", passthru_y},
    {"elu", passthru_y},
    {"fc", fc_y, fc_setup},
    {"jpegdecoder", jpegdecoder_y},
    {"maxpool", pooling_y},
    {"mul", passthru_y},
    {"relu", passthru_y},
    {"swish", passthru_y},
    {"leakyrelu", passthru_y},
    {"reshape", reshape_y},
    {"softmax", passthru_y},
    {"sigmoid", passthru_y},
    {"tanh", passthru_y},
    {"spatialtransform", spatialtransform_y},
    {"stats", stats_y},
    {"sum", sum_y},
    {"convert", convert_y},
    {"loss", loss_y, loss_setup},
    {"window", window_y},
};

std::shared_ptr<Tensor>
Node::inferTensor_y(const std::optional<const std::string> &name)
{
    for(size_t i = 0; i < sizeof(nodetypes) / sizeof(nodetypes[0]); i++) {
        if(m_type == nodetypes[i].name) {
            return nodetypes[i].infer_y(*this, name);
        }
    }

    fprintf(stderr, "Failed to compute output tensor for type %s\n",
            m_type.c_str());
    print();
    abort();
}

std::string
Attribute::to_string() const
{
    if(auto v = std::get_if<int>(this)) {
        return std::to_string(*v);
    } else if(auto v = std::get_if<float>(this)) {
        return std::to_string(*v);
    } else if(auto v = std::get_if<bool>(this)) {
        return *v ? "true" : "false";
    } else if(auto v = std::get_if<std::vector<int>>(this)) {
        std::string r("[");
        const char *pfx = "";
        for(const auto &ele : *v) {
            r += pfx;
            r += std::to_string(ele);
            pfx = ", ";
        }
        r += "]";
        return r;
    } else if(auto v = std::get_if<Dims>(this)) {
        return v->to_string();
    }
    return "?";
}

void
Node::print() const
{
    printf("%s:\n", m_type.c_str());

    for(const auto &t : m_inputs) {
        printf("\t Input: %s: %s\n", t.first.c_str(), t.second->info().c_str());
    }

    for(const auto &t : m_outputs) {
        printf("\tOutput: %s: %s\n", t.first.c_str(), t.second->info().c_str());
    }

    for(const auto &a : m_attributes) {
        printf("\tAttrib: %s: %s\n", a.first.c_str(),
               a.second.to_string().c_str());
    }
}

std::vector<std::shared_ptr<Node>>
Node::make(const std::string &type, const Tensors &inputs,
           const Attributes &attributes, Tensors &named_tensors,
           const std::optional<const std::string> &name)
{
    auto n = std::make_shared<Node>(type, name);
    n->m_inputs = inputs;
    n->m_attributes = attributes;

    std::vector<std::shared_ptr<Node>> nodes({n});

    for(size_t i = 0; i < sizeof(nodetypes) / sizeof(nodetypes[0]); i++) {
        if(type == nodetypes[i].name && nodetypes[i].setup) {
            nodes = nodetypes[i].setup(n, named_tensors);
        }
    }

    if(!nodes.empty()) {
        auto &last = nodes.back();
        auto y = last->inferTensor_y();
        if(!y) {
            fprintf(stderr, "%s: Unable to infer y tensor\n", __FUNCTION__);
            last->print();
            exit(1);
        }
        last->m_outputs["y"] = y;
    }
    return nodes;
}

std::vector<std::shared_ptr<Node>>
Node::make(const std::string &type, Loader loader, const Attributes &attributes)
{
    auto n = std::make_shared<Node>(type);
    n->m_loader = loader;
    n->m_attributes = attributes;

    n->m_outputs["y"] = n->inferTensor_y();
    std::vector<std::shared_ptr<Node>> nodes({n});
    return nodes;
}

std::shared_ptr<Tensor>
Node::y()
{
    return m_outputs.get("y");
}

Nodes::iterator
Nodes::findSingleDownStreamNode(std::shared_ptr<Tensor> t,
                                const std::string &type)
{
    Nodes::iterator it = end();

    for(auto jt = begin(); jt != end(); jt++) {
        const auto &n = *jt;
        if(n->m_type != type)
            continue;
        for(const auto &s : n->m_inputs) {
            if(s.second == t) {
                if(it == end()) {
                    it = jt;
                } else {
                    // Multiple
                    return end();
                }
            }
        }
    }
    return it;
}

}  // namespace saga
