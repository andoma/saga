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

#include <sys/stat.h>
#include <errno.h>
#include <string.h>
#include <dirent.h>
#include "saga.hpp"

namespace saga {

void
Tensors::loadTensors(const char *path)
{
    struct dirent **namelist;
    int n = scandir(path, &namelist, NULL, NULL);
    if(n == -1) {
        fprintf(stderr, "Unable to load tensors from %s -- %s\n", path,
                strerror(errno));
        return;
    }

    while(n--) {
        const char *fname = namelist[n]->d_name;
        if(fname[0] != '.') {
            char filepath[PATH_MAX];
            snprintf(filepath, sizeof(filepath), "%s/%s", path, fname);
            auto t = Tensor::load(filepath, fname);
            if(t) {
                (*this)[fname] = t;
            }
        }
        free(namelist[n]);
    }
    free(namelist);
}

bool
Tensors::saveTensors(const char *path, Context *ctx)
{
    char filepath[PATH_MAX];
    mkdir(path, 0777);
    for(const auto &it : *this) {
        auto t = it.second;
        if(ctx != NULL) {
            t = ctx->resolveTensor(t);
            if(t == NULL)
                continue;
        }

        snprintf(filepath, sizeof(filepath), "%s/%s", path, it.first.c_str());
        if(!t->save(filepath))
            return false;
    }
    return true;
}

void
Graph::print() const
{
    for(const auto &n : nodes_) {
        n->print();
    }
}

std::shared_ptr<Node>
Graph::addNode(const std::string &type, const Tensors &inputs,
               const Attributes &attributes,
               const std::optional<const std::string> &name)
{
    auto nodes = Node::make(type, inputs, attributes, *m_named_tensors, name);
    nodes_.insert(nodes_.end(), nodes.begin(), nodes.end());
    if(nodes_.size() == 0)
        return nullptr;
    return nodes_[nodes_.size() - 1];
}

std::shared_ptr<Node>
Graph::addNode(const std::string &type, const std::shared_ptr<Tensor> &t,
               const Attributes &attributes,
               const std::optional<const std::string> &name)
{
    return addNode(type, {{"x", t}}, attributes, name);
}

std::shared_ptr<Node>
Graph::addNode(const std::string &type, Loader loader,
               const Attributes &attributes)
{
    auto nodes = Node::make(type, loader, attributes);
    nodes_.insert(nodes_.end(), nodes.begin(), nodes.end());
    if(nodes_.size() == 0)
        return nullptr;
    return nodes_[nodes_.size() - 1];
}

std::unordered_set<std::shared_ptr<Tensor>>
Graph::inputTensors() const
{
    std::unordered_set<std::shared_ptr<Tensor>> inputs;

    for(const auto &n : nodes_) {
        for(auto &t : n->inputs_) {
            inputs.insert(t.second);
        }
    }
    for(const auto &n : nodes_) {
        for(auto &t : n->outputs_) {
            inputs.erase(t.second);
        }
    }
    return inputs;
}

std::unordered_set<std::shared_ptr<Tensor>>
Graph::outputTensors() const
{
    std::unordered_set<std::shared_ptr<Tensor>> outputs;

    for(const auto &n : nodes_) {
        for(auto &t : n->outputs_) {
            outputs.insert(t.second);
        }
    }
    for(const auto &n : nodes_) {
        for(auto &t : n->inputs_) {
            outputs.erase(t.second);
        }
    }
    return outputs;
}

std::pair<TensorMapping, TensorMapping>
Graph::tensorMappings() const
{
    std::unordered_map<
        std::shared_ptr<Tensor>,
        std::vector<std::pair<std::string, std::shared_ptr<Node>>>>
        input_usage;
    std::unordered_map<
        std::shared_ptr<Tensor>,
        std::vector<std::pair<std::string, std::shared_ptr<Node>>>>
        output_usage;

    for(const auto &n : nodes_) {
        for(const auto &t : n->inputs_) {
            input_usage[t.second].push_back(std::make_pair(t.first, n));
        }
        for(const auto &t : n->outputs_) {
            output_usage[t.second].push_back(std::make_pair(t.first, n));
        }
    }
    return {input_usage, output_usage};
}

void
Graph::loadTensors(const char *path)
{
    m_named_tensors->loadTensors(path);
}

bool
Graph::saveTensors(const char *path, Context *ctx)
{
    return m_named_tensors->saveTensors(path, ctx);
}

void
Graph::statsTensors(Context *ctx)
{
    for(const auto &it : *m_named_tensors) {
        auto t = it.second;
        if(ctx != NULL) {
            t = ctx->resolveTensor(t);
            if(t != NULL)
                t->printStats(it.first.c_str());
        }
    }
}

// -----------------------------------------------------------------------
// Node creation helpers
// -----------------------------------------------------------------------

std::shared_ptr<Node>
Graph::addConvert(std::shared_ptr<Tensor> input, Tensor::DataType dt,
                  float scale)
{
    return addNode("convert", input, {{"scale", scale}, {"datatype", (int)dt}});
}

std::shared_ptr<Node>
Graph::addJpegDecoder(int width, int height, Tensor::DataType dt, Loader loader)
{
    auto n =
        addNode("jpegdecoder", loader, {{"width", width}, {"height", height}});

    return addConvert(n->y(), dt, 1 / 255.0f);
}

std::shared_ptr<Node>
Graph::addSpatialTransform(std::shared_ptr<Tensor> input,
                           std::shared_ptr<Tensor> theta, int width, int height,
                           bool inference)
{
    Attributes a;
    if(width > 0)
        a["width"] = width;
    if(height > 0)
        a["height"] = height;
    a["inference"] = inference;

    return addNode("spatialtransform", Tensors{{"x", input}, {"theta", theta}},
                   a);
}

std::shared_ptr<Node>
Graph::addResNetBottleNeck(std::shared_ptr<Tensor> input,
                           int squeeze_activations, int output_activations,
                           bool downsample, const std::string &name)
{
    std::shared_ptr<Node> a;
    std::shared_ptr<Tensor> x1;
    a = addNode("conv", input,
                {{"size", 1},
                 {"activations", squeeze_activations},
                 {"stride", 1},
                 {"pad", 0}},
                name + "-conv-1");
    a = addNode("batchnorm", a->y(), {}, name + "-bn-1");
    a = addNode("relu", a->y());

    a = addNode("conv", a->y(),
                {{"size", 3},
                 {"activations", squeeze_activations},
                 {"stride", downsample ? 2 : 1},
                 {"pad", 1}},
                name + "-conv-2");
    a = addNode("batchnorm", a->y(), {}, name + "-bn-2");
    a = addNode("relu", a->y());

    a = addNode("conv", a->y(),
                {{"size", 1},
                 {"activations", output_activations},
                 {"stride", 1},
                 {"pad", 0}},
                name + "-conv-3");
    a = addNode("batchnorm", a->y(), {}, name + "-bn-3");

    bool need_projection = a->y()->dims_ != input->dims_;

    if(need_projection) {
        auto c = addNode("conv", input,
                         {{"size", 1},
                          {"activations", output_activations},
                          {"stride", downsample ? 2 : 1},
                          {"pad", 0}},
                         name + "-conv-p");
        c = addNode("batchnorm", c->y(), {}, name + "-bn-p");
        x1 = c->y();
    } else {
        x1 = input;
    }

    auto s = addNode("sum", Tensors{{"x0", a->y()}, {"x1", x1}}, {});
    return addNode("relu", s->y());
}

std::shared_ptr<Node>
Graph::addResNetBottleNeck_transposed(std::shared_ptr<Tensor> input,
                                      int squeeze_activations,
                                      int output_activations, bool upsample,
                                      const std::string &name)
{
    std::shared_ptr<Node> a;
    std::shared_ptr<Tensor> x1;

    a = addNode("conv", input,
                {{"size", 1},
                 {"activations", squeeze_activations},
                 {"stride", 1},
                 {"pad", 0},
                 {"transpose", true}},
                name + "-conv-3");
    a = addNode("batchnorm", a->y(), {}, name + "-bn-3t");
    a = addNode("relu", a->y());

    a = addNode("conv", a->y(),
                {{"size", 3},
                 {"activations", squeeze_activations},
                 {"stride", upsample ? 2 : 1},
                 {"pad", 1},
                 {"outputpad", upsample ? 1 : 0},
                 {"transpose", true}},
                name + "-conv-2");
    a = addNode("batchnorm", a->y(), {}, name + "-bn-2t");
    a = addNode("relu", a->y());

    a = addNode("conv", a->y(),
                {{"size", 1},
                 {"activations", output_activations},
                 {"stride", 1},
                 {"pad", 0},
                 {"transpose", true}},
                name + "-conv-1");
    a = addNode("batchnorm", a->y(), {}, name + "-bn-1t");

    bool need_projection = a->y()->dims_ != input->dims_;

    if(need_projection) {
        auto c = addNode("conv", input,
                         {{"size", 1},
                          {"activations", output_activations},
                          {"stride", upsample ? 2 : 1},
                          {"outputpad", upsample ? 1 : 0},
                          {"transpose", true}},
                         name + "-conv-p");
        c = addNode("batchnorm", c->y(), {}, name + "-bn-pt");
        x1 = c->y();
    } else {
        x1 = input;
    }

    auto s = addNode("sum", Tensors{{"x0", a->y()}, {"x1", x1}}, {});
    return addNode("relu", s->y());
}

std::shared_ptr<Node>
Graph::addResNet(std::shared_ptr<Tensor> input, int output_activations,
                 bool downsample, const std::string &name)
{
    std::shared_ptr<Node> a;
    std::shared_ptr<Tensor> x1;

    a = addNode("conv", input,
                {{"size", 3},
                 {"activations", output_activations},
                 {"stride", downsample ? 2 : 1},
                 {"pad", 1}},
                name + "-conv-1");
    a = addNode("batchnorm", a->y(), {}, name + "-bn-1");
    a = addNode("relu", a->y());

    a = addNode("conv", a->y(),
                {{"size", 3},
                 {"activations", output_activations},
                 {"stride", 1},
                 {"pad", 1}},
                name + "-conv-2");
    a = addNode("batchnorm", a->y(), {}, name + "-bn-2");

    bool need_projection = a->y()->dims_ != input->dims_;

    if(need_projection) {
        auto c = addNode("conv", input,
                         {{"size", 1},
                          {"activations", output_activations},
                          {"stride", downsample ? 2 : 1},
                          {"pad", 0}},
                         name + "-conv-p");
        c = addNode("batchnorm", c->y(), {}, name + "-bn-p");
        x1 = c->y();
    } else {
        x1 = input;
    }

    auto s = addNode("sum", Tensors{{"x0", a->y()}, {"x1", x1}}, {});
    return addNode("relu", s->y());
}

}  // namespace saga
