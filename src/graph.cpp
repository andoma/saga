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

#include "saga.h"

namespace saga {

void
Graph::print() const
{
  for(const auto &n : nodes_) {
    n->print();
  }
}

std::shared_ptr<Tensor>
Graph::addNode(const std::string &type,
               const Tensors &inputs,
               const Attributes &attributes,
               const std::optional<const std::string> &name)
{
  auto nodes = Node::make(type, inputs, attributes, tensors_, name);
  nodes_.insert(nodes_.end(), nodes.begin(), nodes.end());
  if(nodes_.size() == 0)
    return nullptr;
  return nodes_[nodes_.size() - 1]->outputs_["y"];
}



std::pair<TensorMapping, TensorMapping>
Graph::tensorMappings()
{
  std::unordered_map<std::shared_ptr<Tensor>,
                     std::vector<std::shared_ptr<Node>>> input_usage;
  std::unordered_map<std::shared_ptr<Tensor>,
                     std::vector<std::shared_ptr<Node>>> output_usage;

  for(const auto &n : nodes_) {
    for(const auto &t : n->inputs_) {
      input_usage[t.second].push_back(n);
    }
    for(const auto &t : n->outputs_) {
      output_usage[t.second].push_back(n);
    }
  }
  return {input_usage, output_usage};
}

void
Graph::createGradients()
{
  auto mappings = tensorMappings();

  for(const auto &n : nodes_) {
    auto y = n->outputs_["y"];
    auto dy = std::make_shared<Tensor>(y->data_type_,
                                       y->dims_,
                                       y->namePostfix("grad"));
    n->inputs_["dy"] = dy;

    for(const auto &u : mappings.first[y]) {
      u->outputs_["dx"] = dy;
    }
  }
}

}
