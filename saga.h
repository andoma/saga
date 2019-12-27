// -*-c++-*-

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

#pragma once

#include <memory>
#include <vector>
#include <unordered_map>
#include <functional>
#include <variant>

#include <assert.h>
#include <cudnn.h>
#include <cublas_v2.h>


namespace saga {

enum class ActivationMode {
  RELU,
  ELU
};


typedef std::variant<float, int, std::vector<int>> Attribute;

class Attributes : public std::unordered_map<std::string, Attribute> {
public:
  template< typename T > T get(const std::string &n, T def) const {
    auto it = find(n);
    if(it == end())
      return def;
    auto p = std::get_if<T>(&it->second);
    if(p == NULL)
      return def;
    return *p;
  }
};



class Tensor {

public:
  std::vector<unsigned int> dims_;
  std::string name_;
};



class Node {
public:
  enum class Type {
    RESHAPE,
    CONV,
    BATCHNORM,
    RELU,
    SOFTMAX,
    MAXPOOL,
    AVGPOOL,
    CONCAT,
    SUM,
    DROPOUT,
    FC
  };

  Node(Type t) : type_(t) {};

  std::unordered_map<std::string, std::shared_ptr<Tensor>> inputs_;
  std::unordered_map<std::string, std::shared_ptr<Tensor>> outputs_;
  Attributes attributes_;
  Type type_;
};



class Graph {
public:
  std::vector<std::shared_ptr<Node>> nodes_;
  std::vector<std::shared_ptr<Tensor>> inputs_;
  std::vector<std::shared_ptr<Tensor>> outputs_;
  std::unordered_map<std::string, std::shared_ptr<Tensor>> tensors_;

  static std::shared_ptr<Graph> load(const char *path);

};

class Operation {

};

}
