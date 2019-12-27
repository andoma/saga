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


//------------------------------------------------------------------------
//------------------------------------------------------------------------

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


//------------------------------------------------------------------------
//------------------------------------------------------------------------

typedef std::vector<int64_t> Dims;

class TensorAccess {

protected:
  TensorAccess() {};
public:
  virtual ~TensorAccess() {};
  virtual Dims strides() = 0;
  virtual void *data() = 0;

  virtual double get(const std::vector<int64_t> &element) const = 0;
  virtual void set(const std::vector<int64_t> &element, double value) = 0;

  TensorAccess(TensorAccess const&) = delete;
  TensorAccess& operator=(TensorAccess const&) = delete;
};


class Tensor {

public:

  enum class DataType {
    U8,
    HALF,
    FLOAT,
    INT64,
  };


  Tensor(const std::string &name, DataType data_type, Dims dims)
    : name_(name)
    , data_type_(data_type)
    , dims_(dims)
  {};

  virtual ~Tensor() {};

  virtual std::string info() const;

  virtual std::unique_ptr<TensorAccess> access();

  void print(const char *prefix);

  const std::string name_;
  const DataType data_type_;
  const Dims dims_;
};


std::shared_ptr<Tensor> makeCPUTensor(const std::string &name, Tensor::DataType data_type,
                                      Dims dims);

class Tensors : public std::unordered_map<std::string,
                                          std::shared_ptr<Tensor>> {
public:
  std::shared_ptr<Tensor> get(const std::string &n) const {
    auto it = find(n);
    return it == end() ? nullptr : it->second;
  }
};

//------------------------------------------------------------------------
//------------------------------------------------------------------------

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

  Tensors inputs_;
  Tensors outputs_;
  Attributes attributes_;
  Type type_;

  std::shared_ptr<Tensor> makeOutputTensor(const std::string &name);
};


//------------------------------------------------------------------------
//------------------------------------------------------------------------

class Graph {
public:
  std::vector<std::shared_ptr<Node>> nodes_;
  std::vector<std::shared_ptr<Tensor>> inputs_;
  std::vector<std::shared_ptr<Tensor>> outputs_;
  Tensors tensors_;

  static std::shared_ptr<Graph> load(const char *path);

  std::shared_ptr<Tensor> loadTensor(const char *path);

  void resolve();

};

class Operation {

};

}
