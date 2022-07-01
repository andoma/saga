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
#include <unordered_set>
#include <optional>
#include <string>

#include <assert.h>

namespace saga {

//------------------------------------------------------------------------
//------------------------------------------------------------------------

enum class DimParam {
    BATCH_SIZE,
    UNCHANGED,
    REDUCE,
};

struct Dim : public std::variant<int64_t, DimParam> {
    using std::variant<int64_t, DimParam>::variant;

    std::string to_string() const;

    operator int() const
    {
        if(auto v = std::get_if<int64_t>(&*this)) {
            return *v;
        } else {
            fprintf(stderr, "Unable to convert parameterized dim to int\n");
            abort();
        }
    }
    Dim &operator++();
    Dim &operator+=(const Dim &);
};

class Dims : public std::vector<Dim> {
    using std::vector<Dim>::vector;

public:
    Dims(const std::vector<Dim> &v) : std::vector<Dim>(v) {}
    Dims(const std::vector<int> &v);

    Dims n(int64_t v) const;
    std::vector<int64_t> i64() const;
    std::vector<int32_t> i32() const;
    size_t elements(size_t from_rank = 0) const;
    std::string to_string() const;
    bool similar(const Dims &) const;

    Dims transform(std::function<Dim(const DimParam &dp, size_t i)> fn) const;
};

//------------------------------------------------------------------------
//------------------------------------------------------------------------

struct Attribute
  : public std::variant<float, int, std::vector<int>, bool, Dims> {
    using std::variant<float, int, std::vector<int>, bool, Dims>::variant;

    std::string to_string() const;
};

class Attributes : public std::unordered_map<std::string, Attribute> {
public:
    using std::unordered_map<std::string, Attribute>::unordered_map;

    template <typename T>
    T get(const std::string &n, T def) const
    {
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

enum class TensorLayout {
    Auto,
    NHWC,
    NCHW,
};

class Tensors;

class TensorAccess {
protected:
    TensorAccess(){};

public:
    virtual ~TensorAccess(){};
    virtual Dims strides() = 0;
    virtual void *data() = 0;

    virtual void copyBytesFrom(const Dims &element, const void *data,
                               size_t size) = 0;

    virtual void *getAddr(const Dims &element) { return nullptr; }
    virtual double get(const Dims &element) = 0;
    virtual void set(const Dims &element, double value) = 0;

    TensorAccess(TensorAccess const &) = delete;
    TensorAccess &operator=(TensorAccess const &) = delete;
};

class Tensor {
public:
    Tensor &operator=(Tensor const &) = delete;
    Tensor(Tensor const &) = delete;

    struct Stats {
        double min;
        double max;
        double mean;
        double stddev;
    };

    enum class DataType {
        U8,
        HALF,
        FLOAT,
        INT64,
        I32,
        I16,
    };

    static size_t DataTypeSize(DataType dt);

    static const char *DataTypeStr(DataType dt);

    Tensor(DataType data_type, const Dims &size,
           const std::optional<const std::string> &name = std::nullopt);

    virtual ~Tensor(){};

    virtual std::string info() const;

    // Make this const?
    virtual std::unique_ptr<TensorAccess> access() { return nullptr; }

    virtual std::shared_ptr<Tensor> slice(const Dims &offset, const Dims &size)
    {
        return nullptr;
    }

    virtual void copyFrom(Tensor &t);

    double sse(Tensor &t);

    static std::shared_ptr<Tensor> loadProtoBuf(const char *path);

    static std::shared_ptr<Tensor> load(
        const char *path, const std::optional<const std::string> &name);
    bool save(const char *path);

    static std::shared_ptr<Tensor> find(
        Tensor::DataType data_type, const Dims &size, double init_mean,
        double init_stddev, Tensors &named_tensors,
        const std::optional<const std::string> &name = std::nullopt);

    static std::shared_ptr<Tensor> make(Tensor::DataType data_type,
                                        const Dims &size, double init_mean,
                                        double init_stddev);

    // Info / Debug / etc
    std::shared_ptr<Tensor> toRGB(
        std::optional<std::pair<float, float>> range = std::nullopt);
    void print(const char *prefix, int elements_per_rank = 0);
    void printRGB(const char *prefix,
                  std::optional<std::pair<float, float>> range = std::nullopt);
    virtual Stats stats();
    void printStats(const char *prefix);
    std::string statsString(void);

    std::optional<const std::string> namePostfix(
        const std::string &postfix) const;

    const std::optional<const std::string> name_;
    const DataType data_type_;
    const Dims dims_;
};

std::shared_ptr<Tensor> makeCPUTensor(
    Tensor::DataType data_type, const Dims &size,
    const std::optional<const std::string> &name = std::nullopt);

class Tensors
  : public std::unordered_map<std::string, std::shared_ptr<Tensor>> {
public:
    using std::unordered_map<std::string,
                             std::shared_ptr<Tensor>>::unordered_map;

    std::shared_ptr<Tensor> get(const std::string &n) const
    {
        auto it = find(n);
        return it == end() ? nullptr : it->second;
    }

    std::vector<std::shared_ptr<Tensor>> getv(const std::string &n) const
    {
        std::vector<std::shared_ptr<Tensor>> v;
        for(int i = 0;; i++) {
            auto it = find(n + std::to_string(i));
            if(it == end())
                break;
            v.push_back(it->second);
        }
        return v;
    }
};

//------------------------------------------------------------------------
//------------------------------------------------------------------------

typedef std::function<size_t(long batch, int n, uint8_t *data, size_t capacity)>
    Loader;

class Node {
public:
    Node(const std::string &type,
         const std::optional<const std::string> &name = std::nullopt)
      : type_(type), name_(name)
    {
    }

    Node(const Node &n)
      : type_(n.type_)
      , name_(n.name_)
      , inputs_(n.inputs_)
      , attributes_(n.attributes_)
      , outputs_(n.outputs_)
    {
    }

    const std::string type_;
    const std::optional<const std::string> name_;

    Tensors inputs_;
    Attributes attributes_;
    Tensors outputs_;
    Loader loader_;

    std::shared_ptr<Tensor> inferTensor_y(
        const std::optional<const std::string> &name = std::nullopt);
    void print() const;

    static std::vector<std::shared_ptr<Node>> make(
        const std::string &type, const Tensors &inputs,
        const Attributes &attributes, Tensors &named_tensors,
        const std::optional<const std::string> &name = std::nullopt);

    static std::vector<std::shared_ptr<Node>> make(
        const std::string &type, Loader loader, const Attributes &attributes);

    std::shared_ptr<Tensor> y();
};

class Nodes : public std::vector<std::shared_ptr<Node>> {
public:
    using std::vector<std::shared_ptr<Node>>::vector;

    iterator findSingleDownStreamNode(std::shared_ptr<Tensor> t,
                                      const std::string &type);
};

//------------------------------------------------------------------------
//------------------------------------------------------------------------

class Program;

typedef std::unordered_map<
    std::shared_ptr<Tensor>,
    std::vector<std::pair<std::string, std::shared_ptr<Node>>>>
    TensorMapping;

class Graph {
public:
    Nodes nodes_;
    std::unordered_set<std::shared_ptr<Tensor>> inputs_;
    std::unordered_set<std::shared_ptr<Tensor>> outputs_;
    Tensors tensors_;

    std::shared_ptr<Node> addNode(
        const std::string &type, const Tensors &inputs,
        const Attributes &attributes,
        const std::optional<const std::string> &name = std::nullopt);

    std::shared_ptr<Node> addNode(const std::string &type, Loader loader,
                                  const Attributes &attributes);

    static std::shared_ptr<Graph> load(const char *path);

    void loadTensors(const char *path);

    bool saveTensors(const char *path, Program *p);

    void print() const;

    std::pair<TensorMapping, TensorMapping> tensorMappings() const;

    std::unordered_set<std::shared_ptr<Tensor>> inputTensors() const;

    std::unordered_set<std::shared_ptr<Tensor>> outputTensors() const;

    // Build helpers

    std::shared_ptr<Node> addJpegDecoder(int width, int height,
                                         Tensor::DataType output_data_type,
                                         Loader loader);

    std::shared_ptr<Node> addConvert(std::shared_ptr<Tensor> input,
                                     Tensor::DataType dt, float scale);

    std::shared_ptr<Node> addSpatialTransform(
        std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> theta,
        int output_width = -1, int output_height = -1,
        bool also_during_inference = false);

    std::shared_ptr<Node> addResNetBottleNeck(std::shared_ptr<Tensor> input,
                                              int squeeze_activations,
                                              int output_activations,
                                              bool downsample,
                                              const std::string &name);

    std::shared_ptr<Node> addResNet(std::shared_ptr<Tensor> input,
                                    int output_activations, bool downsample,
                                    const std::string &name);
};

//------------------------------------------------------------------------
//------------------------------------------------------------------------

struct UI {
    virtual void updateBatchInfo(const char *what, int batch_size,
                                 int total_batches) = 0;
    virtual void updateProgress(int current_batch) = 0;
    virtual void updateLoss(double loss) = 0;
    virtual void updateMemUsage(size_t use, size_t total) = 0;
    virtual void updateMpScaling(double scaling) = 0;
};

std::shared_ptr<UI> make_statbar();

//------------------------------------------------------------------------
//------------------------------------------------------------------------

enum class BTM { PRE = 0x0, POST = 0x1, VALUE = 0x0, GRADIENT = 0x2 };

inline constexpr BTM
operator|(BTM a, BTM b)
{
    return static_cast<BTM>(static_cast<int>(a) | static_cast<int>(b));
}

inline constexpr BTM
operator&(BTM a, BTM b)
{
    return static_cast<BTM>(static_cast<int>(a) & static_cast<int>(b));
}

inline constexpr bool
operator!(BTM a)
{
    return static_cast<bool>(!static_cast<int>(a));
}

typedef std::pair<std::shared_ptr<Tensor>, BTM> BatchedTensor;

typedef std::unordered_set<BatchedTensor> BatchedTensors;

typedef std::function<void(long batch, bool training,
                           std::unordered_map<BatchedTensor, TensorAccess *>)>
    TensorBatchCallback;

typedef std::function<bool(void)> StopCheck;

struct ProgramConfig {
    bool inference{true};
    bool training{false};
    int batch_size{1};
    float initial_learning_rate{0};
    TensorLayout tensor_layout{TensorLayout::Auto};
    StopCheck stop_check = nullptr;
    std::shared_ptr<UI> ui;
};

enum class ExecResult {
    OK,
    STOPPED,
    ERROR,
};

class Program {
public:
    virtual ~Program() {}
    virtual std::shared_ptr<Tensor> resolveTensor(
        std::shared_ptr<Tensor> t) = 0;
    virtual std::shared_ptr<Tensor> resolveTensorGradient(
        std::shared_ptr<Tensor> t) = 0;
    virtual ExecResult infer(long batches = 1,
                             const TensorBatchCallback &pre = nullptr,
                             const TensorBatchCallback &post = nullptr) = 0;
    virtual ExecResult train(long batches = 1,
                             const TensorBatchCallback &pre = nullptr,
                             const TensorBatchCallback &post = nullptr) = 0;
    virtual void print(bool detailed = false) const = 0;
    virtual void debug(bool on) = 0;
    virtual bool dumpGraph(const char *path) { return false; }
};

//------------------------------------------------------------------------
//------------------------------------------------------------------------

class Context {
public:
    virtual ~Context() {}
    virtual std::shared_ptr<Program> createProgram(
        const Graph &graph, const ProgramConfig &pc,
        const BatchedTensors &bts) = 0;

    virtual std::string info() const = 0;

    virtual void print() const = 0;
};

std::shared_ptr<Context> createContext();

std::vector<std::shared_ptr<Context>> createContexts();

int64_t Now();

};  // namespace saga

namespace std {

template <>
struct hash<saga::BatchedTensor> {
    std::size_t operator()(const saga::BatchedTensor &k) const
    {
        return std::hash<std::shared_ptr<saga::Tensor>>()(k.first) ^
               std::hash<saga::BTM>()(k.second);
    }
};

}  // namespace std
