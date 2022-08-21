#include <signal.h>
#include <assert.h>
#include <unistd.h>
#include <sys/stat.h>
#include <fcntl.h>

#include <algorithm>
#include <numeric>
#include <string.h>

#include "saga.hpp"
#include "cli.h"

/**
   Tests a standard resnet node with skip connection:


   -+->  [ conv ] ---> [ bn ] --> [sum] --> [relu] ->
    |                               ^
    +-------------------------------+

  including backprop

 */

using namespace saga;

struct TensorData {
    Dims dims;
    std::vector<float> data;
};

// -----------------------------------------------

static const TensorData conv_input_x = {
    {2, 4, 2, 2},
    {
        -0.391, -0.003, 0.139,  -0.086, -0.312, -0.242, 0.267,  -0.087,
        -0.098, 0.227,  0.183,  -0.253, -0.444, -0.138, -0.181, -0.467,
        0.086,  -0.130, -0.433, -0.232, -0.271, 0.463,  0.241,  -0.099,
        -0.484, -0.422, 0.027,  0.178,  -0.447, 0.494,  0.355,  0.323,
    }};

static const TensorData conv_input_w = {
    {4, 4, 1, 1},
    {
        -0.416, -0.115, 0.377,  0.226,  0.285,  -0.187, -0.229, -0.143,
        0.016,  -0.094, -0.219, -0.221, 0.375,  -0.400, 0.474,  -0.258,
        -0.043, 0.184,  -0.131, -0.084, -0.173, 0.176,  0.293,  -0.123,
        -0.154, 0.221,  -0.311, -0.392, 0.177,  -0.188, 0.182,  -0.045,
    }};

static const TensorData batchnorm_input_b = {{4},
                                             {
                                                 0.250,
                                                 0.316,
                                                 0.339,
                                                 0.017,
                                             }};

static const TensorData batchnorm_input_s = {{4},
                                             {
                                                 0.381,
                                                 0.373,
                                                 0.147,
                                                 0.076,
                                             }};

static const TensorData batchnorm_input_m = {{4},
                                             {
                                                 -0.306,
                                                 0.378,
                                                 -0.283,
                                                 0.457,
                                             }};

static const TensorData batchnorm_input_v = {{4},
                                             {
                                                 -0.043,
                                                 -0.236,
                                                 0.290,
                                                 0.329,
                                             }};

static const TensorData grad = {
    {2, 4, 2, 2},
    {
        -0.429, 0.476,  -0.062, 0.069,  0.227, 0.181, -0.413, 0.158,
        0.089,  0.453,  0.484,  0.312,  0.158, 0.216, -0.475, 0.227,
        -0.069, -0.138, 0.115,  0.010,  0.212, 0.400, 0.234,  0.023,
        -0.285, 0.447,  -0.062, -0.026, 0.131, 0.414, 0.420,  0.237,
    }};

static const TensorData expected_y_with_bypass = {
    {2, 4, 2, 2},
    {
        0.000000, 0.406095, 0.233654, 0.000000, 0.116006, 0.130703, 0.536843,
        0.564521, 0.374853, 0.534650, 0.459271, 0.246868, 0.000000, 0.000000,
        0.000000, 0.000000, 0.000000, 0.000000, 0.324272, 0.536588, 0.737216,
        0.542300, 0.000000, 0.000000, 0.094470, 0.000000, 0.195250, 0.357146,
        0.000000, 0.354826, 0.286820, 0.347170,
    }};

static const TensorData expected_dx_with_bypass = {
    {2, 4, 2, 2},
    {
        0.155229,  0.197027,  -0.342228, 0.023995,  0.197370,  -0.000986,
        -0.169423, 0.141864,  -0.056134, 0.652049,  0.604917,  0.214839,
        -0.078773, 0.070593,  0.048659,  -0.067788, -0.016880, 0.295287,
        0.129472,  0.097099,  0.247891,  0.266593,  0.021801,  0.059889,
        -0.199848, -0.220140, 0.003728,  -0.034411, 0.100566,  0.301342,
        0.464925,  0.231476,
    }};

static const TensorData expected_y_no_bypass = {
    {2, 4, 2, 2},
    {
        0.360596, 0.409180, 0.094543, 0.000000, 0.427979, 0.372559, 0.269775,
        0.651367, 0.472900, 0.307617, 0.276367, 0.499756, 0.053528, 0.114624,
        0.063843, 0.039856, 0.000000, 0.124756, 0.757324, 0.768555, 1.007812,
        0.079468, 0.000000, 0.000000, 0.578613, 0.229614, 0.168213, 0.179199,
        0.047302, 0.000000, 0.000000, 0.024170,
    }};

static const TensorData expected_dx_no_bypass = {
    {2, 4, 2, 2},
    {
        0.495605,  -0.343750, -0.405762, -0.003531, 0.043243,  -0.198608,
        0.298828,  -0.061249, -0.410645, 0.283447,  0.115479,  -0.007568,
        -0.260254, 0.119812,  0.145020,  -0.054382, -0.050140, 0.372070,
        -0.079956, 0.015427,  0.005383,  -0.161133, 0.030167,  0.043213,
        0.146118,  -0.364258, 0.141479,  0.095825,  0.112976,  -0.245850,
        0.124390,  0.058044,
    }};

// -----------------------------------------------

static void
fill_tensor(std::shared_ptr<Tensor> t, const TensorData &td)
{
    auto dst = t->access();
    const size_t elements = td.dims.elements();

    Dims e(td.dims.size(), 0);
    for(size_t i = 0; i < elements; i++) {
        dst->set(e, td.data[i]);

        for(ssize_t j = e.size() - 1; j >= 0; j--) {
            ++e[j];
            if(e[j] == td.dims[j]) {
                e[j] = 0;
            } else {
                break;
            }
        }
    }
}

static std::shared_ptr<Tensor>
create_tensor(Tensor::DataType dt, const TensorData &td)
{
    auto t = makeCPUTensor(dt, td.dims);
    fill_tensor(t, td);
    return t;
}

static int
expect(std::shared_ptr<Tensor> result, std::shared_ptr<Tensor> expect,
       const char *name)
{
    double sse = result->sse(*expect);

    if(sse > 1e-4) {
        printf("Tensor %s mismaches, SSE=%f\n", name, sse);
        result->print("RESULT");
        expect->print("EXPECT");
        return 1;
    } else {
        printf("Tensor %s OK, SSE=%f\n", name, sse);
    }
    return 0;
}

static int
runtest(std::shared_ptr<Context> ctx, Tensor::DataType dt, bool bypass,
        bool check_dx)
{
    ctx->reset();

    int batch_size = 2;

    const TensorData *expected_y =
        bypass ? &expected_y_with_bypass : &expected_y_no_bypass;

    const TensorData *expected_dx =
        bypass ? &expected_dx_with_bypass : &expected_dx_no_bypass;

    printf("\nTesting resnet node: dt:%s bypass:%s read-out-dx:%s\n",
           Tensor::DataTypeStr(dt), bypass ? "yes" : "no",
           check_dx ? "yes" : "no");

    Graph g;
    auto x = create_tensor(dt, conv_input_x);
    auto w = create_tensor(dt, conv_input_w);
    auto n = g.addNode("conv", Tensors{{"x", x}, {"w", w}},
                       {{"activations", 4}, {"size", 1}});

    auto s = create_tensor(Tensor::DataType::FLOAT, batchnorm_input_s);
    auto b = create_tensor(Tensor::DataType::FLOAT, batchnorm_input_b);
    auto m = create_tensor(Tensor::DataType::FLOAT, batchnorm_input_m);
    auto v = create_tensor(Tensor::DataType::FLOAT, batchnorm_input_v);

    n = g.addNode(
        "batchnorm",
        Tensors{{"x", n->y()}, {"s", s}, {"b", b}, {"m", m}, {"v", v}},
        {{"epsilon", 0.00001f}, {"expavgf", 0.0f}});

    if(bypass) {
        n = g.addNode("sum", Tensors{{"x0", x}, {"x1", n->y()}}, {});
    }
    n = g.addNode("relu", n->y());

    auto p = ctx->createProgram({.graph = g, .batch_size = batch_size},
                                ProgramType::TRAINING,
                                {.tensor_layout = TensorLayout::Auto});

    auto y = ctx->resolveTensor(n->y());
    auto dx = check_dx ? ctx->resolveTensor(x->grad()) : nullptr;

    fill_tensor(ctx->resolveTensor(n->y()->grad()), grad);

    p->finalize();

    p->dump(stdout, true);

    if(p->run(1) != ExecResult::OK) {
        printf("Execution failed\n");
        return 1;
    }
    if(expect(y, create_tensor(dt, *expected_y), "Y"))
        return 1;

    if(dx) {
        if(expect(dx, create_tensor(dt, *expected_dx), "DX"))
            return 1;
    }

    return 0;
}

static int
resnode_main(int argc, char **argv)
{
    auto ctx = createContext();

    if(runtest(ctx, Tensor::DataType::FLOAT, true, true))
        return 1;
    if(runtest(ctx, Tensor::DataType::FLOAT, true, false))
        return 1;
    if(runtest(ctx, Tensor::DataType::FLOAT, false, true))
        return 1;
    if(runtest(ctx, Tensor::DataType::FLOAT, false, false))
        return 1;
    if(runtest(ctx, Tensor::DataType::HALF, true, true))
        return 1;
    if(runtest(ctx, Tensor::DataType::HALF, true, false))
        return 1;
    if(runtest(ctx, Tensor::DataType::HALF, false, true))
        return 1;
    if(runtest(ctx, Tensor::DataType::HALF, false, false))
        return 1;

    return 0;
}

SAGA_CLI_CMD("resnode", "resnode [OPTIONS ...]",
             "Run test of standard ResNet node", resnode_main);
