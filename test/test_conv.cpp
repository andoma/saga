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

using namespace saga;

struct TensorData {
    Dims dims;
    std::vector<float> data;
};

static void
load_tensor(TensorAccess &ta, const TensorData &td)
{
    const size_t elements = td.dims.elements();

    Dims e(td.dims.size(), 0);
    for(size_t i = 0; i < elements; i++) {
        ta.set(e, td.data[i]);

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
load_tensor(Tensor::DataType dt, const TensorData &td)
{
    auto t = makeCPUTensor(dt, td.dims);
    auto dst = t->access();
    load_tensor(*dst, td);
    return t;
}

// clang-format off

static const TensorData conv_input_x = {
    {2, 1, 5, 5},
    {
        1, 0, 0, 0, 0,
        0, 1, 0, 0, 0,
        0, 0, 1, 0, 0,
        0, 0, 0, 1, 0,
        0, 0, 0, 0, 1,

        0, 0, 0, 0, 1,
        0, 0, 0, 1, 0,
        0, 0, 1, 0, 0,
        0, 1, 0, 0, 0,
        1, 0, 0, 0, 0,

    }
};

// clang-format on

static int
conv_main(int argc, char **argv)
{
    int verbose = 0;
    int opt;
    auto dt = Tensor::DataType::FLOAT;
    int batch_size = 1;

    while((opt = getopt(argc, argv, "hv")) != -1) {
        switch(opt) {
        case 'h':
            dt = Tensor::DataType::HALF;
            break;
        case 'v':
            verbose++;
            break;
        }
    }

    argc -= optind;
    argv += optind;

    auto ctx = createContext();

    Graph g;

    auto x = load_tensor(dt, conv_input_x);

    auto n = g.addNode("conv", x,
                       {{"transpose", false},
                        {"bias", true},
                        {"stride", 2},
                        {"pad", 0},
                        {"size", 3},
                        {"activations", 5}},
                       "node");

    std::shared_ptr<saga::Tensor> mid;

    n = g.addNode("relu", n->y());

    n = g.addNode("fc", n->y(),
                  {{"outputs", 20}, {"bias", true}, {"transW", true}}, "fc1");

    mid = n->y();

    n = g.addNode("relu", n->y());

    n = g.addNode("reshape", n->y(), {{"shape", saga::Dims({2, 5, 2, 2})}});

    n = g.addNode("conv", n->y(),
                  {{"transpose", true},
                   {"bias", true},
                   {"stride", 2},
                   {"pad", 0},
                   {"size", 3},
                   {"activations", 1}},
                  "node");
    auto last = n->y();

    auto target = saga::makeTensor(dt, last->m_dims);

    n = g.addNode("loss", {{"x", n->y()}, {"target", target}});

    auto out = n->y();

    if(verbose)
        g.print();

    auto p = ctx->createProgram(
        {
            .graph = g,
            .batch_size = batch_size,
        },
        ProgramType::TRAINING, {});

    auto mmss = ctx->resolveTensor(n->m_outputs["mmss"]);
    target = ctx->resolveTensor(target);
    mid = ctx->resolveTensor(mid);
    out = ctx->resolveTensor(out);
    auto bp = ctx->resolveTensor(last->grad());

    if(verbose)
        p->dump(stdout, verbose > 1);

    int i = 0;
    while(1) {
        {
            auto target_ta = target->access();
            load_tensor(*target_ta, conv_input_x);
        }
        if(p->run() != ExecResult::OK)
            break;

        if((i % 1000) == 0) {
            bp->print("last");
            out->print("out");
            mmss->print("MMSS");
        }
        i++;
    }

    return 0;
}

SAGA_CLI_CMD("conv", "conv [OPTIONS ...]", "Run test of convolutions",
             conv_main);
