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

#include <thread>

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

#define BATCH_SIZE 1

// clang-format off

static const TensorData input_data = {
    {BATCH_SIZE, 1},
    {
        0.1,
        0.1

    }
};

static TensorData target_data = {
    {BATCH_SIZE, 1},
    {
        2e4,
        4e4
    }
};


static const TensorData weights_data = {
    {1, 1},
    {
        0.1,
    }
};

// clang-format on

static int
mp_main(int argc, char **argv)
{
    int verbose = 0;
    int opt;
    auto dt = Tensor::DataType::HALF;
    int batch_size = 1;
    bool multi = false;
    while((opt = getopt(argc, argv, "vmt:")) != -1) {
        switch(opt) {
        case 'v':
            verbose++;
            break;
        case 'm':
            multi = true;
            break;
        case 't':
            target_data.data[0] = atoi(optarg);
            break;
        }
    }

    argc -= optind;
    argv += optind;

    auto engine = createEngine(saga::make_nui());
    auto contexts = engine->createContexts(multi);

    auto ctx = createContext();

    Graph g;

    auto x = load_tensor(dt, input_data);
    auto w = load_tensor(dt, weights_data);

    auto n = g.addNode("fc", {{"x", x}, {"w", w}},
                       {{"outputs", 1}, {"bias", false}}, "fc1");
    auto y = n->y();

    auto target = load_tensor(dt, target_data);

    n = g.addNode("loss", {{"x", n->y()}, {"target", target}});

    if(verbose)
        g.print();

    printf("Target: %f\n", target_data.data[0]);

    auto barrier = saga::Barrier::make(contexts.size());
    std::vector<std::thread> threads;
    for(size_t thread_index = 0; thread_index < contexts.size();
        thread_index++) {
        threads.push_back(std::thread([=, &barrier] {
            auto &ctx = contexts[thread_index];

            auto p = ctx->createProgram(
                {
                    .graph = g,
                    .batch_size = batch_size,
                },
                ProgramType::TRAINING, {.learning_rate = 0.01});

            auto mmss = ctx->resolveTensor(n->m_outputs["mmss"]);
            auto out = ctx->resolveTensor(n->y());
            auto dx = ctx->resolveTensor(x->grad());
            auto dy = ctx->resolveTensor(y->grad());
            auto wl = ctx->resolveTensor(w);

            if(verbose)
                p->dump(stdout, verbose > 1);

            barrier->wait();
            p->run();
            barrier->wait();

            out->print(" y");
            dy->print("dy");
            dx->print("dx");
            wl->print(" w");
        }));
    }

    for(auto &t : threads) {
        t.join();
    }

    return 0;
}

SAGA_CLI_CMD("mp", "mp [OPTIONS ...]", "Run some tests for mixed precision",
             mp_main);
