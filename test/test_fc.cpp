#include <assert.h>
#include <unistd.h>
#include <sys/stat.h>
#include <fcntl.h>

#include <algorithm>

#include "saga.hpp"
#include "cli.h"

using namespace saga;

static int
fc_main(int argc, char **argv)
{
    int opt;

    int inputs = 4;
    int outputs = 8;
    int batch_size = 2;
    bool transposed = false;

    srand(getpid() ^ time(NULL));

    auto dt = Tensor::DataType::FLOAT;

    while((opt = getopt(argc, argv, "i:o:tb:h")) != -1) {
        switch(opt) {
        case 'i':
            inputs = atoi(optarg);
            break;
        case 'o':
            outputs = atoi(optarg);
            break;
        case 't':
            transposed = true;
            break;
        case 'b':
            batch_size = atoi(optarg);
            break;
        case 'h':
            dt = Tensor::DataType::HALF;
            break;
        }
    }

    auto ctx = createContext();

    Graph g;

    auto input = makeTensor(dt, Dims({DimParam::BATCH_SIZE, inputs}), "input");

    std::shared_ptr<Tensor> weights;

    if(transposed) {
        weights = makeTensor(dt, Dims({outputs, inputs}), "weights");
    } else {
        weights = makeTensor(dt, Dims({inputs, outputs}), "weights");
    }

    auto target =
        makeTensor(dt, Dims({DimParam::BATCH_SIZE, outputs}), "target");

    auto n = g.addNode("fc", {{"x", input}, {"w", weights}},
                       {{"outputs", outputs}, {"transW", transposed}}, "fc");

    auto output = n->y();

    auto l = g.addNode("loss", {{"x", n->y()}, {"target", target}});
    auto result = l->y();

    auto p = ctx->createProgram({.graph = g, .batch_size = batch_size},
                                ProgramType::TRAINING, {.learning_rate = 0.1});

    auto activations = ctx->resolveTensor(input->grad());
    input = ctx->resolveTensor(input);
    weights = ctx->resolveTensor(weights);
    result = ctx->resolveTensor(result);
    target = ctx->resolveTensor(target);
    auto grad = ctx->resolveTensor(output->grad());
    output = ctx->resolveTensor(output);
    auto mmss = ctx->resolveTensor(l->m_outputs["mmss"]);

    auto input_ta = input->access();
    for(int n = 0; n < batch_size; n++) {
        for(int i = 0; i < inputs; i++) {
            input_ta->set(Dims{{n, i}}, 1 + n * 10 + i);
        }
    }

    auto weights_ta = weights->access();
    for(int i = 0; i < inputs; i++) {
        if(transposed) {
            weights_ta->set(Dims{{0, i}}, i);
        } else {
            weights_ta->set(Dims{{i, 0}}, i);
        }
    }

    auto target_ta = target->access();
    for(int n = 0; n < batch_size; n++) {
#if 0
        for(int i = 0; i < outputs; i++) {
            target_ta->set(Dims{{n, i}}, 1 + (10 - n) * 10 + (10 - i));
        }
#endif
        int i = 1;
        target_ta->set(Dims{{n, i}}, n + 1);
    }

    p->finalize();

    p->dump(stdout, true);

    weights->print("WEIGHTS");
    printf("\n");

    p->run();

    printf("\n");
    input->print("  INPUT");
    printf("\n");
    weights->print("WEIGHTS");
    printf("\n");
    result->print("RESULT");
    printf("\n");
    target->print("TARGET");
    printf("\n");
    grad->print("GRAD");
    printf("\n");
    activations->print("ACTIVATIONS");
    printf("\n");
    //    mmss->print("MMSS");
    //    printf("\n");
    return 0;
}

SAGA_CLI_CMD("fc", "fc [OPTIONS ...]", "Run test of fc", fc_main);
