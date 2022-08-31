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
    int batch_size = 2;
    int opt;

    srand(getpid() ^ time(NULL));

    auto dt = Tensor::DataType::FLOAT;

    while((opt = getopt(argc, argv, "b:h")) != -1) {
        switch(opt) {
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

    auto input = makeTensor(dt, Dims({DimParam::BATCH_SIZE, 4}), "input");
    auto n = g.addNode("fc", input, {{"outputs", 8}}, "fc");
    auto output1 = n->y();
    n = g.addNode("fc", n, {{"outputs", 4}, {"transW", true}}, "fc");

    auto output2 = n->y();
    auto p = ctx->createProgram({g, .batch_size = batch_size},
                                ProgramType::INFERENCE, {});

    input = ctx->resolveTensor(input);
    auto weights = ctx->resolveTensor(n->inputs_["w"]);
    output1 = ctx->resolveTensor(output1);
    output2 = ctx->resolveTensor(output2);

    auto input_ta = input->access();
    for(size_t i = 0; i < 4; i++) {
        input_ta->set(Dims{{0, (int)i}}, -0.75f + i * 0.5f);
        //        input_ta->set(Dims{{1, (int)i}}, i * 10);
    }
#if 0
    auto weights_ta = weights->access();
    for(size_t i = 0; i < 4; i++) {
        weights_ta->set(Dims{{(int)i, 0}}, 1.0f - i * 0.1f);
        weights_ta->set(Dims{{(int)i, 1}}, i * 0.1f);
    }
#endif
    p->finalize();

    p->run();

    p->dump(stdout, true);
    printf("\n");
    input->print("  INPUT");
    printf("\n");
    weights->print("WEIGHTS");
    printf("\n");
    output1->print("OUTPUT1");
    printf("\n");
    output2->print("OUTPUT2");
    printf("\n");
    output2->printStats("OUTPUT2");
    return 0;
}

SAGA_CLI_CMD("fc", "fc [OPTIONS ...]", "Run test of fc", fc_main);
