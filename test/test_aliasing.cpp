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

extern int
aliasing_main(int argc, char **argv)
{
    int opt;

    auto dt = Tensor::DataType::FLOAT;

    while((opt = getopt(argc, argv, "h")) != -1) {
        switch(opt) {
        case 'h':
            dt = Tensor::DataType::HALF;
            break;
        }
    }

    argc -= optind;
    argv += optind;

    Graph g;

    auto x = makeTensor(dt, Dims({1, 1, 4, 4}), "input");

    auto w0 = g.addNode("window", {{"x", x}},
                        {{"shape", std::vector<int>{1, 1, 2, 4}},
                         {"offset", std::vector<int>{0, 0, 0, 0}}});
    auto w1 = g.addNode("window", {{"x", x}},
                        {{"shape", std::vector<int>{1, 1, 2, 4}},
                         {"offset", std::vector<int>{0, 0, 2, 0}}});

    auto r0 = g.addNode("relu", {{"x", w0->y()}}, {});
    auto r1 = g.addNode("relu", {{"x", w1->y()}}, {});

    auto output =
        g.addNode("concat", {{"x0", r0->y()}, {"x1", r1->y()}}, {{"axis", 2}});

    g.print();

    auto y = output->y();

    auto ctx = createContext();
    auto p = ctx->createProgram(g,
                                {.inference = true,
                                 .training = true,
                                 .batch_size = 1,
                                 .initial_learning_rate = 1e-3,
                                 .tensor_layout = TensorLayout::NCHW},
                                {});

    p->dump(stdout, true);

    auto dx = p->resolveTensorGradient(x);
    x = p->resolveTensor(x);
    auto dy = p->resolveTensorGradient(y);
    y = p->resolveTensor(y);

    for(int i = 0; i < 4; i++) {
        for(int j = 0; j < 4; j++) {
            x->access()->set({0, 0, i, j}, i * 4 + j);
        }
    }

    for(int i = 0; i < 4; i++) {
        for(int j = 0; j < 4; j++) {
            dy->access()->set({0, 0, i, j}, 0.1 * (i * 4 + j));
        }
    }

    x->print("x");

    p->train(1);

    y->print("y");
    dy->print("dy");
    dx->print("dx");

    p->dump(stdout, true);

    return 0;
}

SAGA_CLI_CMD("aliasing", "aliasing", "Run small aliasing test", aliasing_main);
