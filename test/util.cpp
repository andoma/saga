#include <stdio.h>

#include "saga.hpp"

using namespace saga;

extern int
util_showtensors(int argc, char **argv)
{
    if(argc < 1) {
        fprintf(stderr, "usage: <path>\n");
        return 1;
    }

    Network net(false);
    net.loadTensors(argv[0]);

    for(const auto &it : net.named_tensors_) {
        it.second->printStats(it.first.c_str());
    }
    return 0;
}
