#include <assert.h>
#include <unistd.h>
#include <sys/stat.h>
#include <fcntl.h>

#include <algorithm>

#include "saga.hpp"

using namespace saga;

int
test_concat_main(int argc, char **argv)
{
    int batch_size = 2;

    Network net(true);

    std::vector<float> test(24 + 14);
    for(int i = 0; i < 24; i++) test[i] = i;

    Tensor i1(Size(batch_size, 3, 2, 2), Tensor::Type::FLOAT);

    auto c1 = net.addLayer(makeInput(&i1, true));
    i1.load(test);
    c1->output()->dump("c1");

    Tensor i2(Size(batch_size, 2, 2, 2), Tensor::Type::FLOAT);

    for(int i = 0; i < 24; i++) test[i] = 100 + i;

    i2.load(test);
    auto c2 = net.addLayer(makeInput(&i2, true));
    c2->output()->dump("c2");

    auto tail = net.addLayer(makeConcat({c1.get(), c2.get()}, net));

    net.forward(true);

    tail->output()->dump("output");

    for(int i = 0; i < 24 + 16; i++) test[i] = 1000 + i;

    tail->gradient()->load(test);
    tail->gradient()->dump("g");

    net.backprop(0);

    c1->gradient()->dump("c1g");
    c2->gradient()->dump("c2g");

    return 0;
}
