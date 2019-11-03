#include <assert.h>
#include <unistd.h>
#include <sys/stat.h>
#include <fcntl.h>

#include <algorithm>

#include "saga.h"

using namespace saga;

int
test_fc_main(int argc, char **argv)
{
  int batch_size = 2;

  int opt;

  auto dt = Tensor::Type::FLOAT;

  while((opt = getopt(argc, argv, "b:h")) != -1) {
    switch(opt) {
    case 'b':
      batch_size = atoi(optarg);
      break;
    case 'h':
      dt = Tensor::Type::HALF;
      break;
    }
  }

  Network net(true);

  Tensor i1(Size(batch_size, 4, 1, 1), dt);



  auto input = net.addLayer(makeInput(&i1, true));

  auto w = std::make_shared<Tensor>(Size(4, 4, 1, 1), dt);
  w->allocate(CUDNN_TENSOR_NHWC);
  for(int i = 0; i < 4; i++)
    for(int j = 0; j < 4; j++)
      w->set(i, j, 0, 0, i * 10 + j);
  net.named_tensors_["w"] = w;

  auto b = std::make_shared<Tensor>(Size(1, 4, 1, 1), dt);
  b->allocate(CUDNN_TENSOR_NHWC);
  b->fill(1);
  net.named_tensors_["b"] = b;
  for(int j = 0; j < 4; j++)
    b->set(0, j, 0, 0, j * 70);

  auto l = net.addLayer(makeFullyConnected(4, *input, net, "w", "b"));

  for(int n = 0; n < batch_size; n++)
    for(int c = 0; c < 4; c++)
      i1.set(n, c, 0, 0, n * 100 + c);

  net.forward();

  input->output()->dump("input");
  w->dump("weights");
  b->dump("bias");

  l->output()->dump("output");

  auto grad = l->gradient();
  for(int n = 0; n < batch_size; n++)
    for(int c = 0; c < 4; c++)
      grad->set(n, c, 0, 0, n * 10 + c);

  net.backprop(0);

  return 0;
}
