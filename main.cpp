#include "saga.h"

using namespace saga;


void
mnist_train(const char *path)
{
  int batch_size = 32;

  auto input = std::make_shared<Tensor>(CUDNN_DATA_FLOAT,
                                        Size(batch_size, 1, 28, 28));
  Network net(*input, false);

  auto tail = input;

  tail = net.addLayer(makeConvolution(32, 5, 1, 0, tail, {}, net));
  tail = net.addLayer(makeActivation(ActivationMode::RELU, 0, tail, net));
  tail = net.addLayer(makePooling(PoolingMode::MAX, 2, 2, tail, net));

  tail = net.addLayer(makeConvolution(64, 5, 1, 0, tail, {}, net));
  tail = net.addLayer(makeActivation(ActivationMode::RELU, 0, tail, net));
  tail = net.addLayer(makePooling(PoolingMode::MAX, 2, 2, tail, net));

  tail = net.addLayer(makeFullyConnected(1024, tail, {}, net));
  tail = net.addLayer(makeActivation(ActivationMode::RELU, 0, tail, net));

  tail = net.addLayer(makeFullyConnected(10, tail, {}, net));
  tail = net.addLayer(makeSoftmax(tail, net));

  net.initialize();

}







int
main(int argc, char **argv)
{
  mnist_train("/tmp");
}
