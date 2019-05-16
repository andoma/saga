#include <assert.h>
#include <unistd.h>
#include <sys/stat.h>
#include <fcntl.h>

#include <algorithm>

#include "saga.h"


using namespace saga;



static std::vector<uint8_t>
load(const std::string &path)
{
  int fd = open(path.c_str(), O_RDONLY);
  if(fd == -1) {
    fprintf(stderr, "Unable to open %s -- %s\n", path.c_str(), strerror(errno));
    exit(1);
  }

  struct stat st;
  if(fstat(fd, &st)) {
    fprintf(stderr, "Unable to stat %s -- %s\n", path.c_str(), strerror(errno));
    exit(1);
  }

  std::vector<uint8_t> mem(st.st_size);

  if(read(fd, (void *)&mem[0], st.st_size) != st.st_size) {
    fprintf(stderr, "Unable to read %s\n", path.c_str());
    exit(1);
  }

  close(fd);
  return mem;
}




static uint32_t
rd32(const uint8_t *d)
{
  return (d[0] << 24) | (d[1] << 16) | (d[2] << 8) | d[3];
}


struct LabeledImage {
  LabeledImage(const uint8_t *image, unsigned int label)
    : image(image)
    , label(label)
  {}
  const uint8_t *image;
  unsigned int label;
};


std::vector<LabeledImage>
makeLabeledImages(const uint8_t *images,
                  const uint8_t *labels,
                  size_t image_step,
                  size_t count)
{
  std::vector<LabeledImage> lis;
  lis.reserve(count);

  for(size_t i = 0; i < count; i++) {
    lis.push_back(LabeledImage(images, *labels));
    images += image_step;
    labels++;
  }
  return lis;
}


static void
loadInputTensor(Tensor &t, const LabeledImage *lis)
{
  const size_t batch_size = t.n;

  const uint8_t *images[batch_size];
  for(size_t j = 0; j < batch_size; j++) {
    images[j] = lis[j].image;
  }
  t.load(images);
}


static void
loadOutputTensor(Tensor &t, const LabeledImage *lis)
{
  const size_t n = t.n;
  const size_t c = t.c;

  float values[n * c];

  memset(values, 0, sizeof(float) * n * c);

  for(size_t j = 0; j < n; j++) {
    values[c * j + lis[j].label] = 1.0f;
  }
  t.load(values);
}





extern int
mnist_main(int argc, char **argv)
{
  if(argc < 2) {
    fprintf(stderr, "mnist usage: <path> <batch_size>\n");
    return 1;
  }


  std::string path(argv[0]);
  size_t batch_size = atoi(argv[1]);

  if(batch_size < 1) {
    fprintf(stderr, "Bad batch_size: %s\n", argv[1]);
    return 1;
  }

  const int labels = 10;

  auto train_image_data = load(path + "/train-images-idx3-ubyte");
  auto train_label_data = load(path + "/train-labels-idx1-ubyte");

  auto  test_image_data = load(path + "/t10k-images-idx3-ubyte");
  auto  test_label_data = load(path + "/t10k-labels-idx1-ubyte");

  const int train_images = rd32(&train_image_data[4]);
  const int test_images  = rd32(&test_image_data[4]);

  const int rows = rd32(&train_image_data[8]);
  const int cols = rd32(&train_image_data[12]);
  printf("data: %d x %d\n", cols, rows);
  assert(cols == 28);
  assert(rows == 28);

  const size_t train_inputs = (train_images / batch_size) * batch_size;
  const size_t test_inputs  = (test_images  / batch_size) * batch_size;

  printf("Training inputs: %zd  Test inputs: %zd\n",
         train_inputs, test_inputs);

  auto train_data = makeLabeledImages(&train_image_data[16],
                                      &train_label_data[8],
                                      cols * rows,
                                      train_inputs);

  auto test_data = makeLabeledImages(&test_image_data[16],
                                     &test_label_data[8],
                                     cols * rows,
                                     test_inputs);


  Network net(batch_size, true);

  net.setOptimizer(&makeAdamOptimizer);

  Tensor input(TensorDescriptor(CUDNN_DATA_FLOAT,
                                Size(batch_size, 1, 28, 28)));

  const Tensor *tail = &input;

  tail = net.addLayer(makeConvolution(32, 5, 1, 0, *tail, {}, net));
  tail = net.addLayer(makeActivation(ActivationMode::RELU, 0, *tail, net));
  tail = net.addLayer(makePooling(PoolingMode::MAX, 2, 2, *tail, net));

  tail = net.addLayer(makeConvolution(64, 5, 1, 0, *tail, {}, net));
  tail = net.addLayer(makeActivation(ActivationMode::RELU, 0, *tail, net));
  tail = net.addLayer(makePooling(PoolingMode::MAX, 2, 2, *tail, net));

  tail = net.addLayer(makeFullyConnected(1024, *tail, {}, net));
  tail = net.addLayer(makeActivation(ActivationMode::RELU, 0, *tail, net));

  tail = net.addLayer(makeFullyConnected(labels, *tail, {}, net));
  tail = net.addLayer(makeSoftmax(*tail, net));

  Tensor dy{TensorDescriptor(*tail)};

  printf("%d vs %zd\n", dy.n, batch_size);
  assert(dy.n == batch_size);
  assert(dy.c == labels);

  unsigned int iteration = 0;
  while(1) {
    std::random_shuffle(train_data.begin(), train_data.end());

    //    const float learning_rate = 0.01 * powf(1.0 + 0.01 * iteration, -0.75);

    // Train
    for(size_t i = 0; i < train_inputs; i += batch_size) {
      loadInputTensor(input, &train_data[i]);
      net.forward(&input, false);
      loadOutputTensor(dy, &train_data[i]);
      net.backprop(&input, &dy, iteration);
      iteration++;
    }

    // Test
    int correct = 0;
    for(size_t i = 0; i < test_inputs; i += batch_size) {
      loadInputTensor(input, &test_data[i]);
      net.forward(&input, true);

      float result[batch_size * labels];
      tail->save(result);
      for(size_t n = 0; n < batch_size; n++) {
        unsigned int label = 0;
        for(int c = 1; c < labels; c++) {
          if(result[n * labels + c] > result[n * labels + label]) {
            label = c;
          }
        }
        if(label == test_data[i + n].label)
          correct++;
      }
    }
    printf("test accuracy: %3.3f%%\n", 100.0 * correct / test_inputs);
  }

  return 0;
}
