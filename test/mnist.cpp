#include <math.h>
#include <assert.h>
#include <unistd.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <errno.h>
#include <string.h>

#include <random>
#include <algorithm>
#include <numeric>

#include "saga.h"
#include "cli.h"
#include "test_classifier.h"

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
  const int batch_size = t.dims_[0];
  const size_t size = t.dims_[2] * t.dims_[3];

  uint8_t prep[size * batch_size];
  uint8_t *dst = prep;
  for(int n = 0; n < batch_size; n++) {
    memcpy(dst, lis[n].image, size);
    dst += size;
  }

  auto ta = t.access();
  ta->copyBytesFrom({}, prep, size * batch_size);
}


static void
loadOutputTensor(Tensor &t, const LabeledImage *lis)
{
  const int batch_size = t.dims_[0];
  auto ta = t.access();

  for(int n = 0; n < batch_size; n++) {
    ta->set({n, 0}, lis[n].label);
  }
}





static int
mnist_main(int argc, char **argv)
{
  argv++;
  argc--;

  std::string path(argv[0]);


  auto train_image_data = load(path + "/train-images-idx3-ubyte");
  auto train_label_data = load(path + "/train-labels-idx1-ubyte");

  auto  test_image_data = load(path + "/t10k-images-idx3-ubyte");
  auto  test_label_data = load(path + "/t10k-labels-idx1-ubyte");

  const int train_images = rd32(&train_image_data[4]);
  const int test_images  = rd32(&test_image_data[4]);

  const int rows = rd32(&train_image_data[8]);
  const int cols = rd32(&train_image_data[12]);

  auto train_data = makeLabeledImages(&train_image_data[16],
                                      &train_label_data[8],
                                      cols * rows,
                                      train_images);

  auto test_data = makeLabeledImages(&test_image_data[16],
                                     &test_label_data[8],
                                     cols * rows,
                                     test_images);

  auto input = std::make_shared<Tensor>(Tensor::DataType::U8,
                                        Dims({1, 1, rows, cols}), "input");

  std::random_device rd;
  std::mt19937 rnd(rd());

  test_classifier(argc, argv, input, 255, 10,
                  train_images,
                  test_images,
                  [&](void) {
                    std::shuffle(train_data.begin(), train_data.end(), rnd);
                  },
                  [&](Tensor &x, Tensor &dy, size_t i) {
                    loadInputTensor(x, &train_data[i]);
                    loadOutputTensor(dy, &train_data[i]);
                  },
                  [&](Tensor &x, int *labels, size_t i) {
                    loadInputTensor(x, &test_data[i]);
                    for(int j = 0; j < x.dims_[0]; j++) {
                      labels[j] = test_data[i + j].label;
                    }
                  });
  return 0;
}


SAGA_CLI_CMD("mnist",
             "minst <PATH> [OPTIONS ...]",
             "Infer/Train on mnist dataset",
             mnist_main);


