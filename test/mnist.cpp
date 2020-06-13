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
loadInputTensor(TensorAccess &ta, const LabeledImage *lis,
                int batch_size, size_t size)
{
  uint8_t prep[size * batch_size];
  uint8_t *dst = prep;
  for(int n = 0; n < batch_size; n++) {
    memcpy(dst, lis[n].image, size);
    dst += size;
  }

  ta.copyBytesFrom({}, prep, size * batch_size);
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

  const size_t pixels = rows * cols;

  auto train_data = makeLabeledImages(&train_image_data[16],
                                      &train_label_data[8],
                                      cols * rows,
                                      train_images);

  auto test_data = makeLabeledImages(&test_image_data[16],
                                     &test_label_data[8],
                                     cols * rows,
                                     test_images);

  std::random_device rd;
  std::mt19937 rnd(rd());

  int batch_size = 0;
  bool test = false;

  auto input = std::make_shared<Tensor>(Tensor::DataType::U8,
                                        Dims({1, 1, rows, cols}));

  test_classifier(argc, argv, input, 255, 10,
                  train_images,
                  test_images,
                  [&](int bs, bool m) {
                    batch_size = bs;
                    test = m;
                    if(!test)
                      std::shuffle(train_data.begin(), train_data.end(), rnd);
                  },
                  [&](TensorAccess &ta, long batch) {
                    size_t i = batch * batch_size;
                    if(test)
                      loadInputTensor(ta, &test_data[i], batch_size, pixels);
                    else
                      loadInputTensor(ta, &train_data[i], batch_size, pixels);
                  },
                  [&](long index) -> int {
                    if(test)
                      return test_data[index].label;
                    else
                      return train_data[index].label;
                  }
                  );
  return 0;
}


SAGA_CLI_CMD("mnist",
             "minst <PATH> [OPTIONS ...]",
             "Infer/Train on mnist dataset",
             mnist_main);


