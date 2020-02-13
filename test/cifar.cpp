#include <math.h>
#include <signal.h>
#include <assert.h>
#include <unistd.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <errno.h>
#include <string.h>
#include <sys/mman.h>

#include <random>
#include <algorithm>
#include <numeric>

#include "saga.h"
#include "test_classifier.h"
#include "cli.h"

using namespace saga;

static void *
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

  return mmap(NULL, st.st_size, PROT_READ, MAP_SHARED, fd, 0);
}




struct LabeledImage {
  LabeledImage(const uint8_t *image, unsigned int label)
    : image(image)
    , label(label)
  {}
  const uint8_t *image;
  unsigned int label;
};


static void
load_batch(std::vector<LabeledImage> &lis, const std::string &path)
{
  void *data = load(path);
  const uint8_t *s = (const uint8_t *)data;
  for(size_t i = 0; i < 10000; i++) {
    uint8_t label = *s++;
    assert(label < 10);
    lis.push_back(LabeledImage(s, label));
    s += 3072;
  }
}




static void
loadInputTensor(Tensor &t, const LabeledImage *lis)
{
  const int batch_size = t.dims_[0];
  const size_t size = 32 * 32 * 3;

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
cifar_main(int argc, char **argv, std::shared_ptr<UI> ui)
{
  argc--;
  argv++;

  std::string path(argv[0]);

  std::vector<LabeledImage> train_data, test_data;
  load_batch(train_data, path + "/data_batch_1.bin");
  load_batch(train_data, path + "/data_batch_2.bin");
  load_batch(train_data, path + "/data_batch_3.bin");
  load_batch(train_data, path + "/data_batch_4.bin");
  load_batch(train_data, path + "/data_batch_5.bin");

  load_batch(test_data, path + "/test_batch.bin");

  const size_t train_inputs = train_data.size();
  const size_t test_inputs = test_data.size();


  auto input = std::make_shared<Tensor>(Tensor::DataType::U8,
                                        Dims({1, 3, 32, 32}), "input");

  std::random_device rd;
  std::mt19937 rnd(rd());

  test_classifier(argc, argv, input, 255, 10,
                  train_inputs,
                  test_inputs,
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
                  },
                  ui);

  return 0;
}


SAGA_CLI_CMD("cifar",
             "cifar <PATH> [OPTIONS ...]",
             "Infer/Train on cifar dataset",
             cifar_main);


