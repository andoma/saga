#include <signal.h>
#include <assert.h>
#include <unistd.h>
#include <sys/stat.h>
#include <fcntl.h>

#include <algorithm>
#include <numeric>
#include <string.h>

#include "saga.h"

static int g_run = 1;

static void
stop(int x)
{
  g_run = 0;
}


using namespace saga;

static int64_t __attribute__((unused))
get_ts(void)
{
  struct timespec tv;
  clock_gettime(CLOCK_MONOTONIC, &tv);
  return (int64_t)tv.tv_sec * 1000000LL + (tv.tv_nsec / 1000);
}



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
  auto ta = t.access();

  for(int n = 0; n < batch_size; n++) {
    const uint8_t *src = lis[n].image;

    for(int y = 0; y < t.dims_[2]; y++) {
      for(int x = 0; x < t.dims_[3]; x++) {
        float v = src[y * 28 + x] / 255.0;
        ta->set({n, 0, y, x}, v);
      }
    }
  }
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

#if 0

// SqueezeNet's fire module: https://arxiv.org/pdf/1602.07360.pdf
// + batchnorm
static std::shared_ptr<Layer>
build_fire_module(Network &net, const Layer &input,
                  int s1x1, int e1x1, int e3x3)
{
  auto s = net.addLayer(makeConvolution(s1x1, 1, 1, 0, input, net));
  s = net.addLayer(makeBatchNorm(1e-5, *s, net, 0.25));
  s = net.addLayer(makeActivation(ActivationMode::RELU, 0, *s, net));

  auto e1 = net.addLayer(makeConvolution(e1x1, 1, 1, 0, *s, net, false));
  auto e3 = net.addLayer(makeConvolution(e3x3, 3, 1, 1, *s, net, false));

  e1 = net.addLayer(makeActivation(ActivationMode::RELU, 0, *e1, net));
  e3 = net.addLayer(makeActivation(ActivationMode::RELU, 0, *e3, net));

  return net.addLayer(makeConcat({e1.get(), e3.get()}, net));
}
#endif


extern int
mnist_main(int argc, char **argv)
{
  int batch_size = 64;
  bool learn = true;

  const char *loadpath = NULL;
  const char *savepath = NULL;
  int opt;
  float learning_rate = 3e-4;
  std::string mode = "lecun";

  auto dt = Tensor::DataType::FLOAT;

  while((opt = getopt(argc, argv, "ns:l:b:hm:r:")) != -1) {
    switch(opt) {
    case 's':
      savepath = optarg;
      break;
    case 'l':
      loadpath = optarg;
      break;
    case 'n':
      learn = false;
      break;
    case 'b':
      batch_size = atoi(optarg);
      break;
    case 'h':
      dt = Tensor::DataType::HALF;
      break;
    case 'm':
      mode = optarg;
      break;
    case 'r':
      learning_rate = strtod(optarg, NULL);
      break;
    }
  }

  argc -= optind;
  argv += optind;

  if(argc < 1) {
    fprintf(stderr, "mnist usage: [OPTIONS] <path>\n");
    return 1;
  }

  std::string path(argv[0]);

  signal(SIGINT, stop);

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

  printf("learn=%d\n", learn);
  printf("labels=%d\n", labels);
  printf("loadpath=%s\n", loadpath);
  printf("savepath=%s\n", savepath);
  printf("data_type=%d\n", (int)dt);

  Graph g;

  if(loadpath)
    g.loadRawTensors(loadpath);

  auto x = std::make_shared<Tensor>(Tensor::DataType::FLOAT,
                                    Dims({1, 1, 28, 28}), "input");
  std::shared_ptr<Node> n;

  n = g.addNode("conv", {{"x", x}},
                {{"size", 5}, {"activations", 32}, {"bias", true}},
                "conv1");

  n = g.addNode("relu", {{"x", n->y()}}, {});
  n = g.addNode("maxpool", {{"x", n->y()}}, {{"size", 2}, {"stride", 2}});
  n = g.addNode("conv", {{"x", n->y()}},
                {{"size", 5}, {"activations", 64}, {"bias", true}},
                "conv2");
  n = g.addNode("relu", {{"x", n->y()}}, {});
  n = g.addNode("maxpool", {{"x", n->y()}}, {{"size", 2}, {"stride", 2}});

  n = g.addNode("fc", {{"x", n->y()}},
                {{"outputs", 1024}, {"bias", true}},
                "fc1");

  n = g.addNode("relu", {{"x", n->y()}}, {});

  n = g.addNode("fc", {{"x", n->y()}},
                {{"outputs", 10}, {"bias", true}},
                "fc2");

  n = g.addNode("catclassifier", {{"x", n->y()}}, {});
  auto y = n->y();
  auto loss = n->outputs_["loss"];

  auto dy = g.createGradients();

  g.print();

  auto ctx = createContext();
  auto p = ctx->createProgram(g, ProgramType::TRAINING, batch_size,
                              learning_rate);

  p->print();

  x = p->resolveTensor(x);
  y = p->resolveTensor(y);
  dy = p->resolveTensor(dy);
  loss = p->resolveTensor(loss);

  while(g_run) {
    std::random_shuffle(train_data.begin(), train_data.end());

    // Train
    const int64_t t0 = get_ts();
    double loss_sum = 0;

    for(size_t i = 0; i < train_inputs && g_run; i += batch_size) {
      loadInputTensor(*x, &train_data[i]);
      loadOutputTensor(*dy, &train_data[i]);
      p->exec(true);
      auto la = loss->access();
      for(int i = 0; i < batch_size; i++) {
        loss_sum += la->get({i});
      }
    }


    // Test
    const int64_t t1 = get_ts();
    int correct = 0;
    for(size_t i = 0; i < test_inputs && g_run; i += batch_size) {
      loadInputTensor(*x, &test_data[i]);
      p->exec(false);

      auto ta = y->access();

      for(int j = 0; j < batch_size; j++) {
        if(ta->get({j}) == test_data[i + j].label)
          correct++;
      }
    }

    if(!g_run)
      break;

    const int64_t t2 = get_ts();
    float percentage = 100.0 * correct / test_inputs;
    printf("%3.3f%% Train:%.3fs Test:%.3fs Loss:%f\n",
           percentage,
           (t1 - t0) / 1e6,
           (t2 - t1) / 1e6,
           loss_sum / test_inputs);
    if(percentage > 99)
      break;
  }

  return 0;
}
