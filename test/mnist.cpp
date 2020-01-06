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

  #if 0

static void
loadInputTensor(Tensor &t, const LabeledImage *lis)
{
  const size_t batch_size = t.dims_[0];

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

  uint8_t labels[n];
  for(size_t j = 0; j < n; j++) {
    labels[j] = lis[j].label;
  }

  memcpy(t.hostMem(), labels, n);
}
#endif

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
  size_t batch_size = 64;
  bool learn = true;

  const char *loadpath = NULL;
  const char *savepath = NULL;
  int opt;

  std::string mode = "lecun";

  auto dt = Tensor::DataType::FLOAT;

  while((opt = getopt(argc, argv, "ns:l:b:hm:")) != -1) {
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

  auto input = std::make_shared<Tensor>(Tensor::DataType::FLOAT,
                                        Dims({1, 1, 28, 28}), "input");
  std::shared_ptr<Tensor> t;

  t = g.addNode("conv", {{"x", input}},
                {{"size", 5}, {"activations", 32}, {"bias", true}},
                "conv1");

  t = g.addNode("relu", {{"x", t}}, {});
  t = g.addNode("maxpool", {{"x", t}}, {{"size", 2}, {"stride", 2}});
  t = g.addNode("conv", {{"x", t}},
                {{"size", 5}, {"activations", 64}, {"bias", true}},
                "conv2");
  t = g.addNode("relu", {{"x", t}}, {});
  t = g.addNode("maxpool", {{"x", t}}, {{"size", 2}, {"stride", 2}});

  t = g.addNode("fc", {{"x", t}},
                {{"outputs", 1024}, {"bias", true}},
                "fc1");

  t = g.addNode("relu", {{"x", t}}, {});

  t = g.addNode("fc", {{"x", t}},
                {{"outputs", 10}, {"bias", true}},
                "fc2");

  t = g.addNode("catclassifier", {{"x", t}}, {});

  g.createGradients();

  g.print();

  auto ctx = createContext();
  auto p = ctx->createProgram(g, ProgramType::TRAINING, 1);

  p->print();

#if 0
  Network net(learn);

  if(loadpath)
    net.loadTensors(loadpath);

  Tensor input(Size(batch_size, 1, 28, 28), dt);

  auto tail = net.addLayer(makeInput(&input));

  if(mode == "squeezenet" || mode == "squeezenet-fc") {
    tail = net.addLayer(makeConvolution(64, 3, 1, 0, *tail, net));
    tail = net.addLayer(makeActivation(ActivationMode::RELU, 0, *tail, net));

    tail = net.addLayer(makePooling(PoolingMode::MAX, 3, 0, 2, *tail, net));
    tail = build_fire_module(net, *tail, 16, 64, 64);
    tail = build_fire_module(net, *tail, 16, 64, 64);

    tail = net.addLayer(makePooling(PoolingMode::MAX, 3, 0, 2, *tail, net));
    tail = build_fire_module(net, *tail, 32, 128, 128);
    tail = build_fire_module(net, *tail, 32, 128, 128);

    tail = net.addLayer(makePooling(PoolingMode::MAX, 3, 0, 2, *tail, net));
    tail = build_fire_module(net, *tail, 48, 192, 192);
    tail = build_fire_module(net, *tail, 48, 192, 192);

    tail = net.addLayer(makeDropout(0.25, tail, net));
    tail = net.addLayer(makeConvolution(labels, 1, 1, 0, *tail, net));
    tail = net.addLayer(makeActivation(ActivationMode::RELU, 0, *tail, net));

    if(mode == "squeezenet-fc") {

      tail = net.addLayer(makeFullyConnected(labels, *tail, net));
    } else {
      tail = net.addLayer(makePooling(PoolingMode::AVERAGE, 2, 0, 2,
                                      *tail, net));
    }

  } else if(mode == "lecun") {

    tail = net.addLayer(makeConvolution(32, 5, 1, 0, *tail, net, true,
                                        "c1w", "c1b"));
    tail = net.addLayer(makeActivation(ActivationMode::RELU, 0, *tail, net));
    tail = net.addLayer(makePooling(PoolingMode::MAX, 2, 0, 2, *tail, net));

    tail = net.addLayer(makeConvolution(64, 5, 1, 0, *tail, net, true,
                                        "c2w", "c2b"));
    tail = net.addLayer(makeActivation(ActivationMode::RELU, 0, *tail, net));
    tail = net.addLayer(makePooling(PoolingMode::MAX, 2, 0, 2, *tail, net));

    tail = net.addLayer(makeFullyConnected(1024, *tail, net, "fc1w", "fc1b"));
    tail = net.addLayer(makeActivation(ActivationMode::RELU, 0, *tail, net));

    tail = net.addLayer(makeFullyConnected(labels, *tail, net,
                                           "fc2w", "fc2b"));

  } else {

    fprintf(stderr, "Unknown mode %s", mode.c_str());
    exit(1);
  }
  auto tail_m1 = tail;
  tail = net.addLayer(makeCatClassifier(*tail, Tensor::Type::U8, net));

  unsigned int iteration = 0;
  while(g_run) {
    std::random_shuffle(train_data.begin(), train_data.end());

    // Train

    const int64_t t0 = get_ts();

    double loss_sum = 0;

    for(size_t i = 0; i < train_inputs && g_run; i += batch_size) {
      loadInputTensor(input, &train_data[i]);
      net.forward(false);
      if(learn) {
        loadOutputTensor(*tail->gradient(), &train_data[i]);
        net.backprop(iteration);
        auto loss = tail->loss();

        loss_sum += std::accumulate(loss.begin(), loss.end(), 0.0f);
      }
    }
    iteration++;

    if(!g_run)
      break;

    // Test
    const int64_t t1 = get_ts();
    int correct = 0;
    for(size_t i = 0; i < test_inputs; i += batch_size) {
      loadInputTensor(input, &test_data[i]);
      net.forward(true);

      for(size_t j = 0; j < batch_size; j++) {
        if(tail->output()->get(j, 0, 0, 0) == test_data[i + j].label)
          correct++;
      }
    }
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

  if(savepath)
    net.saveTensors(savepath);
#endif

  return 0;
}
