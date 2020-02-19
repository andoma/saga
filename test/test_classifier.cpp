#include <unistd.h>
#include <signal.h>

#include "saga.h"

using namespace saga;


static int g_run = 1;

static void
stop(int x)
{
  g_run = 0;
}

static int64_t __attribute__((unused))
get_ts(void)
{
  struct timespec tv;
  clock_gettime(CLOCK_MONOTONIC, &tv);
  return (int64_t)tv.tv_sec * 1000000LL + (tv.tv_nsec / 1000);
}


// SqueezeNet's fire module: https://arxiv.org/pdf/1602.07360.pdf
// Optionally with a batch-norm module after squeeze layer

static std::shared_ptr<Node>
firemodule(Graph &g, std::shared_ptr<Node> input,
           int s1x1, int e1x1, int e3x3, bool with_bn,
           const std::string &name)
{
  auto s = g.addNode("conv", {{"x", input->y()}},
                     {{"size", 1}, {"activations", s1x1}, {"bias", !with_bn}},
                     name + "-s1x1");

  if(with_bn)
    s = g.addNode("batchnorm", {{"x", s->y()}}, {});

  s = g.addNode("relu", {{"x", s->y()}}, {});

  auto e1 = g.addNode("conv", {{"x", s->y()}},
                      {{"size", 1}, {"activations", e1x1}, {"bias", true}},
                      name + "-e1x1");
  auto e3 = g.addNode("conv", {{"x", s->y()}},
                      {{"size", 3}, {"activations", e3x3}, {"pad", 1}, {"bias", true}},
                      name + "-e3x3");

  e1 = g.addNode("relu", {{"x", e1->y()}}, {});
  e3 = g.addNode("relu", {{"x", e3->y()}}, {});

  return g.addNode("concat", {{"x0", e1->y()}, {"x1", e3->y()}}, {});
}


static std::shared_ptr<Node>
squeezenet(Graph &g, std::shared_ptr<Node> n, bool with_bn,
           int output_classes)
{
  n = g.addNode("conv", {{"x", n->y()}},
                {{"size", 3}, {"activations", 64}, {"bias", true}},
                "conv0");
  n = g.addNode("relu", {{"x", n->y()}}, {});
  n = g.addNode("maxpool", {{"x", n->y()}}, {{"size", 3}, {"stride", 2}});
  n = firemodule(g, n, 16, 64, 64, with_bn, "f1a");
  n = firemodule(g, n, 16, 64, 64, with_bn, "f1b");
  n = g.addNode("maxpool", {{"x", n->y()}}, {{"size", 3}, {"stride", 2}});
  n = firemodule(g, n, 32, 128, 128, with_bn, "f2a");
  n = firemodule(g, n, 32, 128, 128, with_bn, "f2b");
  n = g.addNode("maxpool", {{"x", n->y()}}, {{"size", 3}, {"stride", 2}});
  n = firemodule(g, n, 48, 192, 192, with_bn, "f3a");
  n = firemodule(g, n, 48, 192, 192, with_bn, "f3b");

  n = g.addNode("conv", {{"x", n->y()}},
                {{"size", 1}, {"activations", output_classes}, {"bias", true}},
                "conv4");
  n = g.addNode("relu", {{"x", n->y()}}, {});
  n = g.addNode("avgpool", {{"x", n->y()}}, {{"global", true}, {"stride", 2}});
  return n;
}





static std::shared_ptr<Node>
lecun(Graph &g, std::shared_ptr<Node> n, int output_classes)
{
  n = g.addNode("conv", {{"x", n->y()}},
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
                {{"outputs", 1024}, {"bias", true}, {"transW", true}},
                "fc1");

  n = g.addNode("relu", {{"x", n->y()}}, {});

  n = g.addNode("fc", {{"x", n->y()}},
                {{"outputs", output_classes}, {"bias", true}, {"transW", true}},
                "fc2");
  return n;
}



static std::shared_ptr<Node>
convrelu(Graph &g, std::shared_ptr<Node> n, bool bn,
         int kernel_size, int activations, const std::string &name)
{
  n = g.addNode("conv", {{"x", n->y()}},
                {{"size", kernel_size},
                 {"activations", activations},
                 {"pad", 1},
                 {"bias", !bn}},
                name + "conv");
  if(bn)
    n = g.addNode("batchnorm", {{"x", n->y()}}, {},
                  name + "bn");
  n = g.addNode("relu", {{"x", n->y()}}, {});
  return n;
}

static std::shared_ptr<Node>
vgg19(Graph &g, std::shared_ptr<Node> n, bool bn, int output_classes)
{
  n = convrelu(g, n, true, 3, 64, "1a");
  n = convrelu(g, n, bn, 3, 64, "1b");
  n = g.addNode("maxpool", {{"x", n->y()}}, {{"size", 2}, {"stride", 2}});
  n = convrelu(g, n, bn, 3, 128, "2a");
  n = convrelu(g, n, bn, 3, 128, "2b");
  n = g.addNode("maxpool", {{"x", n->y()}}, {{"size", 2}, {"stride", 2}});
  n = convrelu(g, n, bn, 3, 256, "3a");
  n = convrelu(g, n, bn, 3, 256, "3b");
  n = convrelu(g, n, bn, 3, 256, "3c");
  n = convrelu(g, n, bn, 3, 256, "3d");
  if(n->y()->dims_[2] > 7)
    n = g.addNode("maxpool", {{"x", n->y()}}, {{"size", 2}, {"stride", 2}});
  n = convrelu(g, n, bn, 3, 512, "4a");
  n = convrelu(g, n, bn, 3, 512, "4b");
  n = convrelu(g, n, bn, 3, 512, "4c");
  n = convrelu(g, n, bn, 3, 512, "4d");
  if(n->y()->dims_[2] > 7)
    n = g.addNode("maxpool", {{"x", n->y()}}, {{"size", 2}, {"stride", 2}});
  n = convrelu(g, n, bn, 3, 512, "5a");
  n = convrelu(g, n, bn, 3, 512, "5b");
  n = convrelu(g, n, bn, 3, 512, "5c");
  n = convrelu(g, n, bn, 3, 512, "5d");
  if(n->y()->dims_[2] > 7)
    n = g.addNode("maxpool", {{"x", n->y()}}, {{"size", 2}, {"stride", 2}});

  n = g.addNode("fc", {{"x", n->y()}},
                {{"outputs", 4096}, {"bias", true}, {"transW", true}},
                "fc1");
  n = g.addNode("relu", {{"x", n->y()}}, {});
  n = g.addNode("dropout", {{"x", n->y()}}, {{"prob", 0.5f}});
  n = g.addNode("fc", {{"x", n->y()}},
                {{"outputs", 4096}, {"bias", true}, {"transW", true}},
                "fc2");
  n = g.addNode("relu", {{"x", n->y()}}, {});
  n = g.addNode("dropout", {{"x", n->y()}}, {{"prob", 0.5f}});
  n = g.addNode("fc", {{"x", n->y()}},
                {{"outputs", output_classes}, {"bias", true}, {"transW", true}},
                "fc3");
   return n;
}



/*
 *  RTX 2070 batchsize 512

   test     float   NCHW    2.68
   test+bn  float   NCHW    3.02
   test     float   NHWC    3.25
   test+bn  float   NHWC   12.49

   test     half    NCHW    1.94
   test+bn  half    NCHW    2.16
   test     half    NHWC    1.20
   test+bn  half    NHWC    7.14  (Without CUDNN_BATCHNORM_SPATIAL_PERSISTENT)
   test+bn  half    NHWC    1.27  (With CUDNN_BATCHNORM_SPATIAL_PERSISTENT)
*/





static std::shared_ptr<Node>
test(Graph &g, std::shared_ptr<Node> n, bool bn, int output_classes)
{
  n = g.addNode("conv", {{"x", n->y()}},
                {{"size", 3}, {"activations", 32}, {"pad", 1}, {"bias", !bn}},
                "conv1");
  if(bn)
    n = g.addNode("batchnorm", {{"x", n->y()}}, {});
  n = g.addNode("relu", {{"x", n->y()}}, {});
  n = g.addNode("maxpool", {{"x", n->y()}}, {{"size", 2}, {"stride", 2}});

  n = g.addNode("conv", {{"x", n->y()}},
                {{"size", 3}, {"activations", 64}, {"pad", 1}, {"bias", !bn}},
                "conv2");
  if(bn)
    n = g.addNode("batchnorm", {{"x", n->y()}}, {});
  n = g.addNode("relu", {{"x", n->y()}}, {});

  n = g.addNode("conv", {{"x", n->y()}},
                {{"size", 3}, {"activations", 64}, {"pad", 1}, {"bias", !bn}},
                "conv3");
  if(bn)
    n = g.addNode("batchnorm", {{"x", n->y()}}, {});
  n = g.addNode("relu", {{"x", n->y()}}, {});

  n = g.addNode("maxpool", {{"x", n->y()}}, {{"size", 2}, {"stride", 2}});

  n = g.addNode("conv", {{"x", n->y()}},
                {{"size", 3}, {"activations", 128}, {"pad", 1}, {"bias", !bn}},
                "conv3");
  if(bn)
    n = g.addNode("batchnorm", {{"x", n->y()}}, {});
  n = g.addNode("relu", {{"x", n->y()}}, {});

  n = g.addNode("maxpool", {{"x", n->y()}}, {{"size", 2}, {"stride", 2}});

  n = g.addNode("conv", {{"x", n->y()}},
                {{"size", 3}, {"activations", 128}, {"pad", 1}, {"bias", !bn}},
                "conv3");
  if(bn)
    n = g.addNode("batchnorm", {{"x", n->y()}}, {});
  n = g.addNode("relu", {{"x", n->y()}}, {});

  n = g.addNode("maxpool", {{"x", n->y()}}, {{"size", 2}, {"stride", 2}});

  n = g.addNode("conv", {{"x", n->y()}},
                {{"size", 3}, {"activations", 256}, {"pad", 1}, {"bias", !bn}},
                "conv3");
  if(bn)
    n = g.addNode("batchnorm", {{"x", n->y()}}, {});
  n = g.addNode("relu", {{"x", n->y()}}, {});

  n = g.addNode("maxpool", {{"x", n->y()}}, {{"size", 2}, {"stride", 2}});

  n = g.addNode("conv", {{"x", n->y()}},
                {{"size", 3}, {"activations", 256}, {"pad", 1}, {"bias", !bn}},
                "conv3");
  if(bn)
    n = g.addNode("batchnorm", {{"x", n->y()}}, {});
  n = g.addNode("relu", {{"x", n->y()}}, {});

  n = g.addNode("maxpool", {{"x", n->y()}}, {{"size", 2}, {"stride", 2}});

  n = g.addNode("fc", {{"x", n->y()}},
                {{"outputs", 1024}, {"bias", true}},
                "fc1");
  n = g.addNode("relu", {{"x", n->y()}}, {});
  n = g.addNode("dropout", {{"x", n->y()}}, {{"prob", 0.5f}});
  n = g.addNode("fc", {{"x", n->y()}},
                {{"outputs", 1024}, {"bias", true}},
                "fc2");
  n = g.addNode("relu", {{"x", n->y()}}, {});
   n = g.addNode("dropout", {{"x", n->y()}}, {{"prob", 0.5f}});
  n = g.addNode("fc", {{"x", n->y()}},
                {{"outputs", output_classes}, {"bias", true}},
                "fc3");

  return n;
}




static std::shared_ptr<Node>
make_network(Graph &g, std::shared_ptr<Node> n, const std::string &name,
             int output_classes)
{
  if(name == "lecun") {
    return lecun(g, n, output_classes);
  } else if(name == "test") {
    return test(g, n, false, output_classes);
  } else if(name == "test+bn") {
    return test(g, n, true, output_classes);
  } else if(name == "squeezenet") {
    return squeezenet(g, n, false, output_classes);
  } else if(name == "squeezenet+bn") {
    return squeezenet(g, n, true, output_classes);
  } else if(name == "test+bn") {
    return squeezenet(g, n, true, output_classes);
  } else if(name == "vgg19") {
    return vgg19(g, n, false, output_classes);
  } else if(name == "vgg19+bn") {
    return vgg19(g, n, true, output_classes);
  } else {
    return nullptr;
  }
}




static std::shared_ptr<Node>
convert(Graph &g, std::shared_ptr<Tensor> x, float scale, Tensor::DataType dt)
{
  return g.addNode("convert", {{"x", x}},
                   {{"scale", scale}, {"datatype", (int)dt}});
}


static void
fill_theta(Tensor *t, int batch_size)
{
  auto ta = t->access();
  for(int i = 0; i < batch_size; i++) {

    float xx = drand48() > 0.5 ? 1 : -1;
    float xy = drand48() > 0.5 ? 1 : -1;
    float yx = drand48() > 0.5 ? 1 : -1;
    float yy = drand48() > 0.5 ? 1 : -1;

    ta->set({i, 0, 0}, ( 0.9 + drand48() * 0.2) * xx);
    ta->set({i, 0, 1}, (-0.1 + drand48() * 0.2) * xy);
    ta->set({i, 0, 2}, -0.1 + drand48() * 0.2);
    ta->set({i, 1, 0}, (-0.1 + drand48() * 0.2) * yx);
    ta->set({i, 1, 1}, (0.9 + drand48() * 0.2) * yy);
    ta->set({i, 1, 2}, -0.1 + drand48() * 0.2);
  }
}


namespace saga {

void
test_classifier(int argc, char **argv,
                std::shared_ptr<Tensor> input,
                float input_range,
                int output_labels,
                size_t train_inputs,
                size_t test_inputs,
                std::function<void(void)> epoch_begin,
                std::function<void(Tensor &x, Tensor &dy, size_t i)> load_train,
                std::function<void(Tensor &x, int *labels, size_t i)> load_test)
{
  signal(SIGINT, stop);

  int batch_size = 64;

  int opt;
  float learning_rate = 3e-4;
  std::string mode = "lecun";
  int verbose = 0;
  bool augmentation = false;
  auto dt = Tensor::DataType::FLOAT;
  auto tensor_layout = TensorLayout::Auto;
  const char *savepath = NULL;
  const char *loadpath = NULL;

  while((opt = getopt(argc, argv, "ns:l:b:hm:r:vacC")) != -1) {
    switch(opt) {
    case 's':
      savepath = optarg;
      break;
    case 'l':
      loadpath = optarg;
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
    case 'v':
      verbose++;
      break;
    case 'a':
      augmentation = true;
      break;
    case 'c':
      tensor_layout = TensorLayout::NHWC;
      break;
    case 'C':
      tensor_layout = TensorLayout::NCHW;
      break;
    }
  }

  printf("Test classifer: DataType:%s BatchSize:%d\n",
         dt == Tensor::DataType::HALF ? "fp16" : "fp32",
         batch_size);

  argc -= optind;
  argv += optind;

  train_inputs = (train_inputs / batch_size) * batch_size;
  test_inputs  = (test_inputs  / batch_size) * batch_size;

  Graph g;

  if(loadpath != NULL)
    g.loadTensors(loadpath);

  std::shared_ptr<Node> n;
  std::shared_ptr<Tensor> theta;

  if(augmentation) {
    theta = makeCPUTensor(Tensor::DataType::FLOAT,
                          Dims({batch_size, 2, 3}), "theta");

    n = convert(g, input, 1.0f / input_range, Tensor::DataType::FLOAT);
    n = g.addNode("spatialtransform", {{"x", n->y()}, {"theta", theta}},
                  {});

    if(dt != Tensor::DataType::FLOAT) {
      n = convert(g, n->y(), 1.0f, dt);
    }
  } else {
    n = convert(g, input, 1.0f / input_range, dt);
  }

  n = make_network(g, n, mode, output_labels);

  n = g.addNode("catclassifier", {{"x", n->y()}}, {});
  auto y = n->y();
  auto loss = n->outputs_["loss"];

  if(verbose)
    g.print();

  auto ctx = createContext();
  auto p = ctx->createProgram(g, {
      .inference = true,
      .training = true,
      .batch_size = batch_size,
      .initial_learning_rate = learning_rate,
      .tensor_layout = tensor_layout
    });

  if(verbose > 1)
    p->print();

  auto x = p->resolveTensor(input);
  y = p->resolveTensor(y);
  auto dy = y->grad();
  loss = p->resolveTensor(loss);
  theta = p->resolveTensor(theta);

  int labels[batch_size];

  while(g_run) {
    if(theta)
      fill_theta(theta.get(), batch_size);

    epoch_begin();

    // Train
    const int64_t t0 = get_ts();
    double loss_sum = 0;

    for(size_t i = 0; i < train_inputs && g_run; i += batch_size) {
      load_train(*x, *dy, i);
      p->train();
      auto la = loss->access();
      for(int i = 0; i < batch_size; i++) {
        loss_sum += la->get({i});
      }
    }


    // Test
    const int64_t t1 = get_ts();
    int correct = 0;
    for(size_t i = 0; i < test_inputs && g_run; i += batch_size) {
      load_test(*x, labels, i);
      p->infer();

      auto ta = y->access();

      for(int j = 0; j < batch_size; j++) {
        if(ta->get({j}) == labels[j])
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

  if(savepath != NULL)
    g.saveTensors(savepath, p.get());

}

}
