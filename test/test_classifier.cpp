#include <string>

#include <math.h>
#include <unistd.h>
#include <signal.h>

#include "saga.hpp"

using namespace saga;

static int g_run = 1;

typedef std::vector<std::shared_ptr<Tensor>> Stats;

static void
stop(int x)
{
    g_run = 0;
}

static int64_t __attribute__((unused)) get_ts(void)
{
    struct timespec tv;
    clock_gettime(CLOCK_MONOTONIC, &tv);
    return (int64_t)tv.tv_sec * 1000000LL + (tv.tv_nsec / 1000);
}

static void
addStats(Graph &g, std::shared_ptr<Tensor> src, Stats *out, bool gradient)
{
    if(!out)
        return;
    auto s = g.addNode("stats", {{"x", src}}, {{"gradient", gradient}});
    out->push_back(s->y());
}

// SqueezeNet's fire module: https://arxiv.org/pdf/1602.07360.pdf
// Optionally with a batch-norm module after squeeze layer

static std::shared_ptr<Node>
firemodule(Graph &g, std::shared_ptr<Node> input, int s1x1, int e1x1, int e3x3,
           bool with_bn, const std::string &name)
{
    auto s = g.addNode("conv", {{"x", input->y()}},
                       {{"size", 1}, {"activations", s1x1}, {"bias", !with_bn}},
                       name + "-s1x1");

    if(with_bn)
        s = g.addNode("batchnorm", {{"x", s->y()}}, {}, name + "-bn");

    s = g.addNode("relu", {{"x", s->y()}}, {});

    auto e1 = g.addNode("conv", {{"x", s->y()}},
                        {{"size", 1}, {"activations", e1x1}, {"bias", true}},
                        name + "-e1x1");
    auto e3 = g.addNode(
        "conv", {{"x", s->y()}},
        {{"size", 3}, {"activations", e3x3}, {"pad", 1}, {"bias", true}},
        name + "-e3x3");

    e1 = g.addNode("relu", {{"x", e1->y()}}, {});
    e3 = g.addNode("relu", {{"x", e3->y()}}, {});

    return g.addNode("concat", {{"x0", e1->y()}, {"x1", e3->y()}}, {});
}

static std::shared_ptr<Node>
squeezenet(Graph &g, std::shared_ptr<Node> n, bool with_bn, int output_classes)
{
    n = g.addNode("conv", {{"x", n->y()}},
                  {{"size", 3}, {"activations", 64}, {"bias", true}}, "conv0");
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

    n = g.addNode(
        "conv", {{"x", n->y()}},
        {{"size", 1}, {"activations", output_classes}, {"bias", true}},
        "conv4");
    n = g.addNode("relu", {{"x", n->y()}}, {});
    n = g.addNode("avgpool", {{"x", n->y()}},
                  {{"global", true}, {"stride", 2}});
    return n;
}

static std::shared_ptr<Node>
lecun(Graph &g, std::shared_ptr<Node> n, int output_classes, Stats *stats)
{
    n = g.addNode("conv", {{"x", n->y()}},
                  {{"size", 5}, {"activations", 32}, {"bias", true}}, "conv1");

    n = g.addNode("relu", {{"x", n->y()}}, {});
    n = g.addNode("maxpool", {{"x", n->y()}}, {{"size", 2}, {"stride", 2}});

    addStats(g, n->y(), stats, false);
    addStats(g, n->y(), stats, true);

    n = g.addNode("conv", {{"x", n->y()}},
                  {{"size", 5}, {"activations", 64}, {"bias", true}}, "conv2");
    n = g.addNode("relu", {{"x", n->y()}}, {});
    n = g.addNode("maxpool", {{"x", n->y()}}, {{"size", 2}, {"stride", 2}});

    addStats(g, n->y(), stats, false);
    addStats(g, n->y(), stats, true);

    n = g.addNode("fc", {{"x", n->y()}},
                  {{"outputs", 1024}, {"bias", true}, {"transW", true}}, "fc1");

    n = g.addNode("relu", {{"x", n->y()}}, {});

    addStats(g, n->y(), stats, false);
    addStats(g, n->y(), stats, true);

    n = g.addNode(
        "fc", {{"x", n->y()}},
        {{"outputs", output_classes}, {"bias", true}, {"transW", true}}, "fc2");
    return n;
}

static std::shared_ptr<Node>
convrelu(Graph &g, std::shared_ptr<Node> n, bool bn, int kernel_size,
         int activations, const std::string &name)
{
    n = g.addNode("conv", {{"x", n->y()}},
                  {{"size", kernel_size},
                   {"activations", activations},
                   {"pad", 1},
                   {"bias", !bn}},
                  name + "conv");
    if(bn)
        n = g.addNode("batchnorm", {{"x", n->y()}}, {}, name + "bn");
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
                  {{"outputs", 4096}, {"bias", true}, {"transW", true}}, "fc1");
    n = g.addNode("relu", {{"x", n->y()}}, {});
    n = g.addNode("dropout", {{"x", n->y()}}, {{"prob", 0.5f}});
    n = g.addNode("fc", {{"x", n->y()}},
                  {{"outputs", 4096}, {"bias", true}, {"transW", true}}, "fc2");
    n = g.addNode("relu", {{"x", n->y()}}, {});
    n = g.addNode("dropout", {{"x", n->y()}}, {{"prob", 0.5f}});
    n = g.addNode(
        "fc", {{"x", n->y()}},
        {{"outputs", output_classes}, {"bias", true}, {"transW", true}}, "fc3");
    return n;
}

/*
 *  RTX 2070 batchsize 512 mnist dataset

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
        n = g.addNode("batchnorm", {{"x", n->y()}}, {}, "bn1");
    n = g.addNode("relu", {{"x", n->y()}}, {});
    n = g.addNode("maxpool", {{"x", n->y()}}, {{"size", 2}, {"stride", 2}});

    n = g.addNode("conv", {{"x", n->y()}},
                  {{"size", 3}, {"activations", 64}, {"pad", 1}, {"bias", !bn}},
                  "conv2");
    if(bn)
        n = g.addNode("batchnorm", {{"x", n->y()}}, {}, "bn2");
    n = g.addNode("relu", {{"x", n->y()}}, {});

    n = g.addNode("conv", {{"x", n->y()}},
                  {{"size", 3}, {"activations", 64}, {"pad", 1}, {"bias", !bn}},
                  "conv3");
    if(bn)
        n = g.addNode("batchnorm", {{"x", n->y()}}, {}, "bn3");
    n = g.addNode("relu", {{"x", n->y()}}, {});

    n = g.addNode("maxpool", {{"x", n->y()}}, {{"size", 2}, {"stride", 2}});

    n = g.addNode(
        "conv", {{"x", n->y()}},
        {{"size", 3}, {"activations", 128}, {"pad", 1}, {"bias", !bn}},
        "conv4");
    if(bn)
        n = g.addNode("batchnorm", {{"x", n->y()}}, {}, "bn4");
    n = g.addNode("relu", {{"x", n->y()}}, {});

    n = g.addNode("maxpool", {{"x", n->y()}}, {{"size", 2}, {"stride", 2}});

    n = g.addNode(
        "conv", {{"x", n->y()}},
        {{"size", 3}, {"activations", 128}, {"pad", 1}, {"bias", !bn}},
        "conv5");
    if(bn)
        n = g.addNode("batchnorm", {{"x", n->y()}}, {}, "bn5");
    n = g.addNode("relu", {{"x", n->y()}}, {});

    n = g.addNode("maxpool", {{"x", n->y()}}, {{"size", 2}, {"stride", 2}});

    n = g.addNode(
        "conv", {{"x", n->y()}},
        {{"size", 3}, {"activations", 256}, {"pad", 1}, {"bias", !bn}},
        "conv6");
    if(bn)
        n = g.addNode("batchnorm", {{"x", n->y()}}, {}, "bn6");
    n = g.addNode("relu", {{"x", n->y()}}, {});

    n = g.addNode("maxpool", {{"x", n->y()}}, {{"size", 2}, {"stride", 2}});

    n = g.addNode(
        "conv", {{"x", n->y()}},
        {{"size", 3}, {"activations", 256}, {"pad", 1}, {"bias", !bn}},
        "conv7");
    if(bn)
        n = g.addNode("batchnorm", {{"x", n->y()}}, {}, "bn7");
    n = g.addNode("relu", {{"x", n->y()}}, {});

    n = g.addNode("maxpool", {{"x", n->y()}}, {{"size", 2}, {"stride", 2}});

    n = g.addNode("fc", {{"x", n->y()}}, {{"outputs", 1024}, {"bias", true}},
                  "fc1");
    n = g.addNode("relu", {{"x", n->y()}}, {});
    n = g.addNode("dropout", {{"x", n->y()}}, {{"prob", 0.5f}});
    n = g.addNode("fc", {{"x", n->y()}}, {{"outputs", 1024}, {"bias", true}},
                  "fc2");
    n = g.addNode("relu", {{"x", n->y()}}, {});
    n = g.addNode("dropout", {{"x", n->y()}}, {{"prob", 0.5f}});
    n = g.addNode("fc", {{"x", n->y()}},
                  {{"outputs", output_classes}, {"bias", true}}, "fc3");

    return n;
}

static std::shared_ptr<Node>
resnet(Graph &g, std::shared_ptr<Node> n, int output_classes, int stage2,
       int stage3, int stage4, int stage5, Stats *stats)
{
    n = g.addNode("spatialtransform", {{"x", n->y()}},
                  {{"width", 224}, {"height", 224}});

    n = g.addNode("conv", {{"x", n->y()}},
                  {{"size", 7}, {"activations", 64}, {"stride", 2}, {"pad", 3}},
                  "s1-conv");
    n = g.addNode("batchnorm", {{"x", n->y()}}, {}, "s1-bn");
    n = g.addNode("relu", {{"x", n->y()}}, {});

    n = g.addNode("maxpool", {{"x", n->y()}},
                  {{"size", 3}, {"stride", 2}, {"pad", 1}});

    addStats(g, n->y(), stats, true);

    for(int i = 0; i < stage2; i++) {
        n = g.addResNet(n->y(), 64, false,
                        std::string("s2_") + std::to_string(i));
    }

    addStats(g, n->y(), stats, true);

    for(int i = 0; i < stage3; i++) {
        n = g.addResNet(n->y(), 128, i == 0,
                        std::string("s3_") + std::to_string(i));
    }

    addStats(g, n->y(), stats, true);

    for(int i = 0; i < stage4; i++) {
        n = g.addResNet(n->y(), 256, i == 0,
                        std::string("s4_") + std::to_string(i));
    }

    addStats(g, n->y(), stats, true);

    for(int i = 0; i < stage5; i++) {
        n = g.addResNet(n->y(), 512, i == 0,
                        std::string("s5_") + std::to_string(i));
    }

    n = g.addNode("avgpool", {{"x", n->y()}}, {{"global", true}});

    n = g.addNode("fc", {{"x", n->y()}},
                  {{"outputs", output_classes}, {"bias", true}}, "fc3");
    return n;
}

static std::shared_ptr<Node>
resnet50(Graph &g, std::shared_ptr<Node> n, int output_classes, Stats *stats)
{
    n = g.addNode("spatialtransform", {{"x", n->y()}},
                  {{"width", 224}, {"height", 224}});

    n = g.addNode("conv", {{"x", n->y()}},
                  {{"size", 7}, {"activations", 64}, {"stride", 2}, {"pad", 3}},
                  "s1-conv");
    n = g.addNode("batchnorm", {{"x", n->y()}}, {}, "s1-bn");
    n = g.addNode("relu", {{"x", n->y()}}, {});

    n = g.addNode("maxpool", {{"x", n->y()}},
                  {{"size", 3}, {"stride", 2}, {"pad", 1}});

    addStats(g, n->y(), stats, true);

    n = g.addResNetBottleNeck(n->y(), 64, 256, false, "s2_1");
    n = g.addResNetBottleNeck(n->y(), 64, 256, false, "s2_2");
    n = g.addResNetBottleNeck(n->y(), 64, 256, false, "s2_3");

    addStats(g, n->y(), stats, true);

    n = g.addResNetBottleNeck(n->y(), 128, 512, true, "s3_1");
    n = g.addResNetBottleNeck(n->y(), 128, 512, false, "s3_2");
    n = g.addResNetBottleNeck(n->y(), 128, 512, false, "s3_3");
    n = g.addResNetBottleNeck(n->y(), 128, 512, false, "s3_4");

    addStats(g, n->y(), stats, true);

    n = g.addResNetBottleNeck(n->y(), 256, 1024, true, "s4_1");
    n = g.addResNetBottleNeck(n->y(), 256, 1024, false, "s4_2");
    n = g.addResNetBottleNeck(n->y(), 256, 1024, false, "s4_3");
    n = g.addResNetBottleNeck(n->y(), 256, 1024, false, "s4_4");
    n = g.addResNetBottleNeck(n->y(), 256, 1024, false, "s4_5");
    n = g.addResNetBottleNeck(n->y(), 256, 1024, false, "s4_6");

    addStats(g, n->y(), stats, true);

    n = g.addResNetBottleNeck(n->y(), 512, 2048, true, "s5_1");
    n = g.addResNetBottleNeck(n->y(), 512, 2048, false, "s5_2");
    n = g.addResNetBottleNeck(n->y(), 512, 2048, false, "s5_3");

    n = g.addNode("avgpool", {{"x", n->y()}}, {{"global", true}});

    n = g.addNode("fc", {{"x", n->y()}},
                  {{"outputs", output_classes}, {"bias", true}}, "fc3");
    return n;
}

static std::shared_ptr<Node>
make_network(Graph &g, std::shared_ptr<Node> n, const std::string &name,
             int output_classes, Stats *stats)
{
    if(name == "lecun") {
        return lecun(g, n, output_classes, stats);
    } else if(name == "test") {
        return test(g, n, false, output_classes);
    } else if(name == "test+bn") {
        return test(g, n, true, output_classes);
    } else if(name == "squeezenet") {
        return squeezenet(g, n, false, output_classes);
    } else if(name == "squeezenet+bn") {
        return squeezenet(g, n, true, output_classes);
    } else if(name == "vgg19") {
        return vgg19(g, n, false, output_classes);
    } else if(name == "vgg19+bn") {
        return vgg19(g, n, true, output_classes);
    } else if(name == "resnet-18") {
        return resnet(g, n, output_classes, 2, 2, 2, 2, stats);
    } else if(name == "resnet-34") {
        return resnet(g, n, output_classes, 3, 4, 6, 3, stats);
    } else if(name == "resnet-50") {
        return resnet50(g, n, output_classes, stats);
    } else {
        return nullptr;
    }
}

static void
fill_theta(Tensor *t, int batch_size, int augmentation_angle,
           int augmentation_zoom)
{
    float rs = augmentation_angle * M_PI * 2 / 360;
    float zs = augmentation_zoom * 0.01;

    auto ta = t->access();
    for(int i = 0; i < batch_size; i++) {
        const float r = (-1 + drand48() * 2) * rs;
        const float z = (-1 + drand48() * 2) * zs + 1.0f;
        ta->set({i, 0, 0}, cos(r) * z);
        ta->set({i, 0, 1}, -sin(r) * z);
        ta->set({i, 1, 0}, sin(r) * z);
        ta->set({i, 1, 1}, cos(r) * z);
    }
}

namespace saga {

void
test_classifier(int argc, char **argv, std::shared_ptr<Tensor> x,
                float input_range, int output_labels, size_t train_inputs,
                size_t test_inputs,
                std::function<void(int batch_size, bool test)> epoch_begin,
                std::function<void(TensorAccess &, long batch)> load_inputs,
                std::function<int(long index)> get_label)
{
    int batch_size = 64;

    int opt;
    float learning_rate = 3e-4;
    std::string mode = "lecun";
    int verbose = 0;
    auto dt = Tensor::DataType::FLOAT;
    auto tensor_layout = TensorLayout::Auto;
    const char *savepath = NULL;
    const char *loadpath = NULL;
    bool no_train = false;

    int augmentation_angle = 0;
    int augmentation_zoom = 0;
    const char *graphdump = NULL;

    std::shared_ptr<std::vector<std::shared_ptr<Tensor>>> stats;

    while((opt = getopt(argc, argv, "ns:l:b:hm:r:va:z:cCSG:")) != -1) {
        switch(opt) {
        case 'n':
            no_train = true;
            break;
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
            augmentation_angle = atoi(optarg);
            break;
        case 'z':
            augmentation_zoom = atoi(optarg);
            break;
        case 'c':
            tensor_layout = TensorLayout::NHWC;
            break;
        case 'C':
            tensor_layout = TensorLayout::NCHW;
            break;
        case 'S':
            stats = std::make_shared<Stats>();
            break;
        case 'G':
            graphdump = optarg;
            break;
        }
    }

    signal(SIGINT, stop);

    printf("Test classifer: DataType:%s BatchSize:%d\n",
           dt == Tensor::DataType::HALF ? "fp16" : "fp32", batch_size);

    argc -= optind;
    argv += optind;

    train_inputs = (train_inputs / batch_size) * batch_size;
    test_inputs = (test_inputs / batch_size) * batch_size;

    Graph g;

    if(loadpath != NULL)
        g.loadTensors(loadpath);

    std::shared_ptr<Tensor> theta;

    auto n = g.addConvert(x, dt, 1.0f / input_range);

    if(augmentation_angle || augmentation_zoom) {
        theta = makeCPUTensor(Tensor::DataType::FLOAT, Dims({batch_size, 2, 3}),
                              "theta");
        n = g.addSpatialTransform(n->y(), theta);
    }

    auto postconv = n->y();

    n = make_network(g, n, mode, output_labels, stats.get());
    if(!n) {
        fprintf(stderr, "Network type %s not available\n", mode.c_str());
        exit(1);
    }
    n = g.addNode("catclassifier", {{"x", n->y()}}, {});

    if(verbose)
        g.print();

    double loss_sum = 0;
    int correct = 0;

    const auto LABELS = n->outputs_["y"]->grad();
    const auto LOSS = n->outputs_["loss"];
    const auto INPUT = x;
    const auto OUTPUT = n->outputs_["y"];

    BatchedTensors bt{{LOSS, Phase::POST},
                      {LABELS, Phase::PRE},
                      {INPUT, Phase::PRE},
                      {OUTPUT, Phase::POST}};

    auto ctx = createContext();

    auto p = ctx->createProgram(g,
                                {.inference = true,
                                 .training = !no_train,
                                 .batch_size = batch_size,
                                 .initial_learning_rate = learning_rate,
                                 .tensor_layout = tensor_layout,
                                 .stop_check = [&]() { return !g_run; },
                                 .ui = saga::make_statbar()},
                                bt);

    if(verbose > 1)
        p->dump(stdout, verbose > 2);

    theta = p->resolveTensor(theta);
    postconv = p->resolveTensor(postconv);
    int epoch = 0;

    if(graphdump) {
        p->dumpGraph(graphdump);
        exit(0);
    }

    auto pre_ops = [&](long batch, bool training, auto tas) {
        load_inputs(*tas[INPUT], batch);

        if(training) {
            auto &labels = *tas[LABELS];
            const size_t offset = batch * batch_size;
            for(int i = 0; i < batch_size; i++) {
                labels.set({i}, get_label(offset + i));
            }
        }
    };

    auto post_ops = [&](long batch, bool training, auto tas) {
        if(training) {
            auto &loss = *tas[LOSS];
            for(int i = 0; i < batch_size; i++) {
                loss_sum += loss.get({i});
            }
        } else {
            auto &output = *tas[OUTPUT];
            const size_t base = batch * batch_size;
            for(int i = 0; i < batch_size; i++) {
                if(output.get({i}) == get_label(base + i))
                    correct++;
            }
        }
    };

    while(g_run) {
        const int64_t t0 = get_ts();

        if(!no_train) {
            if(theta)
                fill_theta(theta.get(), batch_size, augmentation_angle,
                           augmentation_zoom);

            // Train
            epoch_begin(batch_size, false);
            loss_sum = 0;
            if(p->train(train_inputs / batch_size, pre_ops, post_ops) !=
               ExecResult::OK)
                break;
        }

        // Test
        epoch_begin(batch_size, true);
        const int64_t t1 = get_ts();
        correct = 0;
        if(p->infer(test_inputs / batch_size, pre_ops, post_ops) !=
           ExecResult::OK)
            break;

        const int64_t t2 = get_ts();
        float percentage = 100.0 * correct / test_inputs;
        epoch++;
        printf("Epoch %4d: %3.3f%% Train:%.3fs Test:%.3fs Loss:%f\n", epoch,
               percentage, (t1 - t0) / 1e6, (t2 - t1) / 1e6,
               loss_sum / train_inputs);
        if(!g_run || percentage > 99)
            break;
    }

    if(savepath != NULL)
        g.saveTensors(savepath, p.get());
}

}  // namespace saga
