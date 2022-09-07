#include <string>

#include <math.h>
#include <unistd.h>
#include <signal.h>

#include <thread>
#include <mutex>
#include <condition_variable>

#include "saga.hpp"

using namespace saga;

static int g_run = 1;

typedef std::vector<std::shared_ptr<Tensor>> Stats;

static void
stop(int x)
{
    signal(SIGINT, SIG_DFL);
    g_run = 0;
}

struct Barrier {
    Barrier(size_t count) { pthread_barrier_init(&m_barrier, NULL, count); }

    ~Barrier() { pthread_barrier_destroy(&m_barrier); }

    void wait(void) { pthread_barrier_wait(&m_barrier); }

    pthread_barrier_t m_barrier;
};

static void
addStats(Graph &g, std::shared_ptr<Tensor> src, Stats *out, bool gradient)
{
    if(!out)
        return;
    auto s = g.addNode("stats", src, {{"gradient", gradient}});
    out->push_back(s->y());
}

// SqueezeNet's fire module: https://arxiv.org/pdf/1602.07360.pdf
// Optionally with a batch-norm module after squeeze layer

static std::shared_ptr<Node>
firemodule(Graph &g, std::shared_ptr<Node> input, int s1x1, int e1x1, int e3x3,
           bool with_bn, const std::string &name)
{
    auto s = g.addNode("conv", input->y(),
                       {{"size", 1}, {"activations", s1x1}, {"bias", !with_bn}},
                       name + "-s1x1");

    if(with_bn)
        s = g.addNode("batchnorm", s->y(), {}, name + "-bn");

    s = g.addNode("relu", s->y());

    auto e1 = g.addNode("conv", s->y(),
                        {{"size", 1}, {"activations", e1x1}, {"bias", true}},
                        name + "-e1x1");
    auto e3 = g.addNode(
        "conv", s->y(),
        {{"size", 3}, {"activations", e3x3}, {"pad", 1}, {"bias", true}},
        name + "-e3x3");

    e1 = g.addNode("relu", e1->y());
    e3 = g.addNode("relu", e3->y());

    return g.addNode("concat", Tensors{{"x0", e1->y()}, {"x1", e3->y()}});
}

static std::shared_ptr<Node>
squeezenet(Graph &g, std::shared_ptr<Node> n, bool with_bn, int output_classes)
{
    n = g.addNode("conv", n->y(),
                  {{"size", 3}, {"activations", 64}, {"bias", true}}, "conv0");
    n = g.addNode("relu", n->y());
    n = g.addNode("maxpool", n->y(), {{"size", 3}, {"stride", 2}});
    n = firemodule(g, n, 16, 64, 64, with_bn, "f1a");
    n = firemodule(g, n, 16, 64, 64, with_bn, "f1b");
    n = g.addNode("maxpool", n->y(), {{"size", 3}, {"stride", 2}});
    n = firemodule(g, n, 32, 128, 128, with_bn, "f2a");
    n = firemodule(g, n, 32, 128, 128, with_bn, "f2b");
    n = g.addNode("maxpool", n->y(), {{"size", 3}, {"stride", 2}});
    n = firemodule(g, n, 48, 192, 192, with_bn, "f3a");
    n = firemodule(g, n, 48, 192, 192, with_bn, "f3b");

    n = g.addNode(
        "conv", n->y(),
        {{"size", 1}, {"activations", output_classes}, {"bias", true}},
        "conv4");
    n = g.addNode("relu", n->y());
    n = g.addNode("avgpool", n->y(), {{"global", true}, {"stride", 2}});
    return n;
}

static std::shared_ptr<Node>
lecun(Graph &g, std::shared_ptr<Node> n, int output_classes, Stats *stats,
      bool transposed_weights)
{
    n = g.addNode("conv", n->y(),
                  {{"size", 5}, {"activations", 32}, {"bias", true}}, "conv1");

    n = g.addNode("relu", n->y());
    n = g.addNode("maxpool", n->y(), {{"size", 2}, {"stride", 2}});

    addStats(g, n->y(), stats, false);
    addStats(g, n->y(), stats, true);

    n = g.addNode("conv", n->y(),
                  {{"size", 5}, {"activations", 64}, {"bias", true}}, "conv2");
    n = g.addNode("relu", n->y());

    n = g.addNode("maxpool", n->y(), {{"size", 2}, {"stride", 2}});

    addStats(g, n->y(), stats, false);
    addStats(g, n->y(), stats, true);

    n = g.addNode(
        "fc", n->y(),
        {{"outputs", 1024}, {"bias", true}, {"transW", transposed_weights}},
        "fc1");

    n = g.addNode("relu", n->y());

    addStats(g, n->y(), stats, false);
    addStats(g, n->y(), stats, true);

    n = g.addNode("fc", n->y(),
                  {{"outputs", output_classes},
                   {"bias", true},
                   {"transW", transposed_weights}},
                  "fc2");
    return n;
}

static std::shared_ptr<Node>
convrelu(Graph &g, std::shared_ptr<Node> n, bool bn, int kernel_size,
         int activations, const std::string &name)
{
    n = g.addNode("conv", n->y(),
                  {{"size", kernel_size},
                   {"activations", activations},
                   {"pad", 1},
                   {"bias", !bn}},
                  name + "conv");
    if(bn)
        n = g.addNode("batchnorm", n->y(), {}, name + "bn");
    return g.addNode("relu", n->y());
}

static std::shared_ptr<Node>
vgg19(Graph &g, std::shared_ptr<Node> n, bool bn, int output_classes)
{
    n = convrelu(g, n, true, 3, 64, "1a");
    n = convrelu(g, n, bn, 3, 64, "1b");
    n = g.addNode("maxpool", n->y(), {{"size", 2}, {"stride", 2}});
    n = convrelu(g, n, bn, 3, 128, "2a");
    n = convrelu(g, n, bn, 3, 128, "2b");
    n = g.addNode("maxpool", n->y(), {{"size", 2}, {"stride", 2}});
    n = convrelu(g, n, bn, 3, 256, "3a");
    n = convrelu(g, n, bn, 3, 256, "3b");
    n = convrelu(g, n, bn, 3, 256, "3c");
    n = convrelu(g, n, bn, 3, 256, "3d");
    if(n->y()->dims_[2] > 7)
        n = g.addNode("maxpool", n->y(), {{"size", 2}, {"stride", 2}});
    n = convrelu(g, n, bn, 3, 512, "4a");
    n = convrelu(g, n, bn, 3, 512, "4b");
    n = convrelu(g, n, bn, 3, 512, "4c");
    n = convrelu(g, n, bn, 3, 512, "4d");
    if(n->y()->dims_[2] > 7)
        n = g.addNode("maxpool", n->y(), {{"size", 2}, {"stride", 2}});
    n = convrelu(g, n, bn, 3, 512, "5a");
    n = convrelu(g, n, bn, 3, 512, "5b");
    n = convrelu(g, n, bn, 3, 512, "5c");
    n = convrelu(g, n, bn, 3, 512, "5d");
    if(n->y()->dims_[2] > 7)
        n = g.addNode("maxpool", n->y(), {{"size", 2}, {"stride", 2}});

    n = g.addNode("fc", n->y(),
                  {{"outputs", 4096}, {"bias", true}, {"transW", true}}, "fc1");
    n = g.addNode("relu", n->y());
    n = g.addNode("dropout", n->y(), {{"prob", 0.5f}});
    n = g.addNode("fc", n->y(),
                  {{"outputs", 4096}, {"bias", true}, {"transW", true}}, "fc2");
    n = g.addNode("relu", n->y());
    n = g.addNode("dropout", n->y(), {{"prob", 0.5f}});
    n = g.addNode(
        "fc", n->y(),
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
    n = g.addNode("conv", n->y(),
                  {{"size", 3}, {"activations", 32}, {"pad", 1}, {"bias", !bn}},
                  "conv1");
    if(bn)
        n = g.addNode("batchnorm", n->y(), {}, "bn1");
    n = g.addNode("relu", n->y());
    n = g.addNode("maxpool", n->y(), {{"size", 2}, {"stride", 2}});

    n = g.addNode("conv", n->y(),
                  {{"size", 3}, {"activations", 64}, {"pad", 1}, {"bias", !bn}},
                  "conv2");
    if(bn)
        n = g.addNode("batchnorm", n->y(), {}, "bn2");
    n = g.addNode("relu", n->y());

    n = g.addNode("conv", n->y(),
                  {{"size", 3}, {"activations", 64}, {"pad", 1}, {"bias", !bn}},
                  "conv3");
    if(bn)
        n = g.addNode("batchnorm", n->y(), {}, "bn3");
    n = g.addNode("relu", n->y());

    n = g.addNode("maxpool", n->y(), {{"size", 2}, {"stride", 2}});

    n = g.addNode(
        "conv", n->y(),
        {{"size", 3}, {"activations", 128}, {"pad", 1}, {"bias", !bn}},
        "conv4");
    if(bn)
        n = g.addNode("batchnorm", n->y(), {}, "bn4");
    n = g.addNode("relu", n->y());

    n = g.addNode("maxpool", n->y(), {{"size", 2}, {"stride", 2}});

    n = g.addNode(
        "conv", n->y(),
        {{"size", 3}, {"activations", 128}, {"pad", 1}, {"bias", !bn}},
        "conv5");
    if(bn)
        n = g.addNode("batchnorm", n->y(), {}, "bn5");
    n = g.addNode("relu", n->y());

    n = g.addNode("maxpool", n->y(), {{"size", 2}, {"stride", 2}});

    n = g.addNode(
        "conv", n->y(),
        {{"size", 3}, {"activations", 256}, {"pad", 1}, {"bias", !bn}},
        "conv6");
    if(bn)
        n = g.addNode("batchnorm", n->y(), {}, "bn6");
    n = g.addNode("relu", n->y());

    n = g.addNode("maxpool", n->y(), {{"size", 2}, {"stride", 2}});

    n = g.addNode(
        "conv", n->y(),
        {{"size", 3}, {"activations", 256}, {"pad", 1}, {"bias", !bn}},
        "conv7");
    if(bn)
        n = g.addNode("batchnorm", n->y(), {}, "bn7");
    n = g.addNode("relu", n->y());

    n = g.addNode("maxpool", n->y(), {{"size", 2}, {"stride", 2}});

    n = g.addNode("fc", n->y(), {{"outputs", 1024}, {"bias", true}}, "fc1");
    n = g.addNode("relu", n->y());
    n = g.addNode("dropout", n->y(), {{"prob", 0.5f}});
    n = g.addNode("fc", n->y(), {{"outputs", 1024}, {"bias", true}}, "fc2");
    n = g.addNode("relu", n->y());
    n = g.addNode("dropout", n->y(), {{"prob", 0.5f}});
    n = g.addNode("fc", n->y(), {{"outputs", output_classes}, {"bias", true}},
                  "fc3");

    return n;
}

static std::shared_ptr<Node>
resnet(Graph &g, std::shared_ptr<Node> n, int output_classes, int stage2,
       int stage3, int stage4, int stage5, Stats *stats)
{
    n = g.addNode("spatialtransform", n->y(),
                  {{"width", 224}, {"height", 224}});

    n = g.addNode("conv", n->y(),
                  {{"size", 7}, {"activations", 64}, {"stride", 2}, {"pad", 3}},
                  "s1-conv");
    n = g.addNode("batchnorm", n->y(), {}, "s1-bn");
    n = g.addNode("relu", n->y());

    n = g.addNode("maxpool", n->y(), {{"size", 3}, {"stride", 2}, {"pad", 1}});

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

    n = g.addNode("avgpool", n->y(), {{"global", true}});

    n = g.addNode("fc", n->y(), {{"outputs", output_classes}, {"bias", true}},
                  "fc3");
    return n;
}

static std::shared_ptr<Node>
resnet50(Graph &g, std::shared_ptr<Node> n, int output_classes, Stats *stats)
{
    n = g.addNode("spatialtransform", n->y(),
                  {{"width", 224}, {"height", 224}});

    n = g.addNode("conv", n->y(),
                  {{"size", 7}, {"activations", 64}, {"stride", 2}, {"pad", 3}},
                  "s1-conv");
    n = g.addNode("batchnorm", n->y(), {}, "s1-bn");
    n = g.addNode("relu", n->y());

    n = g.addNode("maxpool", n->y(), {{"size", 3}, {"stride", 2}, {"pad", 1}});

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

    n = g.addNode("avgpool", n->y(), {{"global", true}});

    n = g.addNode("fc", n->y(), {{"outputs", output_classes}, {"bias", true}},
                  "fc3");
    return n;
}

static std::shared_ptr<Node>
make_network(Graph &g, std::shared_ptr<Node> n, const std::string &name,
             int output_classes, Stats *stats)
{
    if(name == "lecun") {
        return lecun(g, n, output_classes, stats, false);
    } else if(name == "lecun-t") {
        return lecun(g, n, output_classes, stats, true);
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

static void __attribute__((unused))
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
    bool train = true;

    int augmentation_angle = 0;
    int augmentation_zoom = 0;
    const char *graphdump = NULL;
    bool no_ui = false;
    std::shared_ptr<std::vector<std::shared_ptr<Tensor>>> stats;
    bool split = true;
    bool anomaly_detect = false;
    const char *program_dump_path = NULL;

    while((opt = getopt(argc, argv, "ns:l:b:hm:r:va:z:cCSG:UNAp:")) != -1) {
        switch(opt) {
        case 'A':
            anomaly_detect = true;
            break;
        case 'n':
            train = false;
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
        case 'U':
            no_ui = true;
            break;
        case 'N':
            split = false;
            break;
        case 'p':
            program_dump_path = optarg;
            break;
        }
    }

    printf("Test classifer: DataType:%s BatchSize:%d\n",
           dt == Tensor::DataType::HALF ? "fp16" : "fp32", batch_size);

    argc -= optind;
    argv += optind;

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

    n = make_network(g, n, mode, output_labels, stats.get());
    if(!n) {
        fprintf(stderr, "Network type %s not available\n", mode.c_str());
        exit(1);
    }
    n = g.addNode("catclassifier", n->y());

    if(verbose)
        g.print();

    const auto LABELS = n->outputs_["y"]->grad();
    const auto LOSS = n->outputs_["loss"];
    const auto INPUT = x;
    const auto OUTPUT = n->outputs_["y"];

    BatchedTensors bt{{LOSS, Phase::POST},
                      {LABELS, Phase::PRE},
                      {INPUT, Phase::PRE},
                      {OUTPUT, Phase::POST}};

    auto ui = no_ui ? saga::make_nui() : saga::make_tui();
    auto engine = createEngine(ui);
    auto contexts = engine->createContexts(true);

    const int s = split ? contexts.size() : 1;

    const size_t train_batches = train_inputs / (batch_size * s);
    const size_t test_batches = test_inputs / (batch_size * s);

    const size_t test_inputs_per_ctx = test_batches * batch_size;

    std::vector<std::thread> threads;

    signal(SIGINT, stop);
    auto stop_check = [&]() { return !g_run; };

    Barrier barrier(contexts.size());

    for(size_t thread_index = 0; thread_index < contexts.size();
        thread_index++) {
        threads.push_back(std::thread([=, &barrier] {
            double loss_sum = 0;
            long loss_sum_cnt = 0;
            int correct = 0;

            auto &ctx = contexts[thread_index];

            auto pre_ops = [&](long batch, const Program &p, auto tas) {
                load_inputs(*tas[INPUT], batch);
                if(p.getType() == ProgramType::TRAINING) {
                    auto &labels = *tas[LABELS];
                    const size_t offset = batch * batch_size;
                    for(int i = 0; i < batch_size; i++) {
                        labels.set({i}, get_label(offset + i));
                    }
                }
            };

            auto post_ops = [&](long batch, const Program &p, auto tas) {
                if(p.getType() == ProgramType::TRAINING) {
                    auto &loss = *tas[LOSS];
                    for(int i = 0; i < batch_size; i++) {
                        float v = loss.get({i});
                        loss_sum += v;
                    }
                    loss_sum_cnt += batch_size;
                    ui->updateCell(ctx->getUiPage(), p.getUiRow(), 3,
                                   UI::Align::LEFT, "%f",
                                   loss_sum / loss_sum_cnt);
                } else {
                    auto &output = *tas[OUTPUT];
                    const size_t base = batch * batch_size;
                    for(int i = 0; i < batch_size; i++) {
                        if(output.get({i}) == get_label(base + i))
                            correct++;
                    }
                }
            };

            const ProgramConfig pc{.pre_ops = pre_ops,
                                   .post_ops = post_ops,
                                   .learning_rate = learning_rate,
                                   .l2_lambda = 0.01,
                                   .tensor_layout = tensor_layout,
                                   .anomaly_detect = anomaly_detect};

            const ProgramSource ps{
                .graph = g, .batched_tensors = bt, .batch_size = batch_size};

            auto testing = ctx->createProgram(ps, ProgramType::INFERENCE, pc);

            auto training =
                train ? ctx->createProgram(ps, ProgramType::TRAINING, pc)
                      : nullptr;

            if(training)
                training->finalize();

            testing->finalize();

            if(thread_index == 0) {
                FILE *fp = stdout;

                if(program_dump_path != NULL)
                    fp = fopen(program_dump_path, "w");

                if(verbose > 1) {
                    testing->dump(fp, verbose > 2);

                    if(training) {
                        training->dump(fp, verbose > 2);
                    }
                }
                if(program_dump_path != NULL)
                    fclose(fp);
            }

            if(graphdump && training) {
                training->dumpGraph(graphdump);
                exit(0);
            }

            if(ui) {
                if(training) {
                    ui->updateCell(ctx->getUiPage(), training->getUiRow(), 2,
                                   UI::Align::RIGHT, "Loss:");
                }
                ui->updateCell(ctx->getUiPage(), testing->getUiRow(), 2,
                               UI::Align::RIGHT, "Accuracy:");
            }

            const long train_batch_offset =
                split ? train_batches * thread_index : 0;
            const long test_batch_offset =
                split ? test_batches * thread_index : 0;

            while(1) {
                if(train) {
#if 0
                    if(theta)
                        fill_theta(theta.get(), batch_size, augmentation_angle,
                                   augmentation_zoom);
#endif
                    // Train

                    barrier.wait();
                    if(!g_run)
                        break;
                    if(thread_index == 0)
                        epoch_begin(batch_size, false);
                    barrier.wait();

                    if(training->run(train_batches, stop_check,
                                     train_batch_offset) != ExecResult::OK) {
                        g_run = 0;
                    }

                    loss_sum = 0;
                    loss_sum_cnt = 0;
                }

                // Test
                barrier.wait();
                if(!g_run)
                    break;
                if(thread_index == 0)
                    epoch_begin(batch_size, true);
                barrier.wait();

                correct = 0;
                if(testing->run(test_batches, stop_check, test_batch_offset) !=
                   ExecResult::OK) {
                    g_run = 0;
                }

                double accuracy = (double)correct / test_inputs_per_ctx;

                ui->updateCell(ctx->getUiPage(), testing->getUiRow(), 3,
                               UI::Align::LEFT, "%.2f%%", 100.0f * accuracy);

                epoch++;

                if(accuracy >= 0.99f || epoch == 20) {
                    g_run = 0;
                    ui->refresh();
                }
            }
        }));
    }

    for(auto &t : threads) {
        t.join();
    }

    if(savepath != NULL && contexts.size() > 0)
        g.saveTensors(savepath, contexts[0].get());
}

}  // namespace saga
