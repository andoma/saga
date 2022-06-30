#include <math.h>
#include <signal.h>
#include <assert.h>
#include <unistd.h>
#include <sys/stat.h>
#include <fcntl.h>

#include <algorithm>
#include <numeric>
#include <string.h>

#include "saga.hpp"
#include "cli.h"

using namespace saga;

static int g_verbose = 0;

#include "test_jpegs.h"

static size_t
load_jpeg(long batch, int n, uint8_t *data, size_t len)
{
    const uint8_t *src;
    size_t srclen;
    n += batch;
    if((n % 3) == 0) {
        src = jpeg_r;
        srclen = sizeof(jpeg_r);
    } else if((n % 3) == 1) {
        src = jpeg_g;
        srclen = sizeof(jpeg_g);
    } else {
        src = jpeg_b;
        srclen = sizeof(jpeg_b);
    }
    memcpy(data, src, srclen);
    return srclen;
}

static int
jpeg_decoder_main(int argc, char **argv)
{
    int opt;
    auto dt = Tensor::DataType::FLOAT;

    int h_flip = 0;
    int v_flip = 0;
    int angle = 0;
    const int batch_size = 4;

    while((opt = getopt(argc, argv, "hvxyr:")) != -1) {
        switch(opt) {
        case 'h':
            dt = Tensor::DataType::HALF;
            break;
        case 'v':
            g_verbose++;
            break;
        case 'x':
            h_flip = 1;
            break;
        case 'y':
            v_flip = 1;
            break;
        case 'r':
            angle = atoi(optarg);
            break;
        }
    }

    argc -= optind;
    argv += optind;

    Graph g;

    auto n = g.addJpegDecoder(
        32, 32, dt, [&](long batch, int n, uint8_t *data, size_t len) {
            return load_jpeg(batch, n, data, len);
        });

    std::shared_ptr<Tensor> theta = makeCPUTensor(
        Tensor::DataType::FLOAT, Dims({batch_size, 2, 3}), "theta0");

    if(h_flip) {
        std::shared_ptr<Tensor> theta =
            makeCPUTensor(Tensor::DataType::FLOAT, Dims({1, 2, 3}), "h_flip");

        auto ta = theta->access();
        ta->set({0, 0, 0}, -1);
        ta->set({0, 1, 1}, 1);

        n = g.addSpatialTransform(n->y(), theta, -1, -1, true);
    }

    if(v_flip) {
        std::shared_ptr<Tensor> theta =
            makeCPUTensor(Tensor::DataType::FLOAT, Dims({1, 2, 3}), "v_flip");

        auto ta = theta->access();
        ta->set({0, 0, 0}, 1);
        ta->set({0, 1, 1}, -1);

        n = g.addSpatialTransform(n->y(), theta, -1, -1, true);
    }

    if(angle) {
        std::shared_ptr<Tensor> theta =
            makeCPUTensor(Tensor::DataType::FLOAT, Dims({1, 2, 3}), "rotation");

        const float r = angle * M_PI / 180.0;

        auto ta = theta->access();
        ta->set({0, 0, 0}, cos(r));
        ta->set({0, 0, 1}, -sin(r));
        ta->set({0, 1, 0}, sin(r));
        ta->set({0, 1, 1}, cos(r));

        n = g.addSpatialTransform(n->y(), theta, -1, -1, true);
    }

    if(g_verbose)
        g.print();

    auto ctx = createContext();
    auto p = ctx->createProgram(
        g, {.inference = true, .training = false, .batch_size = batch_size},
        {});

    if(g_verbose > 1)
        p->print();

    auto output = p->resolveTensor(n->y());

    p->infer(2);

    output->printRGB("GBRG");

    return 0;
}

SAGA_CLI_CMD("jpeg-decoder", "jpeg-decoder [OPTIONS ...]",
             "Run JPEG decoder test", jpeg_decoder_main);
