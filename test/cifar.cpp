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

#include "saga.hpp"
#include "test_classifier.h"
#include "cli.h"

using namespace saga;

static void *
load(const std::string &path)
{
    int fd = open(path.c_str(), O_RDONLY);
    if(fd == -1) {
        fprintf(stderr, "Unable to open %s -- %s\n", path.c_str(),
                strerror(errno));
        exit(1);
    }

    struct stat st;
    if(fstat(fd, &st)) {
        fprintf(stderr, "Unable to stat %s -- %s\n", path.c_str(),
                strerror(errno));
        exit(1);
    }

    return mmap(NULL, st.st_size, PROT_READ, MAP_SHARED, fd, 0);
}

struct LabeledImage {
    LabeledImage(const uint8_t *image, unsigned int label)
      : image(image), label(label)
    {
    }
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
loadInputTensor(TensorAccess &ta, const LabeledImage *lis, int batch_size)
{
    const size_t size = 32 * 32 * 3;

    uint8_t prep[size * batch_size];
    uint8_t *dst = prep;
    for(int n = 0; n < batch_size; n++) {
        const uint8_t *src = lis[n].image;

        for(int y = 0; y < 32; y++) {
            for(int x = 0; x < 32; x++) {
                for(int c = 0; c < 3; c++) {
                    *dst++ = src[c * 32 * 32 + y * 32 + x];
                }
            }
        }
    }

    ta.copyBytesFrom({}, prep, size * batch_size);
}

static int
cifar_main(int argc, char **argv)
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

    int batch_size = 0;
    bool test = false;

    auto input = std::make_shared<Tensor>(Tensor::DataType::U8,
                                          Dims({1, 3, 32, 32}), "input");

    std::random_device rd;
    std::mt19937 rnd(rd());

    test_classifier(
        argc, argv, input, 255, 10, train_inputs, test_inputs,
        [&](int bs, bool m) {
            batch_size = bs;
            test = m;
            if(!test)
                std::shuffle(train_data.begin(), train_data.end(), rnd);
        },
        [&](TensorAccess &ta, long batch) {
            size_t i = batch * batch_size;
            if(test)
                loadInputTensor(ta, &test_data[i], batch_size);
            else
                loadInputTensor(ta, &train_data[i], batch_size);
        },
        [&](long index) -> int {
            if(test)
                return test_data[index].label;
            else
                return train_data[index].label;
        });

    return 0;
}

SAGA_CLI_CMD("cifar", "cifar <PATH> [OPTIONS ...]",
             "Infer/Train on cifar dataset", cifar_main);
