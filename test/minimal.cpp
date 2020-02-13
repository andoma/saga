#include <signal.h>
#include <assert.h>
#include <unistd.h>
#include <sys/stat.h>
#include <fcntl.h>

#include <algorithm>
#include <numeric>
#include <string.h>

#include "saga.h"
#include "cli.h"

using namespace saga;

static int64_t __attribute__((unused))
get_ts(void)
{
  struct timespec tv;
  clock_gettime(CLOCK_MONOTONIC, &tv);
  return (int64_t)tv.tv_sec * 1000000LL + (tv.tv_nsec / 1000);
}

#if 0
static const int table_or[12] = {
  0, 0, 0,
  0, 1, 1,
  1, 0, 1,
  1, 1, 1,
};

static const int table_and[12] = {
  0, 0, 0,
  0, 1, 0,
  1, 0, 0,
  1, 1, 1,
};
#endif

static const int table_xor[12] = {
  0, 0, 1,
  0, 1, 0,
  1, 0, 0,
  1, 1, 1,
};






extern int
minimal_main(int argc, char **argv, std::shared_ptr<UI> ui)
{
  int opt;

  auto dt = Tensor::DataType::FLOAT;

  while((opt = getopt(argc, argv, "h")) != -1) {
    switch(opt) {
    case 'h':
      dt = Tensor::DataType::HALF;
      break;
    }
  }

  argc -= optind;
  argv += optind;

  Graph g;

  auto x = std::make_shared<Tensor>(dt, Dims({1, 2}), "input");
  std::shared_ptr<Node> n;
  n = g.addNode("fc", {{"x", x}}, {{"outputs", 2}, {"bias", true}});
  n = g.addNode("relu", {{"x", n->y()}}, {});
  n = g.addNode("fc", {{"x", n->y()}}, {{"outputs", 2}, {"bias", true}});
  n = g.addNode("catclassifier", {{"x", n->y()}}, {});

  auto loss = n->outputs_["loss"];
  auto y = n->y();
  g.print();

  auto ctx = createContext();
  auto p = ctx->createProgram(g, {
      .inference = true,
      .training = true,
      .batch_size = 4,
      .initial_learning_rate = 1e-3,
      .tensor_layout = TensorLayout::NCHW
    });

  p->print();

  x = p->resolveTensor(x);
  y = p->resolveTensor(y);
  auto dy = y->grad();

  loss = p->resolveTensor(loss);

  printf("x: %s\n", x->info().c_str());
  printf("y: %s\n", y->info().c_str());
  printf("dy: %s\n", dy->info().c_str());

  auto xa = x->access();
  auto dya = dy->access();

  const int *tbl = table_xor;
  for(int i = 0; i < 4; i++) {
    xa->set({i, 0}, tbl[0]);
    xa->set({i, 1}, tbl[1]);
    dya->set({i}, tbl[2]);
    tbl += 3;
  }

  x->print("X");
  dy->print("DY");

  int iter = 0;
  while(1) {
    p->train();
    iter++;
    if((iter % 10000) == 0) {
      loss->print("LOSS");
      y->print("Y");
    }
  }

  return 0;
}


SAGA_CLI_CMD("minimal",
             "minimal [OPTIONS ...]",
             "Run some minimal tests",
             minimal_main);
