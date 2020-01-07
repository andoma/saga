#include <signal.h>
#include <assert.h>
#include <unistd.h>
#include <sys/stat.h>
#include <fcntl.h>

#include <algorithm>
#include <numeric>
#include <string.h>

#include "saga.h"


using namespace saga;

static int64_t __attribute__((unused))
get_ts(void)
{
  struct timespec tv;
  clock_gettime(CLOCK_MONOTONIC, &tv);
  return (int64_t)tv.tv_sec * 1000000LL + (tv.tv_nsec / 1000);
}

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

static const int table_xor[12] = {
  0, 0, 1,
  0, 1, 0,
  1, 0, 0,
  1, 1, 1,
};






extern int
minimal_main(int argc, char **argv)
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
  std::shared_ptr<Tensor> t;
  t = g.addNode("fc", {{"x", x}},
                {{"outputs", 4}, {"bias", true}});
  t = g.addNode("relu", {{"x", t}}, {});
  t = g.addNode("fc", {{"x", t}},
                {{"outputs", 2}, {"bias", true}});
  auto y = g.addNode("catclassifier", {{"x", t}}, {});
  auto dy = g.createGradients();
  g.print();

  auto ctx = createContext();
  auto p = ctx->createProgram(g, ProgramType::TRAINING, 4, 1e-3);

  p->print();

  x = p->resolveTensor(x);
  y = p->resolveTensor(y);
  dy = p->resolveTensor(dy);

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

  while(1) {
    p->exec(true);
    y->print("y");
    usleep(10000);
  }

  return 0;
}
