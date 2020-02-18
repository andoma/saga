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




struct TensorData {
  Dims dims;
  std::vector<float> data;
};


static const TensorData relu_input = {
  {1, 4},
  {-100, 0, 100, 200}
};

static const TensorData relu_output = {
  {1, 4},
  {-100, 0, 100, 200}
};



static std::shared_ptr<Tensor>
load_tensor(Tensor::DataType dt, const TensorData &td)
{
  auto t = makeCPUTensor(dt, td.dims);
  auto dst = t->access();
  const size_t elements = td.dims.elements();

  Dims e(td.dims.size(), 0);
  for(size_t i = 0; i < elements; i++) {
    dst->set(e, td.data[i]);

    for(ssize_t j = e.size() - 1; j >= 0; j--) {
      e[j]++;
      if(e[j] == td.dims[j]) {
        e[j] = 0;
      } else {
        break;
      }
    }
  }
  return t;
}



static int
test_op(std::shared_ptr<Context> ctx,
        const char *op,
        const Tensors &inputs,
        const Attributes &attributes,
        std::shared_ptr<Tensor> ref_output)
{
  Graph g;

  auto n = g.addNode(op, inputs, attributes);

  g.print();

  auto p = ctx->createProgram(g, {
      .inference = true,
      .batch_size = 1,
      .initial_learning_rate = 1e-3,
      .tensor_layout = TensorLayout::Auto
    });
  auto y = p->resolveTensor(n->y());
  p->print();
  p->infer();

  double sse = y->sse(*ref_output);

  if(sse > 1e-6) {
    for(auto it : inputs) {
      it.second->print(it.first.c_str());
    }
    y->print("  Y");
    ref_output->print("REF");
    return 1;
  }
  return 0;
}



extern int
ops_main(int argc, char **argv, std::shared_ptr<UI> ui)
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

  auto ctx = createContext();

  test_op(ctx, "relu", {{"x", load_tensor(dt, relu_input)}}, {},
          load_tensor(dt, relu_output));


  return 0;
}


SAGA_CLI_CMD("ops",
             "ops [OPTIONS ...]",
             "Run test of operations",
             ops_main);
