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

// -----------------------------------------------
static const TensorData relu_input = {
  {2, 4},
  {-100, 0, 100, 200,
   -1000, 0, 1000, 2000}
};

static const TensorData relu_output = {
  {2, 4},
  {0, 0, 100, 200,
   0, 0, 1000, 2000}
};

// -----------------------------------------------
static const TensorData conv_input_x = {
  {1, 1, 8, 8},
  {   1,  2,  3,  4,  5,  6,  7,  8,
     11, 12, 13, 14, 15, 16, 17, 18,
     21, 22, 23, 24, 25, 26, 27, 28,
     31, 32, 33, 34, 35, 36, 37, 38,
     41, 42, 43, 44, 45, 46, 47, 48,
     51, 52, 53, 54, 55, 56, 57, 58,
     61, 62, 63, 64, 65, 66, 67, 68,
     71, 72, 73, 74, 75, 76, 77, 78}
};

static const TensorData conv_input_w = {
  {2, 1, 3, 3},
  {   1,  2,  3,
      4,  5,  6,
      7,  8,  9,
     -1, -2, -3,
     -4, -5, -6,
     -7, -8, -9}
};

static const TensorData conv_input_b = {
  {1, 2},
  {   4, 5  }
};

static const TensorData conv_output = {
  {1, 2, 8, 8},
  {
   217,  326,  365,  404,  443,  482,  521,  335,
   505,  730,  775,  820,  865,  910,  955,  598,
   835,  1180,  1225,  1270,  1315,  1360,  1405,  868,
   1165,  1630,  1675,  1720,  1765,  1810,  1855,  1138,
   1495,  2080,  2125,  2170,  2215,  2260,  2305,  1408,
   1825,  2530,  2575,  2620,  2665,  2710,  2755,  1678,
   2155,  2980,  3025,  3070,  3115,  3160,  3205,  1948,
   1099,  1460,  1481,  1502,  1523,  1544,  1565,  905,
  -208, -317, -356, -395, -434, -473, -512, -326,
  -496, -721, -766, -811, -856, -901, -946, -589,
  -826, -1171, -1216, -1261, -1306, -1351, -1396, -859,
  -1156, -1621, -1666, -1711, -1756, -1801, -1846, -1129,
  -1486, -2071, -2116, -2161, -2206, -2251, -2296, -1399,
  -1816, -2521, -2566, -2611, -2656, -2701, -2746, -1669,
  -2146, -2971, -3016, -3061, -3106, -3151, -3196, -1939,
   -1090, -1451, -1472, -1493, -1514, -1535, -1556, -896 }
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
  auto p = ctx->createProgram(g, {
      .inference = true,
      .training = false,
      .batch_size = ref_output->dims_[0],
      .initial_learning_rate = 1e-3,
      .tensor_layout = TensorLayout::Auto
    });
  auto y = p->resolveTensor(n->y());
  p->infer();

  double sse = y->sse(*ref_output);

  if(sse > 1e-6) {
    printf("Test of %s FAILED sse:%e\n", op, sse);
    for(auto it : inputs) {
      it.second->print(it.first.c_str());
    }
    y->print("  Y");
    ref_output->print("REF");
    return 1;
  } else {
    printf("Test of %s OK SSE:%e\n", op, sse);
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

  int r = 0;
  r |= test_op(ctx, "relu", {{"x", load_tensor(dt, relu_input)}}, {},
          load_tensor(dt, relu_output));

  r |= test_op(ctx, "conv", {
      {"x", load_tensor(dt, conv_input_x)},
      {"w", load_tensor(dt, conv_input_w)},
      {"b", load_tensor(dt, conv_input_b)},
    }, {{"size", 3}, {"activations", 2}, {"pad", 1}, {"bias", true}},
    load_tensor(dt, conv_output));

  return r;
}


SAGA_CLI_CMD("ops",
             "ops [OPTIONS ...]",
             "Run test of operations",
             ops_main);
