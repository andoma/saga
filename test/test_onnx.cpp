#include <limits.h>
#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <libgen.h>

#include "saga.h"

using namespace saga;



static int
test_one(const char *model_path, std::shared_ptr<Context> ctx, int verbose)
{
  char dirtmp[PATH_MAX];
  snprintf(dirtmp, sizeof(dirtmp), "%s", model_path);

  char *base_path = dirname(dirtmp);
  char input_path[PATH_MAX];
  char output_path[PATH_MAX];

  auto g = Graph::load(model_path);
  if(g == NULL) {
    fprintf(stderr, "Failed to load model graph %s\n", model_path);
    return 1;
  }
  if(verbose)
    g->print();

  auto p = ctx->createProgram(*g, ProgramType::INFERENCE, 1, 0,
                              TensorLayout::NCHW);
  if(verbose)
    p->print();

  auto input = p->resolveTensor(*g->inputs_.begin());
  auto output = p->resolveTensor(*g->outputs_.begin());

  printf("Test: %s\n", base_path);
  printf("  INPUT: %s\n", input->info().c_str());
  printf("  OUTPUT: %s\n", output->info().c_str());

  for(int i = 0; ; i++) {
    snprintf(input_path, sizeof(input_path), "%s/test_data_set_%d/input_0.pb",
             base_path, i);

    if(access(input_path, R_OK))
      break;

    auto loaded_input = Tensor::loadProtoBuf(input_path);
    if(!loaded_input) {
      fprintf(stderr, "Failed to load input %s\n", input_path);
      return 1;
    }

    input->copyFrom(*loaded_input);

    snprintf(output_path, sizeof(output_path),
             "%s/test_data_set_%d/output_0.pb",
             base_path, i);
    auto loaded_output = Tensor::loadProtoBuf(output_path);
    if(!loaded_output) {
      fprintf(stderr, "Failed to load output %s\n", output_path);
      return 1;
    }

    p->exec();

    const double sse = loaded_output->sse(*output);
    if(sse > 0.001) {
      fprintf(stderr, "  Test-%d: FAILED SSE=%f\n", i, sse);
      return 1;
    }
    printf("  Test-%d: Ok SSE:%f\n", i, sse);
  }

  return 0;
}








int
test_onnx_main(int argc, char **argv)
{
  int opt;
  int verbose = 0;
  while((opt = getopt(argc, argv, "v")) != -1) {
    switch(opt) {
    case 'v':
      verbose++;
      break;
    }
  }

  argc -= optind;
  argv += optind;

  auto ctx = createContext();

  if(argc == 1) {
    return test_one(argv[0], ctx, verbose);
  }

  if(argc != 0) {
    fprintf(stderr, "Usage: onnx <modelpath>\n");
    exit(1);
  }

  if(test_one("models/squeezenet1.1/squeezenet1.1.onnx", ctx, verbose)) {
    exit(1);
  }

  if(test_one("models/resnet50/model.onnx", ctx, verbose)) {
    exit(1);
  }

  return 0;
}
