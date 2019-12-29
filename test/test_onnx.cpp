#include <limits.h>
#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>

#include "saga.h"

using namespace saga;



static int
test_one(const char *base_path, const char *model_name, int num_tests,
         std::shared_ptr<Context> ctx)
{
  char input_path[PATH_MAX];
  char output_path[PATH_MAX];
  char model_path[PATH_MAX];


  snprintf(model_path, sizeof(model_path), "%s/%s", base_path, model_name);

  auto g = Graph::load(model_path);
  if(g == NULL) {
    fprintf(stderr, "Failed to load model graph %s\n", model_path);
    return 1;
  }

  auto p = ctx->createProgram(*g, ProgramType::INFERENCE, 1);

  auto input = *p->inputs_.begin();
  auto output = *p->outputs_.begin();

  printf("Test: %s\n", base_path);
  printf("  INPUT: %s\n", input->info().c_str());
  printf("  OUTPUT: %s\n", output->info().c_str());

  for(int i = 0; i < num_tests; i++) {
    snprintf(input_path, sizeof(input_path), "%s/test_data_set_%d/input_0.pb",
             base_path, i);

    auto loaded_input = Tensor::load(input_path);
    if(!loaded_input) {
      fprintf(stderr, "Failed to load input %s\n", input_path);
      return 1;
    }

    input->copyFrom(*loaded_input);

    snprintf(output_path, sizeof(output_path),
             "%s/test_data_set_%d/output_0.pb",
             base_path, i);
    auto loaded_output = Tensor::load(output_path);
    if(!loaded_output) {
      fprintf(stderr, "Failed to load output %s\n", output_path);
      return 1;
    }

    p->exec();

    const double sse = loaded_output->sse(*output);
    if(sse > 0.001) {
      fprintf(stderr, "  Test-%d: Failed sse=%f\n", i, sse);
      return 1;
    }
    printf("  Test-%d: OK\n", i);
  }

  return 0;
}








int
test_onnx_main(int argc, char **argv)
{
  auto ctx = createContext();

  if(argc == 3) {
    return test_one(argv[0], argv[1], atoi(argv[2]), ctx);
  }

  if(argc != 0) {
    fprintf(stderr, "Usage: onnx <basepath> <modelname> <num_test_datas>\n");
    exit(1);
  }

  if(test_one("models/squeezenet1.1", "squeezenet1.1.onnx", 3, ctx)) {
    exit(1);
  }

  if(test_one("models/resnet50", "model.onnx", 9, ctx)) {
    exit(1);
  }

  return 0;
}
