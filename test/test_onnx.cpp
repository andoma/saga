#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>

#include "saga.h"

using namespace saga;



static int
test_one(const char *base_path,
         const char *model_name,
         const char *input_name,
         int num_tests)
{
  //  char input_path[PATH_MAX];
  //  char output_path[PATH_MAX];
  char model_path[PATH_MAX];

#if 0
  Network n(false); // No backprop

  snprintf(input_path, sizeof(input_path), "%s/test_data_set_0/input_0.pb",
           base_path);
  auto input = Tensor::createFromPB(input_path);
  if(!input) {
    fprintf(stderr, "Failed to load %s\n", input_path);
    return 1;
  }

  auto inputLayer = n.nameLayer(n.addLayer(makeInput(input.get())),
                                input_name);

#endif

  snprintf(model_path, sizeof(model_path), "%s/%s", base_path, model_name);

  auto g = Graph::load(model_path);
  if(g == NULL) {
    fprintf(stderr, "Failed to load model graph %s\n", model_path);
    return 1;
  }

  //  g->resolve();
  exit(0);
#if 0
  for(int i = 0; i < num_tests; i++) {
    snprintf(input_path, sizeof(input_path), "%s/test_data_set_%d/input_0.pb",
             base_path, i);

    auto loaded_input = Tensor::createFromPB(input_path);
    if(!loaded_input) {
      fprintf(stderr, "Failed to load input %s\n", input_path);
      return 1;
    }
    input->copyFrom(*loaded_input);

    snprintf(output_path, sizeof(output_path),
             "%s/test_data_set_%d/output_0.pb",
             base_path, i);
    auto loaded_output = Tensor::createFromPB(output_path);
    if(!loaded_output) {
      fprintf(stderr, "Failed to load output %s\n", output_path);
      return 1;
    }

    n.forward(false);

    auto out = n.layers_[n.layers_.size() - 1];
    float diff = out->output()->compare(*loaded_output);

    printf("Test data %d max tensor diff: %f\n", i, diff);

    if(diff > 0.001) {
      out->output()->dump("OUTPUT");
      loaded_output->dump("REFERENCE");
      return -1;
    }
  }
#endif

  return 0;
}








int
test_onnx_main(int argc, char **argv)
{
  if(argc == 4) {
    return test_one(argv[0], argv[1], argv[2], atoi(argv[3]));
  }

  if(argc != 0) {
    fprintf(stderr, "Usage: onnx <basepath> <modelname> <inputname> <num_test_datas>\n");
    exit(1);
  }

  if(test_one("models/resnet50", "model.onnx", "gpu_0/data_0", 9)) {
    exit(1);
  }

  if(test_one("models/squeezenet1.1", "squeezenet1.1.onnx", "data", 3)) {
    exit(1);
  }

  return 0;
}
