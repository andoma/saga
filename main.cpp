#include <string.h>

#include "saga.h"




extern int mnist_main(int argc, char **argv);

extern int test_concat_main(int argc, char **argv);

extern int test_onnx_main(int argc, char **argv);

extern int test_fc_main(int argc, char **argv);

extern int util_showtensors(int argc, char **argv);



int
main(int argc, char **argv)
{
  if(argc < 2) {
    fprintf(stderr, "Usage %s <cmd> ...\n", argv[0]);
    return 1;
  }

  if(!strcmp(argv[1], "onnx")) {
    argv += 1;
    argc -= 1;
    return test_onnx_main(argc, argv);
  } else if(!strcmp(argv[1], "mnist")) {
    argv += 1;
    argc -= 1;
    return mnist_main(argc, argv);
  } else {
    fprintf(stderr, "Unknown command: %s\n", argv[1]);
    return 1;
  }
}
