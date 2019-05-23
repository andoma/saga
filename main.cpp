#include "saga.h"




extern int mnist_main(int argc, char **argv);

extern int test_concat_main(int argc, char **argv);


int
main(int argc, char **argv)
{
  if(argc < 2) {
    fprintf(stderr, "Usage %s <cmd> ...\n", argv[0]);
    return 1;
  }

  if(!strcmp(argv[1], "mnist")) {
    argv += 2;
    argc -= 2;
    return mnist_main(argc, argv);
  } else if(!strcmp(argv[1], "concat")) {
    argv += 2;
    argc -= 2;
    return test_concat_main(argc, argv);
  } else {
    fprintf(stderr, "Unknown command: %s\n", argv[1]);
    return 1;
  }
}
