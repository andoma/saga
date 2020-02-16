#include <string.h>
#include <stdio.h>
#include <unistd.h>
#include <pthread.h>

#include "saga.h"
#include "test/cli.h"


using namespace saga;

std::vector<CliCmd> saga::clicmds;

static int
usage(void)
{
  printf("\nUsage: saga [opts] <COMMAND> ...\n\n");
  printf("Available commands:\n\n");
  for(auto &c : clicmds) {
    printf("\t%-30s %s\n", c.argpattern, c.description);
  }
  printf("\n");
  printf("Global options:\n\n");
  printf("\t-u   Start Graphical User Interface\n");
  printf("\n");

  return 1;
}


struct Args {
  int argc;
  char **argv;
  std::shared_ptr<UI> ui;
  CliCmd *cmd;
};



static void *
argthread(void *aux)
{
  Args *a = (Args *)aux;
  a->cmd->fn(a->argc, a->argv, a->ui);
  return NULL;

}


extern int optind;

int
main(int argc, char **argv)
{
  std::shared_ptr<UI> ui;
  argv++;
  argc--;

  if(argc > 1 && !strcmp(argv[1], "-u")) {
    ui = createUI();
    argv++;
    argc--;
  }

  if(argc < 1)
    return usage();

  for(auto &c : clicmds) {
    if(!strcmp(c.cmd, argv[0])) {

      if(ui) {
        auto a = new Args();
        a->argc = argc;
        a->argv = argv;
        a->cmd = &c;
        a->ui = ui;
        pthread_t tid;
        pthread_create(&tid, NULL, argthread, a);
        ui->run();
        exit(0);
      } else {
        return c.fn(argc, argv, nullptr);
      }
    }
  }

  return usage();
}
