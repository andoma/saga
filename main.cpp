#include <string.h>
#include <stdio.h>

#include "saga.hpp"
#include "test/cli.h"

std::vector<saga::CliCmd> saga::clicmds;

static int
usage(void)
{
    printf("\nUsage: saga <cmd> ...\n\n");
    printf("Available commands:\n\n");
    for(auto &c : saga::clicmds) {
        printf("\t%-30s %s\n", c.argpattern, c.description);
    }
    printf("\n");
    return 1;
}

int
main(int argc, char **argv)
{
    if(argc < 2)
        return usage();

    argv += 1;
    argc -= 1;

    for(auto &c : saga::clicmds) {
        if(!strcmp(c.cmd, argv[0])) {
            return c.fn(argc, argv);
        }
    }

    return usage();
}
