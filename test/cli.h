// -*-c++-*-

namespace saga {

struct CliCmd {
  const char *cmd;
  const char *argpattern;
  const char *description;
  int (*fn)(int argc, char **argv);
};

extern std::vector<CliCmd> clicmds;

};


#define SAGA_CLI_CMD(cmd, argpattern, desc, fn)                     \
  static void init_cli_cmd(void) __attribute__((constructor));      \
  static void init_cli_cmd(void) {                                  \
    clicmds.push_back(CliCmd{cmd, argpattern, desc, fn});           \
  }
