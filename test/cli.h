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

#define SAGA_CLI_CMD_GLUE(x, y) x ## y
#define SAGA_CLI_CMD_GLUE2(x, y) SAGA_CLI_CMD_GLUE(x, y)

#define SAGA_CLI_CMD(cmd, argpattern, desc, fn)                        \
  static void  __attribute__((constructor))                            \
  SAGA_CLI_CMD_GLUE2(init_cli_cmd, __COUNTER__)(void) {                \
    clicmds.push_back(CliCmd{cmd, argpattern, desc, fn});              \
  }

