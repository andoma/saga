#include "saga.h"


namespace saga {


class NullContext : public Context {
public:

  std::shared_ptr<Program> createProgram(const Graph &graph,
                                         const ProgramConfig &pc);

};


std::shared_ptr<Program>
NullContext::createProgram(const Graph &g,
                            const ProgramConfig &pc)
{
  fprintf(stderr, "Warning: NullContext can't create program\n");
  return nullptr;
}


static std::shared_ptr<Context> (*createContextFn)(void);

std::shared_ptr<Context> createContext()
{
  if(createContextFn)
    return createContextFn();
  return std::make_shared<NullContext>();
}

void
registerContextFactory(std::shared_ptr<Context> (*fn)(void))
{
  createContextFn = fn;
}


}
