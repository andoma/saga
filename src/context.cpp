#include <map>
#include "saga.h"
#include "context.h"


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

static std::map<ContextType, std::shared_ptr<Context> (*)(void)> allfactories;

std::shared_ptr<Context> createContext()
{
  auto it = allfactories.begin();
  if(it == allfactories.end()) {
    return std::make_shared<NullContext>();
  }
  return it->second();
}


void
registerContextFactory(ContextType type,
                       std::shared_ptr<Context> (*fn)(void))
{
  allfactories[type] = fn;
}


std::vector<std::shared_ptr<Context> (*)(void)> allContextFactories()
{
  std::vector<std::shared_ptr<Context> (*)(void)> r;

  for(const auto &i : allfactories) {
    r.push_back(i.second);
  }
  return r;
}


}
