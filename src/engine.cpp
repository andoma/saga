#include <map>
#include <string>
#include <memory>

#include "saga.hpp"
#include "engine.hpp"

namespace saga {

// static std::map<int, std::shared_ptr<Context> (*)(void)> allfactories;

static std::map<std::string,
                std::shared_ptr<Engine> (*)(const std::shared_ptr<UI> &ui)>
    allfactories;

std::shared_ptr<Engine>
createEngine(const std::shared_ptr<UI> &ui)
{
    auto it = allfactories.begin();
    if(it == allfactories.end()) {
        abort();
    }
    return it->second(ui);
}

void
registerEngineFactory(const char *name, std::shared_ptr<Engine> (*fn)(
                                            const std::shared_ptr<UI> &ui))
{
    allfactories[name] = fn;
}

std::shared_ptr<Context>
createContext(const std::shared_ptr<UI> &ui)
{
    auto e = createEngine(ui ? ui : make_nui());
    auto ctxs = e->createContexts(false);
    return ctxs[0];
}

}  // namespace saga
