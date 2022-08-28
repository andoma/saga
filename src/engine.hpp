#include <memory>
#include <vector>

namespace saga {

void registerEngineFactory(
    const char *name,
    std::shared_ptr<Engine> (*fn)(const std::shared_ptr<UI> &ui));

};  // namespace saga
