// -*-c++-*-

#include <memory>

namespace saga {
class Context;

void registerContextFactory(std::shared_ptr<Context> (*fn)(void));

};
