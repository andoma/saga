// -*-c++-*-

#include <memory>
#include <vector>
namespace saga {

class Context;

// These are by priority, highest to lowest
enum class ContextType {
  CUDA,
  DNNL,
};

void registerContextFactory(ContextType type,
                            std::shared_ptr<Context> (*fn)(void));

};
