#include "saga.hpp"

namespace saga {

int64_t
Now()
{
    struct timespec tv;
    clock_gettime(CLOCK_MONOTONIC, &tv);
    return (int64_t)tv.tv_sec * 1000000LL + (tv.tv_nsec / 1000);
}

}  // namespace saga
