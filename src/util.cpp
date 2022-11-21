#include "saga.hpp"

#include <stdarg.h>
#include <stdio.h>

namespace saga {

int64_t
now()
{
    struct timespec tv;
    clock_gettime(CLOCK_MONOTONIC, &tv);
    return (int64_t)tv.tv_sec * 1000000LL + (tv.tv_nsec / 1000);
}

std::string
fmt(const char* fmt, ...)
{
    char* out;
    va_list ap;
    va_start(ap, fmt);
    int len = vasprintf(&out, fmt, ap);
    va_end(ap);

    if(len < 0)
        abort();
    auto r = std::string(out, out + len);
    free(out);
    return r;
}

namespace impl {

struct Barrier : public saga::Barrier {
    Barrier(size_t count) { pthread_barrier_init(&m_barrier, NULL, count); }

    ~Barrier() { pthread_barrier_destroy(&m_barrier); }

    void wait(void) { pthread_barrier_wait(&m_barrier); }

    pthread_barrier_t m_barrier;
};

}  // namespace impl

std::shared_ptr<Barrier>
Barrier::make(size_t count)
{
    return std::make_shared<impl::Barrier>(count);
}

}  // namespace saga
