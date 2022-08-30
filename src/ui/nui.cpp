#include "saga.hpp"

namespace saga {

struct NUI : public UI {
    void updateCell(size_t row, size_t column, Align a, const char *fmt,
                    ...) override
    {
    }

    size_t alloc_row(size_t count) override { return 0; }

    void refresh(void) override{};
};

std::shared_ptr<UI>
make_nui()
{
    return std::make_shared<NUI>();
}

}  // namespace saga
