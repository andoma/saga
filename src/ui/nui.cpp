#include "saga.hpp"

namespace saga {

struct NUI : public UI {
    void updateCell(size_t row, size_t column, Align a, const char *fmt,
                    ...) override;

    size_t alloc_row(void) override;
};

size_t
NUI::alloc_row(void)
{
    return 0;
}

void
NUI::updateCell(size_t row, size_t col, Align a, const char *fmt, ...)
{
}

std::shared_ptr<UI>
make_nui()
{
    return std::make_shared<NUI>();
}

}  // namespace saga
