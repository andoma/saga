#include "saga.hpp"

namespace saga {

struct NUI : public UI {
    void updateCell(Page page, size_t row, size_t column, Align a,
                    const char *fmt, ...) override
    {
    }

    void refresh(void) override{};
};

std::shared_ptr<UI>
make_nui()
{
    return std::make_shared<NUI>();
}

}  // namespace saga
