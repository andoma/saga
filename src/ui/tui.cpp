#include "saga.hpp"

#include <cmath>
#include <map>
#include <cstdarg>
#include <mutex>

namespace saga {

struct Cell {
    UI::Align a{UI::Align::LEFT};
    std::string str;
};

struct TUI : public UI {
    void updateCell(size_t row, size_t column, Align a, const char *fmt,
                    ...) override;

    size_t alloc_row(void) override;

    void refresh() override;

    void maybe_refresh();

    void refresh_locked();

    std::vector<std::vector<Cell>> m_rows;
    std::vector<size_t> m_col_width;
    bool m_need_layout{false};
    int64_t m_last_refresh{0};
    int m_rewind_rows{0};

    size_t m_rowgen{0};

    std::mutex m_mutex;
};

void
TUI::refresh()
{
    std::unique_lock lock{m_mutex};
    m_last_refresh = Now();
    refresh_locked();
}

size_t
TUI::alloc_row(void)
{
    std::unique_lock lock{m_mutex};
    return m_rowgen++;
}

void
TUI::updateCell(size_t row, size_t col, Align a, const char *fmt, ...)
{
    char tmp[1024];
    va_list ap;
    va_start(ap, fmt);
    size_t len = vsnprintf(tmp, sizeof(tmp), fmt, ap);
    va_end(ap);

    std::unique_lock lock{m_mutex};

    if(m_rows.size() <= row) {
        m_rows.resize(row + 1);
    }
    auto &r = m_rows[row];

    if(r.size() <= col) {
        r.resize(col + 1);
    }

    if(m_col_width.size() <= col) {
        m_col_width.resize(col + 1);
    }

    auto &c = r[col];
    c.str = tmp;
    c.a = a;

    m_col_width[col] = std::max(m_col_width[col], len);

    maybe_refresh();
}

void
TUI::maybe_refresh()
{
    int64_t now = Now();
    if(now < m_last_refresh + 250000)
        return;
    m_last_refresh = now;
    refresh_locked();
}

void
TUI::refresh_locked()
{
    if(m_rewind_rows) {
        printf("\033[%dA", m_rewind_rows);
    }

    for(const auto &row : m_rows) {
        printf("\033[K");
        for(size_t i = 0; i < row.size(); i++) {
            const auto &cell = row[i];
            int width = m_col_width[i] + 2;

            switch(cell.a) {
            case Align::LEFT:
                printf("%-*.*s", width, width, cell.str.c_str());
                break;
            case Align::RIGHT:
                printf("%*.*s", width, width, cell.str.c_str());
                break;
            case Align::CENTER:
                printf("%*.*s", width, width, cell.str.c_str());
                break;
            }
        }
        printf("\n");
    }

    m_rewind_rows = m_rows.size();
}

std::shared_ptr<UI>
make_tui()
{
    return std::make_shared<TUI>();
}

}  // namespace saga
