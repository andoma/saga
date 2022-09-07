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
    void updateCell(Page page, size_t row, size_t column, Align a,
                    const char *fmt, ...) override;

    void refresh() override;

    void maybe_refresh();

    void refresh_locked();

    std::map<int, std::vector<std::pair<std::vector<Cell>, bool>>> m_pages;
    std::vector<size_t> m_col_width;
    bool m_need_layout{false};
    int64_t m_last_refresh{0};
    int m_rewind_rows{0};
    bool m_all_is_dirty{false};
    std::mutex m_mutex;
};

void
TUI::refresh()
{
    std::unique_lock lock{m_mutex};
    m_last_refresh = Now();
    refresh_locked();
}

void
TUI::updateCell(Page page, size_t row, size_t col, Align a, const char *fmt,
                ...)
{
    char tmp[1024];
    va_list ap;
    va_start(ap, fmt);
    size_t len = vsnprintf(tmp, sizeof(tmp), fmt, ap);
    va_end(ap);

    std::unique_lock lock{m_mutex};

    auto &rows = m_pages[(int)page];

    if(rows.size() <= row) {
        rows.resize(row + 1);
        m_all_is_dirty = true;
    }
    auto &cells = rows[row];

    if(cells.first.size() <= col) {
        cells.first.resize(col + 1);
    }

    if(m_col_width.size() <= col) {
        m_col_width.resize(col + 1);
    }

    auto &cell = cells.first[col];
    cells.second = true;
    cell.str = tmp;
    cell.a = a;

    if(len > m_col_width[col]) {
        m_all_is_dirty = true;
        m_col_width[col] = len;
    }

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

    int rewind = 0;

    for(auto &[id, page] : m_pages) {
        int new_section = 1;
        for(auto &row : page) {
            rewind++;
            if(row.second || m_all_is_dirty) {
                // dirty flag
                row.second = false;

                if(new_section) {
                    printf("\033[48:5:233m");
                }
                printf("\033[K");
                for(size_t i = 0; i < row.first.size(); i++) {
                    const auto &cell = row.first[i];
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
                if(new_section) {
                    printf("\033[0m");
                }
            }
            new_section = 0;
            printf("\n");
        }
    }

    m_rewind_rows = rewind;
    m_all_is_dirty = false;
}

std::shared_ptr<UI>
make_tui()
{
    return std::make_shared<TUI>();
}

}  // namespace saga
