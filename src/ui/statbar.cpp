#include "saga.hpp"

#include <cmath>
#include <map>
#include <cstdarg>

namespace saga {

struct StatBar : public UI {
    struct ProgInfo {
        int64_t m_total_samples{0};
        double m_loss{NAN};
        double m_mp_scaling{NAN};
        std::string m_extra;
        std::string m_name;
        int64_t m_start{0};
    };

    void updateBatchInfo(int batch_size, int total_batches) override
    {
        m_batch_size = batch_size;
        m_total_batches = total_batches;

        for(auto &it : m_pi) {
            it.second.m_start = 0;
        }

        maybe_refresh();
    }

    void updateMemUsage(size_t use, size_t total) override
    {
        m_mem_use = use;
        m_mem_total = total;
        maybe_refresh();
    }

    void updateCurrentBatch(int current_batch) override
    {
        m_current_batch = current_batch;
        maybe_refresh();
    }

    void updateName(int program_index, const std::string &name) override
    {
        auto &pi = m_pi[program_index];
        pi.m_name = name;
    }

    void updateProgress(int program_index, int64_t total_samples) override
    {
        auto &pi = m_pi[program_index];
        if(pi.m_start == 0)
            pi.m_start = Now();
        pi.m_total_samples = total_samples;
    }

    void updateLoss(int program_index, double loss) override
    {
        auto &pi = m_pi[program_index];
        if(std::isfinite(pi.m_loss)) {
            pi.m_loss += (loss - pi.m_loss) * 0.99f;
        } else {
            pi.m_loss = loss;
        }
        maybe_refresh();
    }

    void updateMpScaling(int program_index, double scaling) override
    {
        auto &pi = m_pi[program_index];
        pi.m_mp_scaling = scaling;
        maybe_refresh();
    }

    void updateExtra(int program_index, const std::string &extra) override
    {
        auto &pi = m_pi[program_index];
        pi.m_extra = extra;
        maybe_refresh();
    }

    void maybe_refresh(void)
    {
        int64_t now = Now();
        if(now > m_last_update + 500000) {
            refresh(now);
        }
    }

    void addCell(const char *fmt, ...)
    {
        char buf[512];
        va_list ap;
        va_start(ap, fmt);
        vsnprintf(buf, sizeof(buf), fmt, ap);
        va_end(ap);

        if(m_cells.size() == 0) {
            m_cells.push_back({});
        }

        m_cells[m_cells.size() - 1].push_back(buf);
    }

    void nextRow() { m_cells.push_back({}); }

    void refresh(int64_t now)
    {
        m_last_update = now;
        m_cells.clear();

        addCell("Saga");
        addCell("Batch: %d / %d", m_current_batch, m_total_batches);
        if(m_mem_use)
            addCell("Memory: %zu/%-zu", m_mem_use >> 20, m_mem_total >> 20);

        for(const auto &it : m_pi) {
            nextRow();
            int index = it.first;
            const auto &pi = it.second;

            if(pi.m_name.size()) {
                addCell("%s", pi.m_name.c_str());
            } else {
                addCell("Prog: %d", index);
            }

            if(pi.m_total_samples) {
                const int64_t total_samples = pi.m_total_samples;
                addCell("Samples/s: %6.2f",
                        total_samples * 1e6 / (double)(now - pi.m_start));
            }

            if(std::isfinite(pi.m_loss)) {
                addCell("Loss: %8.4f", pi.m_loss);
            }

            if(std::isfinite(pi.m_mp_scaling)) {
                addCell("MPS: %1.1e", pi.m_mp_scaling);
            }

            if(pi.m_extra.size()) {
                addCell("%s", pi.m_extra.c_str());
            }
        }

        if(m_status_size) {
            printf("\033[%dA", m_status_size);
        }

        m_status_size = m_cells.size();

        size_t maxlen = 0;
        for(size_t i = 0; i < m_cells.size(); i++) {
            const auto &r = m_cells[i];
            maxlen = std::max(maxlen, r.size());
        }

        std::vector<int> colwidth(maxlen);
        for(size_t i = 0; i < m_cells.size(); i++) {
            const auto &r = m_cells[i];
            for(size_t j = 0; j < r.size(); j++) {
                colwidth[j] = std::max(colwidth[j], (int)r[j].size());
            }
        }

        for(size_t i = 0; i < m_cells.size(); i++) {
            const auto &r = m_cells[i];
            printf("\033[K");
            for(size_t j = 0; j < r.size(); j++) {
                printf("%s%-*s", j ? " | " : "", colwidth[j], r[j].c_str());
            }
            printf("\n");
        }
    }

    std::vector<std::vector<std::string>> m_cells;
    size_t m_max_cell_length;

    std::map<int, ProgInfo> m_pi;

    int m_batch_size{1};
    int m_total_batches{0};
    int m_current_batch{0};
    int m_current_epoch{0};
    size_t m_mem_use{0};
    size_t m_mem_total{0};

    int64_t m_last_update{0};

    int m_status_size{0};
};

std::shared_ptr<UI>
make_statbar()
{
    return std::make_shared<StatBar>();
}

}  // namespace saga
