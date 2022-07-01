#include "saga.hpp"

struct StatBar : public UI {

    void updateBatchInfo(const char *what, int batch_size,
                         int total_batches) override {

    }
    void updateProgress(int current_batch) override {

    }
    void updateLoss(double loss) override {

    }
    void updateMemUsage(size_t use, size_t total) override {

    }
    void updateMpScaling(double scaling) override {

    }

    int64_t m_last_update(0);
};



std::shared_ptr<UI> make_statbar()
{
    return std::make_shared<StatBar>();
}

