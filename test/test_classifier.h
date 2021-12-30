// -*-c++-*-

#include "saga.hpp"


namespace saga {

void
test_classifier(int argc, char **argv,
                std::shared_ptr<Tensor> input,
                float input_range,
                int output_labels,
                size_t train_inputs,
                size_t test_inputs,
                std::function<void(int batch_size, bool test)> epoch_begin,
                std::function<void(TensorAccess &, long batch)> load_inputs,
                std::function<int(long index)> get_labels);

}
