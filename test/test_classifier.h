// -*-c++-*-

#include "saga.h"


namespace saga {

void
test_classifier(int argc, char **argv,
                std::shared_ptr<Tensor> input,
                float input_range,
                int output_labels,
                size_t train_inputs,
                size_t test_inputs,
                std::function<void(void)> epoch_begin,
                std::function<void(Tensor &x, Tensor &dy, size_t i)> load_train,
                std::function<void(Tensor &x, int *labels, size_t i)> load_test,
                std::shared_ptr<UI> ui);
}
