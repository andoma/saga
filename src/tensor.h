// -*-c++-*-

#include "saga.h"

namespace saga {

class TensorStorageAccess {

public:

  typedef double (getfn_t)(const void *base, size_t offset);
  typedef void (setfn_t)(void *base, size_t offset, double value);

  TensorStorageAccess(Tensor::DataType data_type);

  double get(size_t offset) const {
    return get_(data_, offset);
  }

  void set(size_t offset, double value) {
    set_(data_, offset, value);
  }

  const getfn_t *get_;
  const setfn_t *set_;
  const Tensor::DataType data_type_;
  const size_t element_size_;
  void *data_;
};



};
