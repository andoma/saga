// -*-c++-*-

#include "saga.h"

namespace saga {

class TensorStorage {

public:

  typedef double (getfn_t)(const void *base, size_t offset);
  typedef void (setfn_t)(void *base, size_t offset, double value);

  TensorStorage(Tensor::DataType data_type);

  virtual ~TensorStorage() {};

  virtual double get(size_t offset) const {
    return get_(data_, offset);
  }

  virtual void set(size_t offset, double value) {
    set_(data_, offset, value);
  }

  void *data() const {
    return data_;
  }

  getfn_t *get_;
  setfn_t *set_;
  const Tensor::DataType data_type_;
  const size_t element_size_;
protected:
  void *data_ = nullptr;
};


bool copy_tensor(void *dst,
                 int dst_rank,
                 const int *dst_sizes,
                 const int *dst_strides,
                 Tensor::DataType dst_datatype,
                 const Tensor &src,
                 TensorAccess *src_ta);

};
