// -*-c++-*-

#include "saga.hpp"

namespace saga {

class DifferentiableTensor
  : public Tensor,
    public std::enable_shared_from_this<DifferentiableTensor> {
public:
    DifferentiableTensor(
        DataType data_type, const Dims &size,
        const std::optional<const std::string> &name = std::nullopt)
      : Tensor(data_type, size, name)
    {
    }

    virtual std::shared_ptr<Tensor> grad(bool create) override
    {
        return m_gradient;
    }

    virtual std::shared_ptr<Tensor> value() const override
    {
        return m_value.lock();
    }

    std::shared_ptr<DifferentiableTensor> m_gradient;

    std::weak_ptr<DifferentiableTensor> m_value;
};

class TensorStorage {
public:
    typedef double(getfn_t)(const void *base, size_t offset);
    typedef void(setfn_t)(void *base, size_t offset, double value);

    TensorStorage(Tensor::DataType data_type);

    virtual ~TensorStorage(){};

    virtual double get(size_t offset) const { return m_get(m_data, offset); }

    virtual void set(size_t offset, double value)
    {
        m_set(m_data, offset, value);
    }

    void *data() const { return m_data; }

    getfn_t *m_get;
    setfn_t *m_set;
    const Tensor::DataType m_data_type;
    const size_t m_element_size;

protected:
    void *m_data = nullptr;
};

bool copy_tensor(void *dst, int dst_rank, const int *dst_sizes,
                 const int *dst_strides, Tensor::DataType dst_datatype,
                 const Tensor &src, TensorAccess *src_ta,
                 int dst_broadcast_dimension = -1);

};  // namespace saga
