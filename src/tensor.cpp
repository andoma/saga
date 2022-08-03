/*
 * Copyright (c) 2019, Andreas Smas
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <string.h>
#include <errno.h>
#include <sys/mman.h>

#include <random>
#include <sstream>

#include <inttypes.h>
#include <math.h>

#include "saga.hpp"
#include "tensor.hpp"
#include "fp16.h"

#include "turbo_colormap.h"

namespace saga {

//------------------------------------------------------------------------

Dims::Dims(const std::vector<int> &vec)
{
    for(const auto &v : vec) {
        this->push_back(v);
    }
}

Dim &
Dim::operator++()
{
    if(auto v = std::get_if<int64_t>(&*this)) {
        (*v)++;
    } else {
        fprintf(stderr, "Unable to ++ parameterized dim\n");
        abort();
    }
    return *this;
}

Dim &
Dim::operator+=(const Dim &rhs)
{
    if(auto v = std::get_if<int64_t>(&*this)) {
        (*v) += (int)rhs;
    } else {
        fprintf(stderr, "Unable to += parameterized dim\n");
        abort();
    }
    return *this;
}

std::vector<int32_t>
Dims::i32() const
{
    std::vector<int32_t> r;

    for(auto &d : *this) {
        if(auto v = std::get_if<int64_t>(&d)) {
            r.push_back(*v);
        } else {
            fprintf(stderr,
                    "Unable to convert parameterized dim to vector<int>\n");
            abort();
        }
    }
    return r;
}

std::vector<int64_t>
Dims::i64() const
{
    std::vector<int64_t> r;

    for(auto &d : *this) {
        if(auto v = std::get_if<int64_t>(&d)) {
            r.push_back(*v);
        } else {
            fprintf(stderr,
                    "Unable to convert parameterized dim to vector<int>\n");
            abort();
        }
    }
    return r;
}

Dims
Dims::transform(std::function<Dim(const DimParam &dp, size_t i)> fn) const
{
    Dims r;
    size_t i = 0;
    for(auto &d : *this) {
        if(auto v = std::get_if<int64_t>(&d)) {
            r.push_back(*v);
        } else if(auto v = std::get_if<DimParam>(&d)) {
            r.push_back(fn(*v, i));
        }
        i++;
    }
    return r;
}

Dims
Dims::batch(int64_t N) const
{
    return transform([&](auto dp, size_t i) {
        return dp == DimParam::BATCH_SIZE ? (Dim)N : dp;
    });
}

size_t
Dims::elements(size_t start_rank) const
{
    size_t elements = 1;
    for(size_t i = start_rank; i < size(); i++) {
        elements *= (*this)[i];
    }
    return elements;
}

std::string
Dim::to_string() const
{
    if(auto v = std::get_if<int64_t>(&*this)) {
        return std::to_string(*v);
    } else if(auto v = std::get_if<DimParam>(&*this)) {
        switch(*v) {
        case DimParam::BATCH_SIZE:
            return "N";
        case DimParam::UNCHANGED:
            return "=";
        case DimParam::REDUCE:
            return "R";
        }
    }
    return "?";
}

std::string
Dims::to_string() const
{
    if(size() == 0)
        return "[]";

    const char *prefix = "[";
    std::stringstream ss;
    for(const auto &d : *this) {
        ss << prefix << d.to_string();
        prefix = ", ";
    }
    ss << "]";
    return ss.str();
}

bool
Dims::similar(const Dims &o) const
{
    size_t oskip = 0;
    while(oskip < o.size() && o[oskip] == 1) oskip++;

    size_t tskip = 0;
    while(tskip < size() && (*this)[tskip] == 1) tskip++;

    if(size() - tskip != o.size() - oskip)
        return false;

    for(size_t i = 0; i < std::min(size() - tskip, o.size() - oskip); i++) {
        if(o[oskip + i] != (*this)[tskip + i])
            return false;
    }
    return true;
}
//------------------------------------------------------------------------

static double
get_float(const void *base, size_t offset)
{
    return ((const float *)base)[offset];
}

static double
get_half(const void *base, size_t offset)
{
    return fp16_read((const fp16 *)base + offset);
}

static double
get_u8(const void *base, size_t offset)
{
    return ((const uint8_t *)base)[offset];
}

static double
get_i64(const void *base, size_t offset)
{
    return ((const int64_t *)base)[offset];
}

static double
get_i32(const void *base, size_t offset)
{
    return ((const int32_t *)base)[offset];
}

static double
get_i16(const void *base, size_t offset)
{
    return ((const int16_t *)base)[offset];
}

static void
set_float(void *base, size_t offset, double v)
{
    ((float *)base)[offset] = v;
}

static void
set_half(void *base, size_t offset, double v)
{
    fp16_write((fp16 *)base + offset, v);
}

static void
set_u8(void *base, size_t offset, double v)
{
    ((uint8_t *)base)[offset] = v;
}

static void
set_i64(void *base, size_t offset, double v)
{
    ((int64_t *)base)[offset] = v;
}

static void
set_i32(void *base, size_t offset, double v)
{
    ((int32_t *)base)[offset] = v;
}

static void
set_i16(void *base, size_t offset, double v)
{
    ((int16_t *)base)[offset] = v;
}

TensorStorage::getfn_t *
datatype_get(Tensor::DataType dt)
{
    switch(dt) {
    case Tensor::DataType::U8:
        return &get_u8;
    case Tensor::DataType::HALF:
        return &get_half;
    case Tensor::DataType::FLOAT:
        return &get_float;
    case Tensor::DataType::INT64:
        return &get_i64;
    case Tensor::DataType::I32:
        return &get_i32;
    case Tensor::DataType::I16:
        return &get_i16;
    default:
        abort();
    }
}

TensorStorage::setfn_t *
datatype_set(Tensor::DataType dt)
{
    switch(dt) {
    case Tensor::DataType::U8:
        return &set_u8;
    case Tensor::DataType::HALF:
        return &set_half;
    case Tensor::DataType::FLOAT:
        return &set_float;
    case Tensor::DataType::INT64:
        return &set_i64;
    case Tensor::DataType::I32:
        return &set_i32;
    case Tensor::DataType::I16:
        return &set_i16;
    default:
        abort();
    }
}

const char *
datatype_str(Tensor::DataType dt)
{
    switch(dt) {
    case Tensor::DataType::U8:
        return "u8";
    case Tensor::DataType::HALF:
        return "half";
    case Tensor::DataType::FLOAT:
        return "float";
    case Tensor::DataType::INT64:
        return "i64";
    case Tensor::DataType::I32:
        return "i32";
    case Tensor::DataType::I16:
        return "i16";
    default:
        return "?";
    }
}

const char *
Tensor::DataTypeStr(DataType dt)
{
    return datatype_str(dt);
}

size_t
Tensor::DataTypeSize(DataType dt)
{
    switch(dt) {
    case Tensor::DataType::U8:
        return 1;
    case Tensor::DataType::HALF:
        return 2;
    case Tensor::DataType::FLOAT:
        return 4;
    case Tensor::DataType::INT64:
        return 8;
    case Tensor::DataType::I32:
        return 4;
    case Tensor::DataType::I16:
        return 2;
    default:
        abort();
    }
}

TensorStorage::TensorStorage(Tensor::DataType data_type)
  : m_get(datatype_get(data_type))
  , m_set(datatype_set(data_type))
  , m_data_type(data_type)
  , m_element_size(Tensor::DataTypeSize(data_type))
  , m_data(NULL)
{
}

//------------------------------------------------------------------------

Tensor::Tensor(DataType data_type, const Dims &dims,
               const std::optional<const std::string> &name)
  : name_(name), data_type_(data_type), dims_(dims){};

std::string
Tensor::info() const
{
    std::stringstream ss;
    if(name_) {
        ss << "\"" << *name_ << "\"";
    }
    ss << "<" << datatype_str(data_type_) << ">" << dims_.to_string();
    return ss.str();
}

std::shared_ptr<Tensor>
Tensor::slice(const Dims &offset, const Dims &size)
{
    return nullptr;
}

std::shared_ptr<Tensor>
Tensor::grad(bool create)
{
    return nullptr;
}

std::shared_ptr<Tensor>
Tensor::value() const
{
    return nullptr;
}

static void
print1dTensor(const char *prefix, Tensor &t, TensorAccess &ta)
{
    for(int i = 0; i < t.dims_[0]; i++) {
        printf("%s: [%5d]: %f\n", prefix, (int)i, ta.get({i}));
    }
}

Tensor::Stats
Tensor::stats()
{
    auto ta = access();
    if(ta == nullptr) {
        return Stats({});
    }
    Dims c(dims_.size(), 0);

    const size_t elements = dims_.elements();
    double max = -INFINITY;
    double min = INFINITY;
    double sum = 0;
    double sumsum = 0;

    for(size_t i = 0; i < elements; i++) {
        const double v = ta->get(c);

        max = std::max(max, v);
        min = std::min(min, v);
        sum += v;
        sumsum += v * v;

        for(ssize_t j = c.size() - 1; j >= 0; j--) {
            ++c[j];
            if(c[j] == dims_[j]) {
                c[j] = 0;
            } else {
                break;
            }
        }
    }

    const double mean = sum / elements;
    const double var = (sumsum - sum * sum / elements) / elements;

    return Stats({.min = min, .max = max, .mean = mean, .stddev = sqrt(var)});
}

std::string
Tensor::statsString(void)
{
    auto s = stats();
    char buf[512];
    snprintf(buf, sizeof(buf), "{min:%f mean:%f max:%f stddev:%f}", s.min,
             s.mean, s.max, s.stddev);
    return std::string(buf);
}

void
Tensor::printStats(const char *prefix)
{
    auto s = stats();
    printf("%s: min:%f max:%f mean:%f stddev:%f\n", prefix, s.min, s.max,
           s.mean, s.stddev);
}

void
Tensor::print(const char *prefix, int elements_per_rank)
{
    printf("%s: %s\n", prefix, info().c_str());

    auto ta = access();
    if(ta == nullptr) {
        printf("%s: Abstract (no data)\n", prefix);
        return;
    }

    if(dims_.size() == 1) {
        // We format 1d tensor vertically instead of a long horizontal line
        print1dTensor(prefix, *this, *ta);
        return;
    }

    const size_t rank = dims_.size();
    Dims c(rank, 0);

    const char *lf = "";

    while(1) {
        if(c[rank - 1] == 0) {
            printf("%s%s: [", lf, prefix);
            for(size_t j = 0; j < c.size(); j++) {
                printf("%s%3s", j ? "," : "", c[j].to_string().c_str());
            }
            printf("]");
            lf = "\n";
        }
        printf(" % 1.6f", ta->get(c));

        for(ssize_t j = rank - 1; j >= 0; j--) {
            ++c[j];
            if(c[j] == dims_[j] || c[j] == elements_per_rank) {
                if(j == 0) {
                    printf("%s", lf);
                    return;
                }
                c[j] = 0;
            } else {
                break;
            }
        }
    }
}

void
Tensor::print_anomaly(const char *prefix)
{
    printf("%s: %s\n", prefix, info().c_str());

    auto ta = access();
    if(ta == nullptr) {
        printf("%s: Abstract (no data)\n", prefix);
        return;
    }

    if(dims_.size() == 1) {
        // We format 1d tensor vertically instead of a long horizontal line
        print1dTensor(prefix, *this, *ta);
        return;
    }

    const size_t rank = dims_.size();
    Dims c(rank, 0);

    const char *lf = "";

    while(1) {
        auto x = ta->get(c);

        if(!isfinite(x)) {
            printf("%s: %f {", c.to_string().c_str(), x);
            const uint8_t *u8 = (const uint8_t *)ta->getAddr(c);
            if(u8) {
                for(int i = 0; i < 4; i++) {
                    printf("%02x.", u8[i]);
                }
            }
            printf("}\n");
            break;
        }

        for(ssize_t j = rank - 1; j >= 0; j--) {
            ++c[j];
            if(c[j] == dims_[j]) {
                if(j == 0) {
                    printf("%s", lf);
                    return;
                }
                c[j] = 0;
            } else {
                break;
            }
        }
    }
}

std::shared_ptr<Tensor>
Tensor::toRGB(std::optional<std::pair<float, float>> range)
{
    if(dims_.size() != 4)
        return nullptr;

    const int in = dims_[0];
    const int ic = dims_[1];
    const int ih = dims_[2];
    const int iw = dims_[3];

    float min, max;

    if(range) {
        min = (*range).first;
        max = (*range).second;
    } else {
        auto s = stats();
        min = s.min;
        max = s.max;
    }

    const float offset = -min;
    const float scale = 255.0f / (max - min);

    auto src = access();
    if(src == nullptr)
        return nullptr;

    Dims odims = dims_;

    if(ic > 3) {
        odims.push_back(3);
    } else {
        odims = Dims{in, 1, ih, iw, 3};
    }

    auto ot = makeCPUTensor(Tensor::DataType::U8, odims);
    auto dst = ot->access();

    uint8_t *p = (uint8_t *)dst->data();

    int oc = odims[1];
    for(int n = 0; n < in; n++) {
        for(int c = 0; c < oc; c++) {
            for(int y = 0; y < ih; y++) {
                for(int x = 0; x < iw; x++) {
                    float r, g, b;
                    int i;
                    switch(ic) {
                    case 3:
                        r = (src->get({n, 0, y, x}) + offset) * scale;
                        g = (src->get({n, 1, y, x}) + offset) * scale;
                        b = (src->get({n, 2, y, x}) + offset) * scale;
                        break;
                    case 2:
                        r = (src->get({n, 0, y, x}) + offset) * scale;
                        g = (src->get({n, 1, y, x}) + offset) * scale;
                        b = 0;
                        break;
                    case 1:
                        r = (src->get({n, 0, y, x}) + offset) * scale;
                        g = r;
                        b = r;
                        break;
                    default:
                        r = (src->get({n, c, y, x}) + offset) * scale;
                        i = std::clamp(r, 0.0f, 255.0f);
                        *p++ = turbo_srgb_bytes[i][0];
                        *p++ = turbo_srgb_bytes[i][1];
                        *p++ = turbo_srgb_bytes[i][2];
                        continue;
                    }

                    *p++ = std::clamp(r, 0.0f, 255.0f);
                    *p++ = std::clamp(g, 0.0f, 255.0f);
                    *p++ = std::clamp(b, 0.0f, 255.0f);
                }
            }
        }
    }
    return ot;
}

void
Tensor::printRGB(const char *prefix,
                 std::optional<std::pair<float, float>> range)
{
    printf("%s: %s\n", prefix, info().c_str());

    auto rgb = toRGB(range);
    if(rgb == NULL) {
        printf("%s: Too few dimensions or abstract\n", prefix);
        return;
    }

    const int n = rgb->dims_[0];
    const int c = rgb->dims_[1];
    const int h = rgb->dims_[2];
    const int w = rgb->dims_[3];

    auto ta = rgb->access();
    const auto strides = ta->strides();
    const uint8_t *pixels = (const uint8_t *)ta->data();

    for(int a = 0; a < n; a++) {
        for(int b = 0; b < c; b++) {
            printf("%s: [%d,%d]", prefix, a, b);
            for(int x = 0; x < w; x++) {
                printf("=");
            }
            printf("\n");

            const uint8_t *img = pixels + a * strides[0] + b * strides[1];

            for(int y = 0; y < h; y += 2) {
                printf("%s: [%d,%d]", prefix, a, b);

                const uint8_t *r1 = img + strides[2] * y;
                const uint8_t *r2 = y < h - 1 ? r1 + strides[2] : NULL;

                for(int x = 0; x < w; x++) {
                    if(r2) {
                        const uint8_t r = *r2++;
                        const uint8_t g = *r2++;
                        const uint8_t b = *r2++;
                        printf("\033[48;2;%d;%d;%dm", r, g, b);
                    }
                    const uint8_t r = *r1++;
                    const uint8_t g = *r1++;
                    const uint8_t b = *r1++;
                    printf("\033[38;2;%d;%d;%dm▀", r, g, b);
                }
                printf("\033[0m\n");
            }
        }
    }
}

struct DimInfo {
    int size;
    int dst_stride;
    int src_size;
    int src_stride;
    int src_dim;
    static void reduce(std::vector<DimInfo> &dis);
};

void
DimInfo::reduce(std::vector<DimInfo> &dis)
{
#if 0
  printf("Initial copy plan\n");
  for(size_t i = 0; i < dis.size(); i++) {
    printf("Dim %zd DstStride:%-5d DstSize:%-5d SrcStride:%-5d SrcSize:%-5d\n",
           i, dis[i].dst_stride, dis[i].size, dis[i].src_stride,
           dis[i].src_size);
  }
#endif
    // Any dimensions with a size of 1 which is not the innermost
    // is redundant from a copy perspective
    for(ssize_t i = dis.size() - 2; i >= 0; i--) {
        if(dis[i].size == 1)
            dis.erase(dis.begin() + i);
    }

    // If size*stride == outer dimensions stride, we can merge them
    for(ssize_t i = dis.size() - 1; i > 0; i--) {
        if(dis[i].size * dis[i].dst_stride == dis[i - 1].dst_stride &&
           dis[i].size * dis[i].src_stride == dis[i - 1].src_stride) {
            dis[i - 1].size *= dis[i].size;
            dis[i - 1].dst_stride = dis[i].dst_stride;
            dis[i - 1].src_stride = dis[i].src_stride;

            dis.erase(dis.begin() + i);
        }
    }
#if 0
  printf("Reduced copy plan\n");
  for(size_t i = 0; i < dis.size(); i++) {
    printf("Dim %zd DstStride:%-5d DstSize:%-5d SrcStride:%-5d SrcSize:%-5d\n",
           i, dis[i].dst_stride, dis[i].size, dis[i].src_stride,
           dis[i].src_size);
  }
#endif
}

template <typename T>
static void
copy_tensor_T(T *dst, const T *src, int rank, const DimInfo *di)
{
    const int n = di->size;
    const int src_stride = di->src_stride;
    const int dst_stride = di->dst_stride;
    if(rank == 1) {
        assert(dst_stride == 1);
        if(src_stride == 1 && n >= 64) {
            memcpy(dst, src, n * sizeof(T));
        } else {
            for(int i = 0; i < n; i++) {
                dst[i] = src[i * src_stride];
            }
        }
        return;
    }
    rank--;
    di++;
    for(int i = 0; i < n; i++) {
        copy_tensor_T(dst + i * dst_stride, src + i * src_stride, rank, di);
    }
}

template <typename T>
static void
copy_tensor_T(T *dst, TensorAccess *ta, Dims &selem, int rank,
              const DimInfo *di)
{
    const int n = di->size;
    const int dst_stride = di->dst_stride;
    const int src_dim = di->src_dim;

    if(rank == 1) {
        assert(dst_stride == 1);
        for(int i = 0; i < n; i++) {
            selem[src_dim] = i;
            dst[i] = ta->get(selem);
        }
        return;
    }
    rank--;
    di++;
    for(int i = 0; i < n; i++) {
        selem[src_dim] = i;
        copy_tensor_T(dst + i * dst_stride, ta, selem, rank, di);
    }
}

static void
copy_tensor_half(fp16 *dst, TensorAccess *ta, Dims &selem, int rank,
                 const DimInfo *di)
{
    const int n = di->size;
    const int dst_stride = di->dst_stride;
    const int src_dim = di->src_dim;

    if(rank == 1) {
        assert(dst_stride == 1);
        for(int i = 0; i < n; i++) {
            selem[src_dim] = i;
            fp16_write(dst + i, ta->get(selem));
        }
        return;
    }
    rank--;
    di++;
    for(int i = 0; i < n; i++) {
        selem[src_dim] = i;
        copy_tensor_half(dst + i * dst_stride, ta, selem, rank, di);
    }
}

bool
copy_tensor(void *dst, int dst_rank, const int *dst_sizes,
            const int *dst_strides, Tensor::DataType datatype, const Tensor &t,
            TensorAccess *ta, int broadcast_dimension)
{
    int dst_dim = dst_rank;
    int src_dim = t.dims_.size();
    auto src_strides = ta->strides();

    const int rank = std::max(dst_dim, src_dim);
    std::vector<DimInfo> dis;
    dis.reserve(rank);

    dst_dim -= rank;
    src_dim -= rank;

    for(int i = 0; i < rank; i++, dst_dim++, src_dim++) {
        int dst_size = dst_dim >= 0 ? dst_sizes[dst_dim] : 1;
        int dst_stride = dst_strides[std::max(dst_dim, 0)];
        int src_size = src_dim >= 0 ? (int)t.dims_[src_dim] : 1;
        int src_stride = src_strides[std::max(src_dim, 0)];

        if(broadcast_dimension == dst_dim) {
            if(src_size == 1 && src_stride == dst_stride && dst_size > 1) {
                src_stride = 0;
                src_size = dst_size;
            }
        }

        dis.push_back(DimInfo{dst_size, dst_stride, src_size, src_stride,
                              std::max(src_dim, 0)});
    }

    std::sort(dis.begin(), dis.end(), [](const DimInfo &a, const DimInfo &b) {
        return a.dst_stride > b.dst_stride;
    });

    size_t dst_elements = 1;
    size_t src_elements = 1;
    for(const auto &d : dis) {
        dst_elements *= d.size;
        src_elements *= d.src_size;
    }

    if(src_elements != dst_elements) {
        return false;
    }

    const void *src = ta->data();

    if(src != NULL && t.data_type_ == datatype) {
        DimInfo::reduce(dis);
        // We can copy using recursive strided copies
        switch(datatype) {
        case Tensor::DataType::U8:
            copy_tensor_T((uint8_t *)dst, (const uint8_t *)src, dis.size(),
                          &dis[0]);
            break;
        case Tensor::DataType::HALF:
            copy_tensor_T((uint16_t *)dst, (const uint16_t *)src, dis.size(),
                          &dis[0]);
            break;
        case Tensor::DataType::FLOAT:
        case Tensor::DataType::I32:
            copy_tensor_T((uint32_t *)dst, (const uint32_t *)src, dis.size(),
                          &dis[0]);
            break;
        case Tensor::DataType::INT64:
            copy_tensor_T((uint64_t *)dst, (const uint64_t *)src, dis.size(),
                          &dis[0]);
            break;
        default:
            fprintf(stderr, "%s can't handle %s\n", __FUNCTION__,
                    datatype_str(datatype));
            return false;
        }
        return true;
    }

    Dims selem(t.dims_.size(), 0);
    switch(datatype) {
    case Tensor::DataType::U8:
        copy_tensor_T((uint8_t *)dst, ta, selem, dis.size(), &dis[0]);
        break;
    case Tensor::DataType::HALF:
        copy_tensor_half((fp16 *)dst, ta, selem, dis.size(), &dis[0]);
        break;
    case Tensor::DataType::FLOAT:
        copy_tensor_T((float *)dst, ta, selem, dis.size(), &dis[0]);
        break;
    case Tensor::DataType::I32:
        copy_tensor_T((int32_t *)dst, ta, selem, dis.size(), &dis[0]);
        break;
    case Tensor::DataType::INT64:
        copy_tensor_T((int64_t *)dst, ta, selem, dis.size(), &dis[0]);
        break;
    default:
        fprintf(stderr, "%s can't handle %s\n", __FUNCTION__,
                datatype_str(datatype));
        return false;
    }
    return true;
}

void
Tensor::copyFrom(Tensor &t)
{
    auto dst = access();
    auto src_ta = t.access();
    if(!copy_tensor(dst->data(), dims_.size(), &dims_.i32()[0],
                    &dst->strides().i32()[0], data_type_, t, src_ta.get())) {
        fprintf(stderr,
                "Tensor copy failed\n"
                "From: %s\n"
                "  To: %s\n",
                t.info().c_str(), info().c_str());
        abort();
    }
}

double
Tensor::sse(Tensor &t)
{
    auto a = t.access();
    auto b = access();
    if(a == nullptr && b == nullptr)
        return 0.0;

    if(a == nullptr || b == nullptr)
        return INFINITY;

    Dims c_a(t.dims_.size(), 0);
    Dims c_b(dims_.size(), 0);

    const size_t elements = dims_.elements();
    assert(elements == t.dims_.elements());

    double r = 0;
    for(size_t i = 0; i < elements; i++) {
        double v = a->get(c_a) - b->get(c_b);
        r += v * v;

        for(ssize_t j = c_a.size() - 1; j >= 0; j--) {
            ++c_a[j];
            if(c_a[j] == t.dims_[j]) {
                c_a[j] = 0;
            } else {
                break;
            }
        }

        for(ssize_t j = c_b.size() - 1; j >= 0; j--) {
            ++c_b[j];
            if(c_b[j] == dims_[j]) {
                c_b[j] = 0;
            } else {
                break;
            }
        }
    }
    return r;
}

std::optional<const std::string>
Tensor::namePostfix(const std::string &postfix) const
{
    return name_ ? std::make_optional(*name_ + "." + postfix) : std::nullopt;
}

//------------------------------------------------------------------------

class EmptyTensor : public DifferentiableTensor {
public:
    EmptyTensor(DataType data_type, const Dims &size,
                const std::optional<const std::string> &name = std::nullopt)
      : DifferentiableTensor(data_type, size, name)
    {
    }

    virtual std::shared_ptr<Tensor> grad(bool create) override
    {
        if(m_gradient || !create)
            return m_gradient;

        m_gradient = std::make_shared<DifferentiableTensor>(
            data_type_, dims_, namePostfix("grad"));
        m_gradient->m_value = shared_from_this();
        return m_gradient;
    }
};

std::shared_ptr<Tensor>
makeTensor(Tensor::DataType data_type, const Dims &size,
           const std::optional<const std::string> &name)
{
    return std::make_shared<EmptyTensor>(data_type, size, name);
}

//------------------------------------------------------------------------

class GenTensorAccess : public TensorAccess {
public:
    GenTensorAccess(size_t rank, double mean, double stddev)
      : rank_(rank), distribution_(mean, stddev)
    {
    }

    Dims strides() { return Dims(rank_, 0); }

    void *data() { return NULL; }

    virtual double get(const Dims &element)
    {
        return distribution_(generator_);
    };

    virtual void set(const Dims &element, double value) {}

    virtual void copyBytesFrom(const Dims &element, const void *data,
                               size_t size)
    {
    }

    const size_t rank_;

    mutable std::normal_distribution<double> distribution_;
    mutable std::default_random_engine generator_;
};

class GenTensor : public Tensor {
public:
    GenTensor(DataType data_type, const Dims &size,
              const std::optional<std::string> &name, double mean,
              double stddev)
      : Tensor(data_type, size, name), mean_(mean), stddev_(stddev)
    {
    }

    std::unique_ptr<TensorAccess> access()
    {
        return std::make_unique<GenTensorAccess>(dims_.size(), mean_, stddev_);
    }

    std::shared_ptr<Tensor> slice(const Dims &offset, const Dims &size)
    {
        return std::make_shared<GenTensor>(data_type_, size, name_, mean_,
                                           stddev_);
    }

    std::string info() const
    {
        std::stringstream ss;
        ss << Tensor::info();
        ss << "(mean:" << mean_ << ", stddev:" << stddev_ << ")";
        return ss.str();
    }

    const double mean_;
    const double stddev_;
};

std::shared_ptr<Tensor>
Tensor::find(Tensor::DataType data_type, const Dims &size, double init_mean,
             double init_stddev, Tensors &named_tensors,
             const std::optional<const std::string> &name)
{
    if(name) {
        auto it = named_tensors.find(*name);
        if(it != named_tensors.end()) {
            auto t = it->second;

            if(t->data_type_ != data_type) {
                fprintf(stderr,
                        "Pre-initialized tensor %s datatype mismatch: "
                        "Loaded:%s != Requested:%s\n",
                        (*name).c_str(), Tensor::DataTypeStr(t->data_type_),
                        Tensor::DataTypeStr(data_type));
                exit(1);
            }

            if(!t->dims_.similar(size)) {
                fprintf(stderr,
                        "Pre-initialized tensor %s dimensions mismatch: "
                        "Loaded:%s != Requested:%s\n",
                        (*name).c_str(), t->dims_.to_string().c_str(),
                        size.to_string().c_str());
                exit(1);
            }
            return t;
        }
    }

    auto t = std::make_shared<GenTensor>(data_type, size, name, init_mean,
                                         init_stddev);

    if(name)
        named_tensors[*name] = t;

    return t;
}

std::shared_ptr<Tensor>
Tensor::make(Tensor::DataType data_type, const Dims &size, double init_mean,
             double init_stddev)
{
    return std::make_shared<GenTensor>(data_type, size, std::nullopt, init_mean,
                                       init_stddev);
}

//------------------------------------------------------------------------

class HostTensorStorage : public TensorStorage {
    const size_t buffer_size_;
    void *mmaped_;

public:
    HostTensorStorage(Tensor::DataType data_type, const Dims &size,
                      const Dims &strides)
      : TensorStorage(data_type)
      , buffer_size_((size.size() ? (size[0] * strides[0]) : 1) *
                     Tensor::DataTypeSize(data_type))
      , mmaped_(NULL)
    {
        m_data = calloc(1, buffer_size_);
    }

    HostTensorStorage(Tensor::DataType data_type, const Dims &size,
                      const Dims &strides, void *mmaped_memory,
                      size_t buffer_size, void *data)
      : TensorStorage(data_type)
      , buffer_size_(buffer_size)
      , mmaped_(mmaped_memory)
    {
        m_data = data;
    }

    ~HostTensorStorage()
    {
        if(mmaped_) {
            munmap(mmaped_, buffer_size_);
        } else {
            free(m_data);
        }
    }
};

//------------------------------------------------------------------------

class CPUTensorAccess : public TensorAccess {
public:
    CPUTensorAccess(const Dims &strides, std::shared_ptr<TensorStorage> storage,
                    int64_t offset)
      : strides_(strides), storage_(storage), offset_(offset)
    {
    }

    ~CPUTensorAccess() {}

    Dims strides() { return strides_; }

    void *data()
    {
        assert(offset_ == 0);
        return storage_->data();
    }

    size_t offsetForElement(const Dims &element) const
    {
        size_t offset = offset_;
        for(size_t i = 0; i < element.size() && i < strides_.size(); i++) {
            offset += element[i] * strides_[i];
        }
        return offset;
    }

    virtual double get(const Dims &element)
    {
        return storage_->get(offsetForElement(element));
    };

    virtual void set(const Dims &element, double value)
    {
        storage_->set(offsetForElement(element), value);
    }

    virtual void *getAddr(const Dims &element)
    {
        const size_t o = offsetForElement(element) * storage_->m_element_size;
        char *p = (char *)storage_->data();
        return (void *)(p + o);
    };

    virtual void copyBytesFrom(const Dims &element, const void *data,
                               size_t size)
    {
        const size_t o = offsetForElement(element) * storage_->m_element_size;
        char *dst = (char *)storage_->data();
        memcpy(dst + o, data, size);
    }

    const Dims strides_;
    const std::shared_ptr<TensorStorage> storage_;
    int64_t offset_;
};

//------------------------------------------------------------------------

static Dims
computeCPUStrides(const Dims &dims)
{
    Dims strides;
    int stride = 1;
    for(int i = dims.size() - 1; i >= 0; i--) {
        strides.insert(strides.begin(), stride);
        stride *= dims[i];
    }
    return strides;
}

class CPUTensor : public DifferentiableTensor {
public:
    CPUTensor(DataType data_type, const Dims &size,
              const std::optional<std::string> &name)
      : DifferentiableTensor(data_type, size, name)
      , strides_(computeCPUStrides(size))
      , storage_(std::make_shared<HostTensorStorage>(data_type, size, strides_))
      , offset_(0)
    {
    }

    CPUTensor(DataType data_type, const Dims &size, const Dims &strides,
              const std::optional<std::string> &name)
      : DifferentiableTensor(data_type, size, name)
      , strides_(strides)
      , storage_(std::make_shared<HostTensorStorage>(data_type, size, strides_))
      , offset_(0)
    {
    }

    CPUTensor(const Dims &size, const Dims &strides,
              std::shared_ptr<TensorStorage> storage, int64_t offset,
              const std::optional<std::string> &name)
      : DifferentiableTensor(storage->m_data_type, size, name)
      , strides_(strides)
      , storage_(storage)
      , offset_(offset)
    {
    }

    std::unique_ptr<TensorAccess> access()
    {
        return std::make_unique<CPUTensorAccess>(strides_, storage_, offset_);
    }

    std::shared_ptr<Tensor> slice(const Dims &offset, const Dims &size);

    virtual std::string info() const;

    const Dims strides_;
    const std::shared_ptr<TensorStorage> storage_;
    const int64_t offset_;
};

std::shared_ptr<Tensor>
CPUTensor::slice(const Dims &offset, const Dims &size)
{
    int64_t o = offset_;

    for(size_t i = 0; i < strides_.size() && i < offset.size(); i++) {
        o += offset[i] * strides_[i];
    }

    return std::make_shared<CPUTensor>(size, strides_, storage_, o,
                                       namePostfix("slice"));
}

std::string
CPUTensor::info() const
{
    std::stringstream ss;

    ss << Tensor::info();
    const char *prefix = "{";
    for(const auto &x : strides_) {
        ss << prefix << x;
        prefix = ", ";
    }
    ss << "}";
    return ss.str();
}

std::shared_ptr<Tensor>
makeCPUTensor(Tensor::DataType data_type, const Dims &size,
              const std::optional<const std::string> &name)
{
    return std::make_shared<CPUTensor>(data_type, size, name);
}

//------------------------------------------------------------------------
// Raw tensor disk IO.

typedef enum {
    TENSOR_DISK_FLOAT = 0,
    TENSOR_DISK_HALF = 1,
} TensorDiskType;

struct TensorDiskHeader {
    uint8_t magic[8];
    TensorDiskType type;
    unsigned int rank;
} __attribute__((packed));

std::shared_ptr<Tensor>
Tensor::load(const char *path, const std::optional<const std::string> &name)
{
    int fd = open(path, O_RDONLY);
    if(fd == -1) {
        fprintf(stderr, "Unable to open %s -- %s\n", path, strerror(errno));
        return nullptr;
    }

    struct stat st;
    if(fstat(fd, &st)) {
        fprintf(stderr, "Unable to stat %s -- %s\n", path, strerror(errno));
        close(fd);
        return nullptr;
    }

    if((size_t)st.st_size < sizeof(TensorDiskHeader)) {
        fprintf(stderr, "Unable to load %s -- Not a saga tensor file\n", path);
        close(fd);
        return nullptr;
    }

    void *mem = mmap(NULL, st.st_size, PROT_READ, MAP_SHARED, fd, 0);
    if(mem == MAP_FAILED) {
        fprintf(stderr, "Unable to load %s -- Out of memory\n", path);
        close(fd);
        return nullptr;
    }
    close(fd);

    const TensorDiskHeader *tdh = (const TensorDiskHeader *)mem;
    if(memcmp(tdh->magic, "sagaT001", 8)) {
        fprintf(stderr, "Unable to load %s -- Not a saga tensor file\n", path);
        munmap(mem, st.st_size);
        return nullptr;
    }

    Tensor::DataType data_type;

    switch(tdh->type) {
    case TENSOR_DISK_FLOAT:
        data_type = Tensor::DataType::FLOAT;
        break;
    case TENSOR_DISK_HALF:
        data_type = Tensor::DataType::HALF;
        break;
    default:
        fprintf(stderr, "Unable to load %s -- Unsupported data type:%d\n", path,
                tdh->type);
        munmap(mem, st.st_size);
        return nullptr;
    }

    if(tdh->rank > 8) {
        fprintf(stderr, "Unable to load %s -- Rank %d too high\n", path,
                tdh->rank);
        munmap(mem, st.st_size);
        return nullptr;
    }

    if((size_t)st.st_size <
       sizeof(TensorDiskHeader) + tdh->rank * sizeof(int)) {
        fprintf(stderr, "Unable to load %s -- File too short\n", path);
        munmap(mem, st.st_size);
        return nullptr;
    }

    Dims dims;
    const unsigned int *d =
        (unsigned int *)((char *)mem + sizeof(TensorDiskHeader));

    for(unsigned int i = 0; i < tdh->rank; i++) {
        dims.push_back(*d++);
    }

    auto strides = computeCPUStrides(dims);

    // Ownership of mmap:ed region 'mem' is transfered to HostTensorStorage
    auto storage = std::make_shared<HostTensorStorage>(
        data_type, dims, strides, mem, st.st_size, (void *)d);

    return std::make_shared<CPUTensor>(dims, strides, storage, 0, name);
}

bool
Tensor::save(const char *path)
{
    TensorDiskHeader tdh;
    memcpy(&tdh.magic, "sagaT001", 8);

    switch(data_type_) {
    case Tensor::DataType::FLOAT:
        tdh.type = TENSOR_DISK_FLOAT;
        break;
    case Tensor::DataType::HALF:
        tdh.type = TENSOR_DISK_HALF;
        break;
    default:
        fprintf(stderr, "Unable to load %s -- Unsupported data type:%d\n", path,
                (int)data_type_);
        return false;
    }

    int fd = open(path, O_CREAT | O_TRUNC | O_WRONLY, 0644);
    if(fd == -1) {
        fprintf(stderr, "Unable to save %s -- %s\n", path, strerror(errno));
        return false;
    }

    tdh.rank = dims_.size();

    if(write(fd, &tdh, sizeof(tdh)) != sizeof(tdh)) {
        fprintf(stderr, "Unable to save %s -- %s\n", path, strerror(errno));
        close(fd);
        return false;
    }

    uint32_t ondiskdims[tdh.rank];
    for(unsigned int i = 0; i < tdh.rank; i++) {
        ondiskdims[i] = dims_[i];
    }

    if(write(fd, &ondiskdims[0], sizeof(int) * tdh.rank) !=
       (int)(sizeof(int) * tdh.rank)) {
        fprintf(stderr, "Unable to save %s -- %s\n", path, strerror(errno));
        close(fd);
        return false;
    }

    CPUTensor copy(data_type_, dims_, std::nullopt);
    copy.copyFrom(*this);

    auto ta = copy.access();

    size_t size =
        copy.strides_[0] * copy.dims_[0] * Tensor::DataTypeSize(data_type_);

    if(write(fd, ta->data(), size) != (int)size) {
        fprintf(stderr, "Unable to save %s -- %s\n", path, strerror(errno));
        close(fd);
        return false;
    }
    close(fd);
    return true;
}

}  // namespace saga
