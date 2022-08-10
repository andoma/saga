#include "saga.hpp"

#include <set>
#include <thread>
#include <mutex>
#include <sys/uio.h>
#include <sys/socket.h>
#include <errno.h>
#include <string.h>
#include <poll.h>
#include <netinet/in.h>
#include <unistd.h>

namespace saga {

#define HEADER_TYPE_TENSOR 1
#define HEADER_TYPE_SYNC   2

struct Header {
    uint32_t type;
    uint32_t payload_length;
    char id[30];
    char datatype;
    uint8_t rank;
    int32_t sizes[4];
};

class TCPPublisher : public Publisher {
public:
    TCPPublisher(int sock);
    ~TCPPublisher();

    void publish(const char *id, Tensor &t, const Dims &offset,
                 TensorAccess *ta) override;

    void sync() override;

    int m_accept_sock;

    std::set<int> m_subscribers;
    std::thread m_thread;
    std::mutex m_mutex;

    void thread();

    int m_pipe[2];
};

TCPPublisher::TCPPublisher(int sock) : m_accept_sock(sock)
{
    if(pipe(m_pipe) < 0) {
        fprintf(stderr, "pipe() failed -- %s\n", strerror(errno));
        abort();
    }

    m_thread = std::thread([&] { thread(); });
}

TCPPublisher::~TCPPublisher()
{
    close(m_pipe[1]);
    m_thread.join();
}

void
TCPPublisher::thread()
{
    std::vector<struct pollfd> fds;

    while(1) {
        fds.resize(m_subscribers.size() + 2);
        fds[0].fd = m_accept_sock;
        fds[0].events = POLLIN;

        fds[1].fd = m_pipe[0];
        fds[1].events = POLLIN | POLLERR | POLLHUP;

        m_mutex.lock();
        int x = 0;
        for(auto fd : m_subscribers) {
            fds[2 + x].fd = fd;
            fds[2 + x].revents = POLLERR | POLLHUP;
            x++;
        }
        m_mutex.unlock();

        int r = poll(fds.data(), fds.size(), -1);
        if(r < 0) {
            fprintf(stderr, "poll() failed -- %s\n", strerror(errno));
            abort();
        }

        if(fds[1].revents) {
            // Destroy class
            break;
        }

        m_mutex.lock();
        if(fds[0].revents & POLLIN) {
            // New connection

            struct sockaddr_in sin;
            socklen_t slen = sizeof(sin);
            int nfd = accept(m_accept_sock, (struct sockaddr *)&sin, &slen);
            if(nfd < 0) {
                fprintf(stderr, "accept() failed -- %s\n", strerror(errno));
                abort();
            }
            m_subscribers.insert(nfd);
        }

        for(size_t i = 2; i < fds.size(); i++) {
            if(fds[i].revents) {
                close(fds[i].fd);
                m_subscribers.erase(fds[i].fd);
            }
        }
        m_mutex.unlock();
    }

    m_mutex.lock();
    close(m_pipe[0]);
    for(auto fd : m_subscribers) {
        close(fd);
    }
    m_subscribers.clear();
    m_mutex.unlock();
}

void
TCPPublisher::publish(const char *id, Tensor &t, const Dims &offset,
                      TensorAccess *ta)
{
    std::unique_ptr<TensorAccess> TA;

    std::unique_lock lock{m_mutex};

    if(m_subscribers.empty())
        return;

    if(ta == NULL) {
        TA = t.access();
        ta = TA.get();
    }

    size_t outsize = Tensor::DataTypeSize(t.data_type_);

    Header h = {};
    snprintf(h.id, sizeof(h.id), "%s", id);

    switch(t.data_type_) {
    case Tensor::DataType::U8:
        h.datatype = '8';
        break;
    case Tensor::DataType::HALF:
        h.datatype = 'h';
        break;
    case Tensor::DataType::FLOAT:
        h.datatype = 'f';
        break;
    case Tensor::DataType::INT64:
        h.datatype = 'l';
        break;
    case Tensor::DataType::I32:
        h.datatype = 'i';
        break;
    case Tensor::DataType::I16:
        h.datatype = 's';
        break;
    default:
        abort();
    }

    assert(offset.size() < t.dims_.size());

    const size_t output_rank = t.dims_.size() - offset.size();

    for(size_t i = 0; i < output_rank; i++) {
        outsize *= t.dims_[i + offset.size()];
        h.sizes[i] = t.dims_[i + offset.size()];
    }

    h.type = HEADER_TYPE_TENSOR;
    h.rank = output_rank;
    h.payload_length = outsize;
    struct iovec iov[2] = {
        {.iov_base = (void *)&h, .iov_len = sizeof(h)},
        {.iov_base = ta->getAddr({offset}), .iov_len = outsize}};

    struct msghdr msg = {
        .msg_iov = iov,
        .msg_iovlen = 2,
    };

    for(auto fd : m_subscribers) {
        sendmsg(fd, &msg, MSG_NOSIGNAL);
    }
}

void
TCPPublisher::sync()
{
    Header h = {};
    h.type = HEADER_TYPE_SYNC;
    std::unique_lock lock{m_mutex};

    for(auto fd : m_subscribers) {
        send(fd, &h, sizeof(h), MSG_NOSIGNAL);
    }
}

std::shared_ptr<Publisher>
make_tcp_publisher(int bind_port)
{
    int s = socket(AF_INET, SOCK_STREAM, 0);
    if(s < 0) {
        fprintf(stderr, "Unable to create socket -- %s\n", strerror(errno));
        return nullptr;
    }

    const int one = 1;
    setsockopt(s, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(int));

    struct sockaddr_in sin = {AF_INET};
    sin.sin_port = ntohs(bind_port);

    if(bind(s, (struct sockaddr *)&sin, sizeof(sin)) < 0) {
        fprintf(stderr, "Unable to bind socket to port %d -- %s\n", bind_port,
                strerror(errno));
        close(s);
        return nullptr;
    }

    listen(s, 10);

    return std::make_shared<TCPPublisher>(s);
}
}  // namespace saga
