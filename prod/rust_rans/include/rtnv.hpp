#pragma once

#include "rtnv.h"
#include <stdexcept>
#include <string>
#include <vector>

namespace rtnv {

class Engine {
public:
  Engine(const std::string &weight_path, const float *global_freq, int vocab = 1024) {
    impl_ = rtnv_engine_load(weight_path.c_str(), global_freq, vocab);
    if (!impl_) {
      throw std::runtime_error("rtnv_engine_load failed");
    }
  }

  Engine(const Engine &) = delete;
  Engine &operator=(const Engine &) = delete;

  ~Engine() { rtnv_engine_free(impl_); }

  std::vector<uint8_t> encode(const int32_t *tokens, int n_frames, int s = 128, bool adaptive = true) {
    std::vector<uint8_t> out(n_frames * s * 8 + 4096);
    size_t n = 0;
    int rc = rtnv_encode(impl_, tokens, n_frames, s, out.data(), out.size(), &n, adaptive ? 1 : 0);
    if (rc != 0) {
      throw std::runtime_error("rtnv_encode failed");
    }
    out.resize(n);
    return out;
  }

  std::vector<int32_t> decode(const uint8_t *bits, size_t bits_len, int n_frames_hint = 0) {
    int nf = n_frames_hint > 0 ? n_frames_hint : 1200;
    std::vector<int32_t> tok(static_cast<size_t>(nf) * 128);
    int got = rtnv_decode(impl_, bits, bits_len, n_frames_hint, tok.data());
    if (got < 0) {
      throw std::runtime_error("rtnv_decode failed");
    }
    tok.resize(static_cast<size_t>(got) * 128);
    return tok;
  }

private:
  RtnvEngine *impl_;
};

} // namespace rtnv
