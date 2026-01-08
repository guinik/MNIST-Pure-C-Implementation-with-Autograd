#include "PRNG.hpp"

// singleton instance
PRNG& PRNG::instance() {
    static PRNG prng;
    return prng;
}

// constructor
PRNG::PRNG() : gen_(std::random_device{}()), dist_(0, std::numeric_limits<u32>::max()) {}

// member functions
u32 PRNG::rand() {
    return dist_(gen_);
}

f32 PRNG::randf() {
    return static_cast<f32>(rand()) / static_cast<f32>(std::numeric_limits<u32>::max());
}
