#pragma once
#include "Types.hpp"
#include <random>

class PRNG {
public:
    static PRNG& instance();
    u32 rand();
    f32 randf();

private:
    PRNG();
    std::mt19937 gen_;
    std::uniform_int_distribution<u32> dist_;
};

inline u32 prng_rand() { return PRNG::instance().rand(); }
inline f32 prng_randf() { return PRNG::instance().randf(); }