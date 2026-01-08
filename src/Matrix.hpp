#pragma once

#include <cstdint>
#include <vector>
#include <memory>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <random>

#include "Types.hpp"

class Matrix {
public:
    u32 rows = 0;
    u32 cols = 0;
    std::vector<f32> data;

    Matrix() = default;
    Matrix(u32 r, u32 c);

    static std::unique_ptr<Matrix> create(u32 rows, u32 cols);
    static std::unique_ptr<Matrix> load(u32 rows, u32 cols, const char* filename);

    bool copy_from(const Matrix& src);
    void clear();
    void fill(f32 x);
    void fill_rand(f32 lower, f32 upper);
    void scale(f32 s);
    f32 sum() const;
    u64 argmax() const;
    u64 size() const;

    f32& at(u32 r, u32 c);
    const f32& at(u32 r, u32 c) const;
};


namespace MatOps {

    bool add(Matrix& out, const Matrix& a, const Matrix& b);
    bool sub(Matrix& out, const Matrix& a, const Matrix& b);

    void mul_nn(Matrix& out, const Matrix& a, const Matrix& b);
    void mul_nt(Matrix& out, const Matrix& a, const Matrix& b);
    void mul_tn(Matrix& out, const Matrix& a, const Matrix& b);
    void mul_tt(Matrix& out, const Matrix& a, const Matrix& b);

    bool mul(Matrix& out, const Matrix& a, const Matrix& b,
        bool zero_out = true, bool transpose_a = false, bool transpose_b = false);

    bool relu(Matrix& out, const Matrix& in);
    bool softmax(Matrix& out, const Matrix& in);
    bool cross_entropy(Matrix& out, const Matrix& p, const Matrix& q);

    bool relu_add_grad(Matrix& out, const Matrix& in, const Matrix& grad);
    bool softmax_add_grad(Matrix& out, const Matrix& softmax_out, const Matrix& grad);
    bool cross_entropy_add_grad(Matrix* p_grad, Matrix* q_grad,
        const Matrix& p, const Matrix& q, const Matrix& grad);

} // namespace MatOps

