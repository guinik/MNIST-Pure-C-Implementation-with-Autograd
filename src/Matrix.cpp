#include <cstring>
#include <vector>
#include <memory>
#include <random>
#include <algorithm>
#include <fstream>
#include <iostream>

#include "PRNG.hpp"
#include "Types.hpp"
#include "Matrix.hpp"

Matrix::Matrix(u32 r, u32 c) : rows(r), cols(c), data(static_cast<u64>(r)* c, 0.0f) {}

std::unique_ptr<Matrix> Matrix::create(u32 rows, u32 cols) {
    return std::make_unique<Matrix>(rows, cols);
}

std::unique_ptr<Matrix> Matrix::load(u32 rows, u32 cols, const char* filename) {
    auto mat = create(rows, cols);

    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return mat;
    }

    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    size = std::min(size, static_cast<std::streamsize>(sizeof(f32) * rows * cols));
    file.read(reinterpret_cast<char*>(mat->data.data()), size);

    return mat;
}

bool Matrix::copy_from(const Matrix& src) {
    if (rows != src.rows || cols != src.cols) return false;
    data = src.data;
    return true;
}

void Matrix::clear() {
    std::fill(data.begin(), data.end(), 0.0f);
}

void Matrix::fill(f32 x) {
    std::fill(data.begin(), data.end(), x);
}

void Matrix::fill_rand(f32 lower, f32 upper) {
    for (auto& v : data) {
        v = prng_randf() * (upper - lower) + lower;
    }
}

void Matrix::scale(f32 s) {
    for (auto& v : data) v *= s;
}

f32 Matrix::sum() const {
    f32 s = 0.0f;
    for (auto v : data) s += v;
    return s;
}

u64 Matrix::argmax() const {
    u64 max_i = 0;
    for (u64 i = 1; i < data.size(); i++) {
        if (data[i] > data[max_i]) max_i = i;
    }
    return max_i;
}

u64 Matrix::size() const { return static_cast<u64>(rows) * cols; }

f32& Matrix::at(u32 r, u32 c) { return data[c + r * cols]; }
const f32& Matrix::at(u32 r, u32 c) const { return data[c + r * cols]; }




namespace MatOps {

    bool add(Matrix& out, const Matrix& a, const Matrix& b) {
        if (a.rows != b.rows || a.cols != b.cols) return false;
        if (out.rows != a.rows || out.cols != a.cols) return false;

        for (u64 i = 0; i < out.size(); i++) {
            out.data[i] = a.data[i] + b.data[i];
        }
        return true;
    }

    bool sub(Matrix& out, const Matrix& a, const Matrix& b) {
        if (a.rows != b.rows || a.cols != b.cols) return false;
        if (out.rows != a.rows || out.cols != a.cols) return false;

        for (u64 i = 0; i < out.size(); i++) {
            out.data[i] = a.data[i] - b.data[i];
        }
        return true;
    }

    void mul_nn(Matrix& out, const Matrix& a, const Matrix& b) {
        for (u64 i = 0; i < out.rows; i++) {
            for (u64 k = 0; k < a.cols; k++) {
                for (u64 j = 0; j < out.cols; j++) {
                    out.data[j + i * out.cols] +=
                        a.data[k + i * a.cols] *
                        b.data[j + k * b.cols];
                }
            }
        }
    }

    void mul_nt(Matrix& out, const Matrix& a, const Matrix& b) {
        for (u64 i = 0; i < out.rows; i++) {
            for (u64 j = 0; j < out.cols; j++) {
                for (u64 k = 0; k < a.cols; k++) {
                    out.data[j + i * out.cols] +=
                        a.data[k + i * a.cols] *
                        b.data[k + j * b.cols];
                }
            }
        }
    }

    void mul_tn(Matrix& out, const Matrix& a, const Matrix& b) {
        for (u64 k = 0; k < a.rows; k++) {
            for (u64 i = 0; i < out.rows; i++) {
                for (u64 j = 0; j < out.cols; j++) {
                    out.data[j + i * out.cols] +=
                        a.data[i + k * a.cols] *
                        b.data[j + k * b.cols];
                }
            }
        }
    }

    void mul_tt(Matrix& out, const Matrix& a, const Matrix& b) {
        for (u64 i = 0; i < out.rows; i++) {
            for (u64 j = 0; j < out.cols; j++) {
                for (u64 k = 0; k < a.rows; k++) {
                    out.data[j + i * out.cols] +=
                        a.data[i + k * a.cols] *
                        b.data[k + j * b.cols];
                }
            }
        }
    }

    bool mul(Matrix& out, const Matrix& a, const Matrix& b,
        bool zero_out, bool transpose_a, bool transpose_b) {
        u32 a_rows = transpose_a ? a.cols : a.rows;
        u32 a_cols = transpose_a ? a.rows : a.cols;
        u32 b_rows = transpose_b ? b.cols : b.rows;
        u32 b_cols = transpose_b ? b.rows : b.cols;

        if (a_cols != b_rows) return false;
        if (out.rows != a_rows || out.cols != b_cols) return false;

        if (zero_out) {
            out.clear();
        }

        u32 transpose = (static_cast<u32>(transpose_a) << 1) | static_cast<u32>(transpose_b);
        switch (transpose) {
        case 0b00: mul_nn(out, a, b); break;
        case 0b01: mul_nt(out, a, b); break;
        case 0b10: mul_tn(out, a, b); break;
        case 0b11: mul_tt(out, a, b); break;
        }

        return true;
    }

    bool relu(Matrix& out, const Matrix& in) {
        if (out.rows != in.rows || out.cols != in.cols) return false;

        for (u64 i = 0; i < out.size(); i++) {
            out.data[i] = std::max(0.0f, in.data[i]);
        }
        return true;
    }

    bool softmax(Matrix& out, const Matrix& in) {
        if (out.rows != in.rows || out.cols != in.cols) return false;

        f32 sum = 0.0f;
        for (u64 i = 0; i < out.size(); i++) {
            out.data[i] = std::exp(in.data[i]);
            sum += out.data[i];
        }
        out.scale(1.0f / sum);

        return true;
    }

    bool cross_entropy(Matrix& out, const Matrix& p, const Matrix& q) {
        if (p.rows != q.rows || p.cols != q.cols) return false;
        if (out.rows != p.rows || out.cols != p.cols) return false;

        for (u64 i = 0; i < out.size(); i++) {
            out.data[i] = (p.data[i] == 0.0f) ? 0.0f : p.data[i] * -std::log(q.data[i]);
        }
        return true;
    }

    bool relu_add_grad(Matrix& out, const Matrix& in, const Matrix& grad) {
        if (out.rows != in.rows || out.cols != in.cols) return false;
        if (out.rows != grad.rows || out.cols != grad.cols) return false;

        for (u64 i = 0; i < out.size(); i++) {
            out.data[i] += (in.data[i] > 0.0f) ? grad.data[i] : 0.0f;
        }
        return true;
    }

    bool softmax_add_grad(Matrix& out, const Matrix& softmax_out, const Matrix& grad) {
        if (softmax_out.rows != 1 && softmax_out.cols != 1) return false;

        u32 size = std::max(softmax_out.rows, softmax_out.cols);
        Matrix jacobian(size, size);

        for (u32 i = 0; i < size; i++) {
            for (u32 j = 0; j < size; j++) {
                jacobian.data[j + i * size] =
                    softmax_out.data[i] * ((i == j ? 1.0f : 0.0f) - softmax_out.data[j]);
            }
        }

        mul(out, jacobian, grad, false, false, false);
        return true;
    }

    bool cross_entropy_add_grad(Matrix* p_grad, Matrix* q_grad,
        const Matrix& p, const Matrix& q, const Matrix& grad) {
        if (p.rows != q.rows || p.cols != q.cols) return false;

        u64 size = p.size();

        if (p_grad != nullptr) {
            if (p_grad->rows != p.rows || p_grad->cols != p.cols) return false;
            for (u64 i = 0; i < size; i++) {
                p_grad->data[i] += -std::log(q.data[i]) * grad.data[i];
            }
        }

        if (q_grad != nullptr) {
            if (q_grad->rows != q.rows || q_grad->cols != q.cols) return false;
            for (u64 i = 0; i < size; i++) {
                q_grad->data[i] += -p.data[i] / q.data[i] * grad.data[i];
            }
        }

        return true;
    }

} // namespace MatOps
