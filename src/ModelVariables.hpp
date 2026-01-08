
#pragma once
#include "Matrix.hpp"
#include <stdio.h>
#include "Types.hpp"

enum ModelVarFlags : u32 {
    MV_FLAG_NONE = 0,
    MV_FLAG_REQUIRES_GRAD = (1 << 0),
    MV_FLAG_PARAMETER = (1 << 1),
    MV_FLAG_INPUT = (1 << 2),
    MV_FLAG_OUTPUT = (1 << 3),
    MV_FLAG_DESIRED_OUTPUT = (1 << 4),
    MV_FLAG_COST = (1 << 5),
};

enum class ModelVarOp : u32 {
    Null = 0,
    Create,

    UnaryStart,
    Relu,
    Softmax,

    BinaryStart,
    Add,
    Sub,
    Matmul,
    CrossEntropy,
};
constexpr u32 MODEL_VAR_MAX_INPUTS = 2;

inline u32 mv_num_inputs(ModelVarOp op) {
    if (op < ModelVarOp::UnaryStart) return 0;
    if (op < ModelVarOp::BinaryStart) return 1;
    return 2;
}

struct ModelVar;

struct ModelVar {
    u32 index = 0;
    u32 flags = 0;

    std::unique_ptr<Matrix> val;
    std::unique_ptr<Matrix> grad;

    ModelVarOp op = ModelVarOp::Null;
    ModelVar* inputs[MODEL_VAR_MAX_INPUTS] = { nullptr, nullptr };
};



struct ModelProgram {
    std::vector<ModelVar*> vars;

    u32 size() const { return static_cast<u32>(vars.size()); }
};


