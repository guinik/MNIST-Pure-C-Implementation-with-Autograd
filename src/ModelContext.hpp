#pragma once
#include <cstdint>
#include <vector>
#include <memory>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <random>

#include "Types.hpp"
#include "ModelVariables.hpp"
class ModelContext {
public:
    std::vector<std::unique_ptr<ModelVar>> all_vars;

    ModelVar* input = nullptr;
    ModelVar* output = nullptr;
    ModelVar* desired_output = nullptr;
    ModelVar* cost = nullptr;

    ModelProgram forward_prog;
    ModelProgram cost_prog;

    u32 num_vars() const { return static_cast<u32>(all_vars.size()); }

    ModelVar* create_var(u32 rows, u32 cols, u32 flags);
    ModelVar* relu(ModelVar* input, u32 flags);
    ModelVar* softmax(ModelVar* input, u32 flags);
    ModelVar* add(ModelVar* a, ModelVar* b, u32 flags);
    ModelVar* sub(ModelVar* a, ModelVar* b, u32 flags);
    ModelVar* matmul(ModelVar* a, ModelVar* b, u32 flags);
    ModelVar* cross_entropy(ModelVar* p, ModelVar* q, u32 flags);

    void compile();
    void feedforward();
    void train(const struct ModelTrainingDesc& desc);

private:
    ModelVar* unary_impl(ModelVar* input, u32 rows, u32 cols, u32 flags, ModelVarOp op);
    ModelVar* binary_impl(ModelVar* a, ModelVar* b, u32 rows, u32 cols, u32 flags, ModelVarOp op);
    ModelProgram create_program(ModelVar* out_var);
};