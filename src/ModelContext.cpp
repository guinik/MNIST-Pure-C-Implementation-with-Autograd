#include "ModelContext.hpp"
#include "ModelTrainingDesc.hpp"
#include "PRNG.hpp"

ModelVar* ModelContext::create_var(u32 rows, u32 cols, u32 flags) {
    auto var = std::make_unique<ModelVar>();

    var->index = num_vars();
    var->flags = flags;
    var->op = ModelVarOp::Create;
    var->val = Matrix::create(rows, cols);

    if (flags & MV_FLAG_REQUIRES_GRAD) {
        var->grad = Matrix::create(rows, cols);
    }

    ModelVar* ptr = var.get();
    all_vars.push_back(std::move(var));

    if (flags & MV_FLAG_INPUT)          input = ptr;
    if (flags & MV_FLAG_OUTPUT)         output = ptr;
    if (flags & MV_FLAG_DESIRED_OUTPUT) desired_output = ptr;
    if (flags & MV_FLAG_COST)           cost = ptr;

    return ptr;
}

ModelVar* ModelContext::unary_impl(ModelVar* input_var, u32 rows, u32 cols, u32 flags, ModelVarOp op) {
    if (input_var->flags & MV_FLAG_REQUIRES_GRAD) {
        flags |= MV_FLAG_REQUIRES_GRAD;
    }

    ModelVar* out = create_var(rows, cols, flags);
    out->op = op;
    out->inputs[0] = input_var;

    return out;
}

ModelVar* ModelContext::binary_impl(ModelVar* a, ModelVar* b, u32 rows, u32 cols, u32 flags, ModelVarOp op) {
    if ((a->flags & MV_FLAG_REQUIRES_GRAD) || (b->flags & MV_FLAG_REQUIRES_GRAD)) {
        flags |= MV_FLAG_REQUIRES_GRAD;
    }

    ModelVar* out = create_var(rows, cols, flags);
    out->op = op;
    out->inputs[0] = a;
    out->inputs[1] = b;

    return out;
}

ModelVar* ModelContext::relu(ModelVar* input_var, u32 flags) {
    return unary_impl(input_var, input_var->val->rows, input_var->val->cols, flags, ModelVarOp::Relu);
}

ModelVar* ModelContext::softmax(ModelVar* input_var, u32 flags) {
    return unary_impl(input_var, input_var->val->rows, input_var->val->cols, flags, ModelVarOp::Softmax);
}

ModelVar* ModelContext::add(ModelVar* a, ModelVar* b, u32 flags) {
    if (a->val->rows != b->val->rows || a->val->cols != b->val->cols) {
        return nullptr;
    }
    return binary_impl(a, b, a->val->rows, a->val->cols, flags, ModelVarOp::Add);
}

ModelVar* ModelContext::sub(ModelVar* a, ModelVar* b, u32 flags) {
    if (a->val->rows != b->val->rows || a->val->cols != b->val->cols) {
        return nullptr;
    }
    return binary_impl(a, b, a->val->rows, a->val->cols, flags, ModelVarOp::Sub);
}

ModelVar* ModelContext::matmul(ModelVar* a, ModelVar* b, u32 flags) {
    if (a->val->cols != b->val->rows) {
        return nullptr;
    }
    return binary_impl(a, b, a->val->rows, b->val->cols, flags, ModelVarOp::Matmul);
}

ModelVar* ModelContext::cross_entropy(ModelVar* p, ModelVar* q, u32 flags) {
    if (p->val->rows != q->val->rows || p->val->cols != q->val->cols) {
        return nullptr;
    }
    return binary_impl(p, q, p->val->rows, p->val->cols, flags, ModelVarOp::CrossEntropy);
}

ModelProgram ModelContext::create_program(ModelVar* out_var) {
    // This is the autograd approach!
    // You order in topological order to get autograd running
    // essentially follow depth first of variable inputs 
    // if you see a variable twice, it means all its inputs have been processed
    // add it to the end order
    std::vector<bool> visited(num_vars(), false);
    std::vector<ModelVar*> stack;
    std::vector<ModelVar*> out;

    stack.push_back(out_var);

    while (!stack.empty()) {
        ModelVar* cur = stack.back();
        stack.pop_back();

        if (cur->index >= num_vars()) continue;

        if (visited[cur->index]) {
            out.push_back(cur);
            continue;
        }

        visited[cur->index] = true;
        stack.push_back(cur);

        u32 num_inputs = mv_num_inputs(cur->op);
        for (u32 i = 0; i < num_inputs; i++) {
            ModelVar* inp = cur->inputs[i];

            if (inp->index >= num_vars() || visited[inp->index]) {
                continue;
            }

            // Remove from stack if already present
            auto it = std::find(stack.begin(), stack.end(), inp);
            if (it != stack.end()) {
                stack.erase(it);
            }

            stack.push_back(inp);
        }
    }

    ModelProgram prog;
    prog.vars = std::move(out);
    return prog;
}

void ModelContext::compile() {
    if (output != nullptr) {
        forward_prog = create_program(output);
    }
    if (cost != nullptr) {
        cost_prog = create_program(cost);
    }
}

void ModelContext::feedforward() {
    for (u32 i = 0; i < forward_prog.size(); i++) {
        ModelVar* cur = forward_prog.vars[i];
        ModelVar* a = cur->inputs[0];
        ModelVar* b = cur->inputs[1];

        switch (cur->op) {
        case ModelVarOp::Null:
        case ModelVarOp::Create:
        case ModelVarOp::UnaryStart:
        case ModelVarOp::BinaryStart:
            break;

        case ModelVarOp::Relu:
            MatOps::relu(*cur->val, *a->val);
            break;
        case ModelVarOp::Softmax:
            MatOps::softmax(*cur->val, *a->val);
            break;
        case ModelVarOp::Add:
            MatOps::add(*cur->val, *a->val, *b->val);
            break;
        case ModelVarOp::Sub:
            MatOps::sub(*cur->val, *a->val, *b->val);
            break;
        case ModelVarOp::Matmul:
            MatOps::mul(*cur->val, *a->val, *b->val, true, false, false);
            break;
        case ModelVarOp::CrossEntropy:
            MatOps::cross_entropy(*cur->val, *a->val, *b->val);
            break;
        }
    }
}




namespace {

    void compute_program(ModelProgram& prog) {
        for (u32 i = 0; i < prog.size(); i++) {
            ModelVar* cur = prog.vars[i];
            ModelVar* a = cur->inputs[0];
            ModelVar* b = cur->inputs[1];

            switch (cur->op) {
            case ModelVarOp::Null:
            case ModelVarOp::Create:
            case ModelVarOp::UnaryStart:
            case ModelVarOp::BinaryStart:
                break;

            case ModelVarOp::Relu:
                MatOps::relu(*cur->val, *a->val);
                break;
            case ModelVarOp::Softmax:
                MatOps::softmax(*cur->val, *a->val);
                break;
            case ModelVarOp::Add:
                MatOps::add(*cur->val, *a->val, *b->val);
                break;
            case ModelVarOp::Sub:
                MatOps::sub(*cur->val, *a->val, *b->val);
                break;
            case ModelVarOp::Matmul:
                MatOps::mul(*cur->val, *a->val, *b->val, true, false, false);
                break;
            case ModelVarOp::CrossEntropy:
                MatOps::cross_entropy(*cur->val, *a->val, *b->val);
                break;
            }
        }
    }

    void compute_grads(ModelProgram& prog) {
        // Clear non-parameter gradients
        for (u32 i = 0; i < prog.size(); i++) {
            ModelVar* cur = prog.vars[i];
            if (!(cur->flags & MV_FLAG_REQUIRES_GRAD)) continue;
            if (cur->flags & MV_FLAG_PARAMETER) continue;
            cur->grad->clear();
        }

        // Initialize output gradient
        prog.vars[prog.size() - 1]->grad->fill(1.0f);

        // Backprop
        for (i64 i = static_cast<i64>(prog.size()) - 1; i >= 0; i--) {
            ModelVar* cur = prog.vars[i];

            if ((cur->flags & MV_FLAG_REQUIRES_GRAD) == 0) continue;

            ModelVar* a = cur->inputs[0];
            ModelVar* b = cur->inputs[1];

            u32 num_inputs = mv_num_inputs(cur->op);

            if (num_inputs == 1 && !(a->flags & MV_FLAG_REQUIRES_GRAD)) continue;
            if (num_inputs == 2 && !(a->flags & MV_FLAG_REQUIRES_GRAD) && !(b->flags & MV_FLAG_REQUIRES_GRAD)) continue;

            switch (cur->op) {
            case ModelVarOp::Null:
            case ModelVarOp::Create:
            case ModelVarOp::UnaryStart:
            case ModelVarOp::BinaryStart:
                break;

            case ModelVarOp::Relu:
                MatOps::relu_add_grad(*a->grad, *a->val, *cur->grad);
                break;

            case ModelVarOp::Softmax:
                MatOps::softmax_add_grad(*a->grad, *cur->val, *cur->grad);
                break;

            case ModelVarOp::Add:
                if (a->flags & MV_FLAG_REQUIRES_GRAD)
                    MatOps::add(*a->grad, *a->grad, *cur->grad);
                if (b->flags & MV_FLAG_REQUIRES_GRAD)
                    MatOps::add(*b->grad, *b->grad, *cur->grad);
                break;

            case ModelVarOp::Sub:
                if (a->flags & MV_FLAG_REQUIRES_GRAD)
                    MatOps::add(*a->grad, *a->grad, *cur->grad);
                if (b->flags & MV_FLAG_REQUIRES_GRAD)
                    MatOps::sub(*b->grad, *b->grad, *cur->grad);
                break;

            case ModelVarOp::Matmul:
                if (a->flags & MV_FLAG_REQUIRES_GRAD)
                    MatOps::mul(*a->grad, *cur->grad, *b->val, false, false, true);
                if (b->flags & MV_FLAG_REQUIRES_GRAD)
                    MatOps::mul(*b->grad, *a->val, *cur->grad, false, true, false);
                break;

            case ModelVarOp::CrossEntropy:
                MatOps::cross_entropy_add_grad(
                    (a->flags & MV_FLAG_REQUIRES_GRAD) ? a->grad.get() : nullptr,
                    (b->flags & MV_FLAG_REQUIRES_GRAD) ? b->grad.get() : nullptr,
                    *a->val, *b->val, *cur->grad
                );
                break;
            }
        }
    }

}


void ModelContext::train(const ModelTrainingDesc& desc) {
    Matrix* train_images = desc.train_images;
    Matrix* train_labels = desc.train_labels;
    Matrix* test_images = desc.test_images;
    Matrix* test_labels = desc.test_labels;

    u32 num_examples = train_images->rows;
    u32 input_size = train_images->cols;
    u32 output_size = train_labels->cols;
    u32 num_tests = test_images->rows;

    u32 num_batches = num_examples / desc.batch_size;

    std::vector<u32> training_order(num_examples);
    for (u32 i = 0; i < num_examples; i++) {
        training_order[i] = i;
    }

    for (u32 epoch = 0; epoch < desc.epochs; epoch++) {
        // Shuffle training order
        for (u32 i = 0; i < num_examples; i++) {
            u32 a = prng_rand() % num_examples;
            u32 b = prng_rand() % num_examples;
            std::swap(training_order[a], training_order[b]);
        }

        for (u32 batch = 0; batch < num_batches; batch++) {
            // Clear parameter gradients
            for (auto& var : all_vars) {
                if (var->flags & MV_FLAG_PARAMETER) {
                    var->grad->clear();
                }
            }

            f32 avg_cost = 0.0f;
            for (u32 i = 0; i < desc.batch_size; i++) {
                u32 order_index = batch * desc.batch_size + i;
                u32 index = training_order[order_index];

                std::memcpy(
                    input->val->data.data(),
                    train_images->data.data() + index * input_size,
                    sizeof(f32) * input_size
                );

                std::memcpy(
                    desired_output->val->data.data(),
                    train_labels->data.data() + index * output_size,
                    sizeof(f32) * output_size
                );

                compute_program(cost_prog);
                compute_grads(cost_prog);

                avg_cost += cost->val->sum();
            }
            avg_cost /= static_cast<f32>(desc.batch_size);

            // Update parameters
            for (auto& var : all_vars) {
                if (!(var->flags & MV_FLAG_PARAMETER)) continue;

                var->grad->scale(desc.learning_rate / desc.batch_size);
                MatOps::sub(*var->val, *var->val, *var->grad);
            }

            std::printf(
                "Epoch %2u / %2u, Batch %4u / %4u, Average Cost: %.4f\r",
                epoch + 1, desc.epochs,
                batch + 1, num_batches, avg_cost
            );
            std::fflush(stdout);
        }
        std::printf("\n");

        // Test accuracy
        u32 num_correct = 0;
        f32 avg_cost = 0.0f;
        for (u32 i = 0; i < num_tests; i++) {
            std::memcpy(
                input->val->data.data(),
                test_images->data.data() + i * input_size,
                sizeof(f32) * input_size
            );

            std::memcpy(
                desired_output->val->data.data(),
                test_labels->data.data() + i * output_size,
                sizeof(f32) * output_size
            );

            compute_program(cost_prog);

            avg_cost += cost->val->sum();
            num_correct += (output->val->argmax() == desired_output->val->argmax()) ? 1 : 0;
        }

        avg_cost /= static_cast<f32>(num_tests);
        std::printf(
            "Test Completed. Accuracy: %5u / %5u (%.1f%%), Average Cost: %.4f\n",
            num_correct, num_tests,
            static_cast<f32>(num_correct) / num_tests * 100.0f,
            avg_cost
        );
    }
}

