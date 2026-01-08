#include <cstdint>
#include <cstdio>
#include <cmath>
#include <cstring>
#include <vector>
#include <memory>
#include <random>
#include <algorithm>
#include <fstream>
#include <iostream>


#include "Types.hpp"
#include "Matrix.hpp"
#include "ModelContext.hpp"
#include "ModelVariables.hpp"
#include "PRNG.hpp"
#include "ModelTrainingDesc.hpp"


// ============================================================================
// Utilities
// ============================================================================

void draw_mnist_digit(const f32* data) {
    for (u32 y = 0; y < 28; y++) {
        for (u32 x = 0; x < 28; x++) {
            f32 num = data[x + y * 28];
            u32 col = 232 + static_cast<u32>(num * 23);
            std::printf("\x1b[48;5;%um  ", col);
        }
        std::printf("\n");
    }
    std::printf("\x1b[0m");
}

void create_mnist_model(ModelContext& model) {
    ModelVar* input = model.create_var(784, 1, MV_FLAG_INPUT);

    ModelVar* W0 = model.create_var(16, 784, MV_FLAG_REQUIRES_GRAD | MV_FLAG_PARAMETER);
    ModelVar* W1 = model.create_var(16, 16, MV_FLAG_REQUIRES_GRAD | MV_FLAG_PARAMETER);
    ModelVar* W2 = model.create_var(10, 16, MV_FLAG_REQUIRES_GRAD | MV_FLAG_PARAMETER);

    f32 bound0 = std::sqrt(6.0f / (784 + 16));
    f32 bound1 = std::sqrt(6.0f / (16 + 16));
    f32 bound2 = std::sqrt(6.0f / (16 + 10));
    W0->val->fill_rand(-bound0, bound0);
    W1->val->fill_rand(-bound1, bound1);
    W2->val->fill_rand(-bound2, bound2);

    ModelVar* b0 = model.create_var(16, 1, MV_FLAG_REQUIRES_GRAD | MV_FLAG_PARAMETER);
    ModelVar* b1 = model.create_var(16, 1, MV_FLAG_REQUIRES_GRAD | MV_FLAG_PARAMETER);
    ModelVar* b2 = model.create_var(10, 1, MV_FLAG_REQUIRES_GRAD | MV_FLAG_PARAMETER);

    ModelVar* z0_a = model.matmul(W0, input, 0);
    ModelVar* z0_b = model.add(z0_a, b0, 0);
    ModelVar* a0 = model.relu(z0_b, 0);

    ModelVar* z1_a = model.matmul(W1, a0, 0);
    ModelVar* z1_b = model.add(z1_a, b1, 0);
    ModelVar* z1_c = model.relu(z1_b, 0);
    ModelVar* a1 = model.add(a0, z1_c, 0);

    ModelVar* z2_a = model.matmul(W2, a1, 0);
    ModelVar* z2_b = model.add(z2_a, b2, 0);
    ModelVar* output = model.softmax(z2_b, MV_FLAG_OUTPUT);

    ModelVar* y = model.create_var(10, 1, MV_FLAG_DESIRED_OUTPUT);

    model.cross_entropy(y, output, MV_FLAG_COST);
}

// ============================================================================
// Main
// ============================================================================

int main() {
    auto train_images = Matrix::load(60000, 784, "train_images.mat");
    auto test_images = Matrix::load(10000, 784, "test_images.mat");
    auto train_labels = Matrix::create(60000, 10);
    auto test_labels = Matrix::create(10000, 10);

    {
        auto train_labels_file = Matrix::load(60000, 1, "train_labels.mat");
        auto test_labels_file = Matrix::load(10000, 1, "test_labels.mat");

        for (u32 i = 0; i < 60000; i++) {
            u32 num = static_cast<u32>(train_labels_file->data[i]);
            train_labels->data[i * 10 + num] = 1.0f;
        }

        for (u32 i = 0; i < 10000; i++) {
            u32 num = static_cast<u32>(test_labels_file->data[i]);
            test_labels->data[i * 10 + num] = 1.0f;
        }
    }

    draw_mnist_digit(test_images->data.data());
    for (u32 i = 0; i < 10; i++) {
        std::printf("%.0f ", test_labels->data[i]);
    }
    std::printf("\n\n");

    ModelContext model;
    create_mnist_model(model);
    model.compile();

    std::memcpy(model.input->val->data.data(), test_images->data.data(), sizeof(f32) * 784);
    model.feedforward();

    std::printf("Pre-training output: ");
    for (u32 i = 0; i < 10; i++) {
        std::printf("%.2f ", model.output->val->data[i]);
    }
    std::printf("\n");

    ModelTrainingDesc training_desc;
    training_desc.train_images = train_images.get();
    training_desc.train_labels = train_labels.get();
    training_desc.test_images = test_images.get();
    training_desc.test_labels = test_labels.get();
    training_desc.epochs = 10;
    training_desc.batch_size = 50;
    training_desc.learning_rate = 0.01f;

    model.train(training_desc);


    const u32 num_test = 10;

    for (u32 n = 0; n < num_test; n++) {
        const f32* img_data = test_images->data.data() + n * 784;
        draw_mnist_digit(img_data);

        std::memcpy(model.input->val->data.data(), img_data, sizeof(f32) * 784);
        model.feedforward();

        u64 pred = model.output->val->argmax();

        std::printf("     Test image %u predicted: %llu\n\n", n, pred);
    }
    std::printf("\n\n");

    return 0;
}
