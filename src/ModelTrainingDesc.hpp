#include "Matrix.hpp"
struct ModelTrainingDesc {
    Matrix* train_images = nullptr;
    Matrix* train_labels = nullptr;
    Matrix* test_images = nullptr;
    Matrix* test_labels = nullptr;

    u32 epochs = 10;
    u32 batch_size = 50;
    f32 learning_rate = 0.01f;
};
