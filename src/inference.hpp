#pragma once

#include "tf_classifier.hpp"

#include <string>

namespace otus::fmnist {

class Inferecne {
  private:
    tf::TFClassifier classifier_;

  public:
    explicit Inferecne(const std::string &path, size_t width, size_t height)
        : classifier_(path, width, height) {}

    size_t predict(features_t data) { return classifier_.predict(data); }
};
}   // namespace otus::fmnist
