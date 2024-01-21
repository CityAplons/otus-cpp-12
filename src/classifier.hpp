#pragma once

#include <vector>
#include <cstdint>

namespace otus::fmnist
{
    using features_t = std::vector<uint8_t>;
    using probes_t = std::vector<float>;

    class Classifier
    {
    public:
        virtual ~Classifier() {}

        virtual size_t num_classes() const = 0;

        virtual size_t predict(const features_t &) const = 0;

        virtual probes_t predict_vector(const features_t &) const = 0;
    };

} // namespace otus::fmnist
