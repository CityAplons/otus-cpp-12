#pragma once

#include <algorithm>
#include <cstdint>
#include <format>
#include <fstream>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace otus::fmnist {
class CSVReader {
  private:
    std::ifstream csv_;
    size_t lines_;
    size_t width_, height_;

  public:
    explicit CSVReader(const std::string &path, size_t stride)
        : width_(stride) {
        csv_ = std::ifstream(path);
        if (!csv_.is_open()) {
            throw std::runtime_error{std::format("Failed to open {}", path)};
        }

        lines_ = std::count(std::istreambuf_iterator<char>(csv_),
                            std::istreambuf_iterator<char>(), '\n');
        lines_ -= 1;

        csv_.clear();
        csv_.seekg(0);
        std::string row, nop;
        std::getline(csv_, row);

        size_t col_count = 0;
        std::stringstream row_stream(row);
        while (std::getline(row_stream, nop, ',')) {
            ++col_count;
        }

        if (col_count < width_) {
            throw std::runtime_error{"Wrong CSV format"};
        }

        col_count -= 1;   // Subtract item id
        height_ = col_count / width_;

        std::cout << std::format(
                         "Opened {}:\n\tlines: {}\n\twidth: {}\n\theight: {}",
                         path, lines_, width_, height_)
                  << std::endl;
    }
    ~CSVReader() { csv_.close(); };

    void rewind() {
        csv_.clear();
        csv_.seekg(0);
    }

    size_t get_width() { return width_; }
    size_t get_height() { return height_; }
    size_t get_entries() { return lines_; }
    std::optional<std::pair<size_t, std::vector<uint8_t>>> get_data() {
        std::vector<uint8_t> image;
        size_t id;

        std::string line;
        if (!std::getline(csv_, line)) {
            return std::nullopt;
        }

        std::string token;
        std::istringstream line_stream{line};
        std::getline(line_stream, token, ',');
        id = std::atoi(token.c_str());
        while (std::getline(line_stream, token, ',')) {
            image.push_back(static_cast<uint8_t>(std::atoi(token.c_str())));
        }

        return {{id, image}};
    };
};
}   // namespace otus::fmnist
