#include "csv_reader.hpp"
#include "inference.hpp"
#include "project.h"

#include <iostream>

#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>
#include <tensorflow/c/c_api.h>

int
main(int argc, char const *argv[]) {
    struct ProjectInfo info = {};

    std::cout << info.nameString << "\t" << info.versionString << '\n';
    std::cout << "Welcome from TensorFlow inference!\n";
    std::cout << "TenosrFlow v." << TF_Version() << std::endl;

    namespace po = boost::program_options;
    namespace fs = boost::filesystem;

    po::options_description desc("Test TensorFlow model");
    desc.add_options()("help,h", "Print this message")(
        "model,m", po::value<fs::path>()->required(),
        "TensorFlow model saved weights path")(
        "test,t", po::value<fs::path>()->required(), "CSV test data path");

    po::variables_map vm;

    try {
        po::store(po::parse_command_line(argc, argv, desc), vm);
        po::notify(vm);
    } catch (const std::exception &e) {
        if (vm.count("help")) {
            std::cout << desc << "\n";
        }
        std::cerr << e.what() << '\n';
        return 2;
    }

    if (vm.count("help")) {
        std::cout << desc << "\n";
        return 1;
    }

    auto csv_file = vm["test"].as<fs::path>();
    auto model_path = vm["model"].as<fs::path>();

    auto csv = otus::fmnist::CSVReader(csv_file.string(), 28);
    auto inference = otus::fmnist::Inferecne(model_path.string(),
                                             csv.get_width(), csv.get_height());

    size_t valid_predictions = 0, it = 0;
    size_t total = csv.get_entries();
    for (; it < total; ++it) {
        std::cout << "\rProgress: " << it + 1 << "/" << total;

        size_t id;
        std::vector<uint8_t> image;
        std::tie(id, image) = *csv.get_data();

        size_t predicted_id = inference.predict(image);
        if (predicted_id == id) {
            ++valid_predictions;
        }
        // else {
        //     std::cout << "\nActual: " << id << "\nPredicted: " <<
        //     predicted_id
        //               << std::endl;
        // }
    }

    std::cout << std::endl;
    std::cout << "Result:\t" << valid_predictions << " out of " << it
              << "\n\tor: " << static_cast<float>(valid_predictions) / total
              << std::endl;

    return 0;
}
