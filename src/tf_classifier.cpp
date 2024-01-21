#include "tf_classifier.hpp"

#include <algorithm>
#include <format>
#include <stdexcept>
#include <string>
#include <vector>

using namespace otus::fmnist::tf;

static void
dummy_deleter(void *data, size_t length, void *arg) {
    (void) data;
    (void) length;
    (void) arg;
    return;
}

void
TFClassifier::delete_tf_session(TF_Session *tf_session) {
    tf_status status{TF_NewStatus(), TF_DeleteStatus};
    TF_DeleteSession(tf_session, status.get());
    if (TF_GetCode(status.get()) != TF_OK) {
        throw std::runtime_error{std::format("Unable to delete TF_Session: {}",
                                             TF_Message(status.get()))};
    }
}

TFClassifier::TFClassifier(const std::string &modelpath, const size_t width,
                           const size_t height)
    : width_{width}, height_{height} {

    tf_status status{TF_NewStatus(), TF_DeleteStatus};
    tf_import_graph_def_options opts{TF_NewImportGraphDefOptions(),
                                     TF_DeleteImportGraphDefOptions};

    TF_Buffer *run_opts = NULL;
    const char *tags = "serve";

    session_.reset(TF_LoadSessionFromSavedModel(
        session_opts_.get(), run_opts, modelpath.c_str(), &tags, 1,
        graph_.get(), nullptr, status.get()));
    if (TF_GetCode(status.get()) != TF_OK) {
        throw std::runtime_error{
            std::format("Unable to import graph from {}: {}", modelpath,
                        TF_Message(status.get()))};
    }

    input_op_ = TF_GraphOperationByName(graph_.get(), "serving_default_input");
    if (input_op_ == nullptr) {
        throw std::runtime_error{"Input not found"};
    }

    output_op_ =
        TF_GraphOperationByName(graph_.get(), "StatefulPartitionedCall");
    if (output_op_ == nullptr) {
        throw std::runtime_error{"Output not found"};
    }
}

size_t
TFClassifier::num_classes() const {
    return 10;
}

size_t
TFClassifier::predict(const features_t &input) const {
    auto probe = predict_vector(input);
    auto argmax = std::max_element(probe.begin(), probe.end());
    return std::distance(probe.begin(), argmax);
}

otus::fmnist::probes_t
TFClassifier::predict_vector(const features_t &input) const {
    if (width_ * height_ != input.size()) {
        throw std::runtime_error{
            std::format("Wrong input size provided!\nInput({}), model({})\n",
                        input.size(), width_ * height_)};
    }

    // Preprocess input features
    probes_t preproc_features;
    preproc_features.reserve(input.size());

    // Divide each bytes by 255
    auto transformer = [](uint8_t val) {
        return static_cast<float>(val) / 255.0;
    };
    std::transform(input.begin(), input.end(),
                   std::back_inserter(preproc_features), transformer);

    std::vector<TF_Output> inputs;
    std::vector<TF_Tensor *> input_values;

    TF_Output input_opout = {input_op_, 0};
    inputs.push_back(input_opout);

    const size_t num_bytes_in = width_ * height_ * sizeof(float);
    int64_t in_dims[] = {1, static_cast<int64_t>(width_),
                         static_cast<int64_t>(height_), 1};
    tf_tensor input_tensor{
        TF_NewTensor(TF_FLOAT, in_dims, 4,
                     static_cast<void *>(preproc_features.data()), num_bytes_in,
                     &dummy_deleter, nullptr),
        TF_DeleteTensor};
    input_values.push_back(input_tensor.get());

    std::vector<TF_Output> outputs;
    std::vector<TF_Tensor *> output_values(outputs.size(), nullptr);

    TF_Output output_opout = {output_op_, 0};
    outputs.push_back(output_opout);

    const size_t num_bytes_out = num_classes() * sizeof(float);
    int64_t out_dims[] = {1, static_cast<int64_t>(num_classes())};
    tf_tensor output_value{
        TF_AllocateTensor(TF_FLOAT, out_dims, 2, num_bytes_out),
        TF_DeleteTensor};
    output_values.push_back(output_value.get());

    tf_status status{TF_NewStatus(), TF_DeleteStatus};
    TF_SessionRun(session_.get(), nullptr, &inputs[0], &input_values[0],
                  inputs.size(), &outputs[0], &output_values[0], outputs.size(),
                  nullptr, 0, nullptr, status.get());
    if (TF_GetCode(status.get()) != TF_OK) {
        throw std::runtime_error{std::format(
            "Unable to run session from graph: {}", TF_Message(status.get()))};
    }

    probes_t probes;
    float *out_vals = static_cast<float *>(TF_TensorData(output_values[0]));
    for (size_t i = 0; i < num_classes(); ++i) {
        probes.push_back(*out_vals++);
    }

    return probes;
}
