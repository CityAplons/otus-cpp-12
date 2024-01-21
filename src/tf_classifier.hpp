#pragma once

#include "classifier.hpp"

#include <memory>
#include <tensorflow/c/c_api.h>

namespace otus::fmnist::tf
{
    class TFClassifier : public Classifier
    {
    public:
        TFClassifier(const std::string &modelpath,
                     const size_t width,
                     const size_t height);
        TFClassifier(const TFClassifier &) = delete;
        TFClassifier &operator=(const TFClassifier &) = delete;

        size_t num_classes() const final;
        size_t predict(const features_t &) const final;
        probes_t predict_vector(const features_t &) const final;

    protected:
        static void delete_tf_session(TF_Session *);

        using tf_graph = std::unique_ptr<TF_Graph, decltype(&TF_DeleteGraph)>;
        using tf_buffer = std::unique_ptr<TF_Buffer, decltype(&TF_DeleteBuffer)>;
        using tf_import_graph_def_options = std::unique_ptr<TF_ImportGraphDefOptions, decltype(&TF_DeleteImportGraphDefOptions)>;
        using tf_status = std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)>;
        using tf_session_options = std::unique_ptr<TF_SessionOptions, decltype(&TF_DeleteSessionOptions)>;
        using tf_tensor = std::unique_ptr<TF_Tensor, decltype(&TF_DeleteTensor)>;
        using tf_session = std::unique_ptr<TF_Session, decltype(&delete_tf_session)>;

    private:
        tf_graph graph_{TF_NewGraph(), TF_DeleteGraph};
        tf_session_options session_opts_{TF_NewSessionOptions(), TF_DeleteSessionOptions};
        tf_session session_ = {nullptr, delete_tf_session};
        TF_Operation *input_op_ = nullptr;
        TF_Operation *output_op_ = nullptr;
        size_t width_;
        size_t height_;
    };

} // namespace otus::fmnist::tf
