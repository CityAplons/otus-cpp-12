# Tensorflow CPP inference for fMNIST

Last task, 12th. Just loads the pre-trained fashion MNIST model and run test.csv dataset against it to validate accuracy.

### Building

```bash
mkdir -p build && cd build/
cmake -DCMAKE_BUILD_TYPE=Release ../
cmake --build .
```

#### Requirements

* libtensorflow - C API for TensorFlow
* protobuf - dependency for TensorFlow
* Eigen3 - dependency for TensorFlow
* boost - just for argument parsing

Probably, it will not be enough. You may use Dockerfile in this route to build a suitable environment.

#### Building with Docker

In the root of a project, you might run the following lines:

```bash
# build an image with the tag name: fashion_mnist
docker build -t fashion_mnist .

# run new container with a built image in interactive mode
docker run -v $(pwd):/src -w /src -u $(id -u):$(id -g) -it --rm fashion_mnist:latest
```

The written `entrypoint.sh` script will be executed and pass your shell to a container with workdir in the "volumed" project root.

Now you may test the build and verify execution. 

P.S.: in order to add an additional volume with datasets and a model, you may add just another volume like `-v ~/Downloads/test_set:/test`. The files in the host's `~/Downlads/test_set` will be available through container's directory `/test`.

### Execution example

You may download the test dataset and a model from my [GDrive](https://drive.google.com/drive/folders/1KwogRpnsFqmKxtyztQ0-KM-sIJeBw1ww?usp=sharing).

```bash
./build/fashion_mnist -m ~/Downloads/12_homework-33395-e3367f/12_CV/saved_model -t ~/Downloads/archive/fashion-mnist_test.csv
2024-01-21 19:14:39.966715: I external/local_tsl/tsl/cuda/cudart_stub.cc:31] Could not find cuda drivers on your machine, GPU will not be used.
tensorflow_task ver. 0.0.1
Welcome from TensorFlow inference!
TenosrFlow v.2.15.0
Opened /home/nmikhailovskiy/Downloads/archive/fashion-mnist_test.csv:
        lines: 10000
        width: 28
        height: 28
2024-01-21 19:14:40.871378: I tensorflow/cc/saved_model/reader.cc:83] Reading SavedModel from: /home/nmikhailovskiy/Downloads/12_homework-33395-e3367f/12_CV/saved_model
2024-01-21 19:14:40.873572: I tensorflow/cc/saved_model/reader.cc:51] Reading meta graph with tags { serve }
2024-01-21 19:14:40.873628: I tensorflow/cc/saved_model/reader.cc:146] Reading SavedModel debug info (if present) from: /home/nmikhailovskiy/Downloads/12_homework-33395-e3367f/12_CV/saved_model
2024-01-21 19:14:40.873695: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
2024-01-21 19:14:40.895812: I external/local_xla/xla/stream_executor/cuda/cuda_executor.cc:901] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero. See more at https://github.com/torvalds/linux/blob/v6.0/Documentation/ABI/testing/sysfs-bus-pci#L344-L355
2024-01-21 19:14:40.896342: W tensorflow/core/common_runtime/gpu/gpu_device.cc:2256] Cannot dlopen some GPU libraries. Please make sure the missing libraries mentioned above are installed properly if you would like to use GPU. Follow the guide at https://www.tensorflow.org/install/gpu for how to download and setup the required libraries for your platform.
Skipping registering GPU devices...
2024-01-21 19:14:40.986897: I tensorflow/compiler/mlir/mlir_graph_optimization_pass.cc:388] MLIR V1 optimization pass is not enabled
2024-01-21 19:14:40.988033: I tensorflow/cc/saved_model/loader.cc:233] Restoring SavedModel bundle.
2024-01-21 19:14:41.042986: I tensorflow/cc/saved_model/loader.cc:217] Running initialization op on SavedModel bundle at path: /home/nmikhailovskiy/Downloads/12_homework-33395-e3367f/12_CV/saved_model
2024-01-21 19:14:41.056411: I tensorflow/cc/saved_model/loader.cc:316] SavedModel load for tags { serve }; Status: success: OK. Took 185042 microseconds.
Progress: 10000/10000
Result: 9142 out of 10000
        or: 0.9142
```
