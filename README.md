# Tensorflow CPP inference for fMNIST

Last task, 12th.

### Building

```bash
mkdir -p build && cd build/
cmake -DCMAKE_BUILD_TYPE=Release ../
cmake --build .
```

You may add a `-DMAKE_PACKAGE=<OFF|ON>` to disable or enable packages of a project (it's enabled by default).
