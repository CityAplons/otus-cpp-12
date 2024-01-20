#!/bin/sh -l
set -e

cmake -B `pwd` -DPROJECT_REVISION=$INPUT_REVISION
cmake --build `pwd`
cmake --build `pwd` --target test
cmake --build `pwd` --target package
