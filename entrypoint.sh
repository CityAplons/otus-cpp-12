#!/bin/sh -l
set -e

[ -z "$INPUT_BUILD" ] && echo "Skipping build...\n Debug mode :)" && /bin/bash && exit 0;

cmake -B `pwd`/build -DPROJECT_REVISION=$INPUT_REVISION
cmake --build `pwd`/build
cmake --build `pwd`/build --target test
cmake --build `pwd`/build --target package
