#!/bin/bash
set -eu -o pipefail

make clean

if [[ "$(uname)" == "Darwin" ]]; then
  make -j build ARCH=x86-64-modern
else
  make -j build ARCH=x86-64-bmi2
fi
