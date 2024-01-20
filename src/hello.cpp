#include <iostream>
#include <tensorflow/c/c_api.h>

#include "project.h"

int main(int argc, char const *argv[]) {
  struct ProjectInfo info = {};

  std::cout << info.nameString << "\t" << info.versionString << '\n';
  std::cout << "Welcome from TensorFlow inference!\n";
  std::cout << "TenosrFlow v." << TF_Version() << std::endl;

  (void)argc;
  (void)argv;
  return 0;
}
