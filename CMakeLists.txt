cmake_minimum_required(VERSION 3.10)

set(PROJECT_REVISION 1 CACHE INTERNAL "Project revision (CI)")
project(tensorflow_task
    VERSION 0.0.${PROJECT_REVISION}
    DESCRIPTION "General CMake template for C++ projects by N.Mikhailovskiy"
    HOMEPAGE_URL "https://github.com/CityAplons/cmake-multiplatform-template"
    LANGUAGES CXX
)

set_property(GLOBAL PROPERTY G_CXX_STANDARD 20)
set_property(GLOBAL PROPERTY G_CXX_STANDARD_REQUIRED ON)

# Include utility functions
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(GNUInstallDirs)
include(compiler)
include(test)

include(FindTensorFlow)
include(FindProtobuf)

find_package(TensorFlow REQUIRED)
find_package (Eigen3 3.3 REQUIRED NO_MODULE)
find_package(Protobuf REQUIRED)

# Boost
set(Boost_USE_STATIC_LIBS OFF) 
set(Boost_USE_RELEASE_LIBS ON)
set(Boost_USE_MULTITHREADED ON)  
set(Boost_USE_STATIC_RUNTIME OFF) 
find_package(Boost COMPONENTS filesystem program_options thread system) 

# Transparent versioning
include_directories(cmake/version)
configure_file(cmake/version/version.h.in version.h)
configure_file(cmake/version/VERSION.in VERSION)
add_library(version cmake/version/project.cpp)
target_include_directories(version
    PRIVATE "${CMAKE_BINARY_DIR}"
)

# Describe build targets
add_executable(fashion_mnist src/hello.cpp)
target_link_libraries(fashion_mnist PRIVATE
    version
    ${TensorFlow_LIBRARIES}
)

set_compile_options(fashion_mnist)
set_compile_options(version)

install(TARGETS fashion_mnist
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

# Describe unit tests
enable_testing()
add_unit_test(test/test_version.cpp version)

# Packaging
option(MAKE_PACKAGE "Add 'package' and 'package_source' build targets for packages" ON)
if (MAKE_PACKAGE)
	add_subdirectory(cmake/packaging)
endif()