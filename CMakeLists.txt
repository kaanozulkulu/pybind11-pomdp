# set (CMAKE_CXX_STANDARD 14)
# add_subdirectory(extern/pybind11)
# pybind11_add_module(example SHARED example.cpp basics.cpp pouct.cpp)


# use: cmake -DPYBIND11_PYTHON_VERSION=3.9.7 . if python version used on machine is 3.9.7

#Options for cmake:
# For OMP long: cmake -DPYBIND11_PYTHON_VERSION=3.9.7 -DOpenMP_C_FLAGS=-fopenmp=lomp -DOpenMP_CXX_FLAGS=-fopenmp=lomp -DOpenMP_C_LIB_NAMES="libomp" -DOpenMP_CXX_LIB_NAMES="libomp" -DOpenMP_libomp_LIBRARY="/usr/local/opt/libomp/lib/libomp.dylib" -DOpenMP_CXX_FLAGS="-Xpreprocessor -fopenmp /usr/local/opt/libomp/lib/libomp.dylib -I/usr/local/opt/libomp/include" -DOpenMP_CXX_LIB_NAMES="libomp" -DOpenMP_C_FLAGS="-Xpreprocessor -fopenmp /usr/local/opt/libomp/lib/libomp.dylib -I/usr/local/opt/libomp/include" .
#For OMP Short: cmake -DPYBIND11_PYTHON_VERSION=3.9.7  -DOpenMP_CXX_FLAGS=-fopenmp=lomp -DOpenMP_CXX_LIB_NAMES="libomp" -DOpenMP_libomp_LIBRARY="/usr/local/opt/libomp/lib/libomp.dylib" -DOpenMP_CXX_FLAGS="-Xpreprocessor -fopenmp /usr/local/opt/libomp/lib/libomp.dylib -I/usr/local/opt/libomp/include" -DOpenMP_CXX_LIB_NAMES="libomp"
#For OMP Shortest: cmake -DPYBIND11_PYTHON_VERSION=3.9.7  -DOpenMP_CXX_FLAGS="-Xpreprocessor -fopenmp /usr/local/opt/libomp/lib/libomp.dylib -I/usr/local/opt/libomp/include" -DOpenMP_CXX_LIB_NAMES="libomp" -DOpenMP_libomp_LIBRARY="/usr/local/opt/libomp/lib/libomp.dylib" 

cmake_minimum_required(VERSION 3.18.2)
set (CMAKE_CXX_STANDARD 17)
project(example)

# set path to libomp
# set(DOpenMP_CXX_FLAGS -Xpreprocessor -fopenmp -I/usr/local/opt/libomp/include)
# set(DOpenMP_CXX_LIB_NAMES omp) 
# set(DOpenMP_omp_LIBRARY /usr/local/opt/libomp/lib/libomp.dylib)

add_subdirectory(extern/pybind11)
pybind11_add_module(example SHARED example.cpp basics.cpp pouct.cpp)

find_package(OpenMP REQUIRED COMPONENTS CXX)

target_link_libraries(example PRIVATE OpenMP::OpenMP_CXX)





