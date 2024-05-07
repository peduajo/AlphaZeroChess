cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(alphazero)

set(CMAKE_CUDA_COMPILER "/usr/local/cuda-12.1/bin/nvcc")
set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda-12.1")

#set(DEP_DIR "${PROJECT_SOURCE_DIR}/deps") # Ajusta esta ruta según sea necesario

include_directories(
    /usr/local/include/torch_tensorrt
)

find_package(Torch REQUIRED)
find_package(Threads REQUIRED)
set(TensorRT_ROOT "~/Documents/TensorRT-8.6.1.6")
set(torchtrt_DIR "/usr/local/lib/cmake/torchtrt")
find_package(torchtrt REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")

add_executable(alphazero_train alphazero_train.cpp lib/game.cpp lib/params.cpp lib/mcts.cpp lib/alphazero.cpp lib/node.cpp lib/ThreadPool.h)
target_link_libraries(alphazero_train PRIVATE torch "-Wl,--no-as-needed" torchtrt "-Wl,--as-needed")
set_property(TARGET alphazero_train PROPERTY CXX_STANDARD 17)

add_executable(alphazero_test alphazero_test.cpp lib/game.cpp lib/params.cpp lib/mcts.cpp lib/node.cpp lib/ThreadPool.h)
#add_executable(alphazero_test alphazero_test.cpp lib/game.cpp lib/params.cpp lib/ThreadPool.h)
target_link_libraries(alphazero_test PRIVATE torch "-Wl,--no-as-needed" torchtrt "-Wl,--as-needed")
set_property(TARGET alphazero_test PROPERTY CXX_STANDARD 17)

set(CMAKE_CUDA_ARCHITECTURES 52;61;75)
set(CMAKE_BUILD_TYPE Debug)