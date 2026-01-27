cmake_minimum_required(VERSION 3.10)

project(CNeuralNet)

include_directories(include)

set(
    SOURCES 
    examples/simple_net.c
    src/matrix.c
    src/math_functions.c
)

add_executable(simple_net_example ${SOURCES})