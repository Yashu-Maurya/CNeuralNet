cmake_minimum_required(VERSION 3.10)

project(CNeuralNet)

include_directories(include)

set(
    SOURCES 
    src/main.c 
    src/matrix.c
)

add_executable(c_neural_net ${SOURCES})