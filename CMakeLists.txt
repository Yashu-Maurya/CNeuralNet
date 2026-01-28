cmake_minimum_required(VERSION 3.10)

project(CNeuralNet)

include_directories(include)

add_library(c_neural_net_lib     
    src/matrix.c
    src/math_functions.c
    src/layer.c
)


set(
    SOURCES 
    examples/simple_net.c
)

add_executable(simple_net_example ${SOURCES})
add_executable(layers_example examples/layers_example.c)

target_link_libraries(simple_net_example c_neural_net_lib)
target_link_libraries(layers_example c_neural_net_lib)
