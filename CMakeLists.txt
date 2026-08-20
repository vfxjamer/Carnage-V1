cmake_minimum_required(VERSION 3.18)

project("Carnage" CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Python3 COMPONENTS Interpreter REQUIRED)

execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c "import torch; print(torch.utils.cmake_prefix_path)"
    RESULT_VARIABLE TORCH_PATH_RESULT
    OUTPUT_VARIABLE TORCH_INSTALL_PREFIX
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_VARIABLE TORCH_PATH_ERROR
)

if(NOT TORCH_PATH_RESULT EQUAL 0 OR NOT TORCH_INSTALL_PREFIX)
    message(FATAL_ERROR
        "Could not locate PyTorch's CMake package. Make sure Python can import torch.\n"
        "Python: ${Python3_EXECUTABLE}\n"
        "Error: ${TORCH_PATH_ERROR}"
    )
endif()

list(APPEND CMAKE_PREFIX_PATH "${TORCH_INSTALL_PREFIX}")
message(STATUS "PyTorch CMake prefix: ${TORCH_INSTALL_PREFIX}")

# GigaLearnCPP owns the actual Torch linkage. We only discover Torch here so
# its package configuration and CUDA libraries are available to the subtree.
find_package(CUDAToolkit QUIET)
if(CUDAToolkit_FOUND)
    enable_language(CUDA)
    message(STATUS "CUDA toolkit found: ${CUDAToolkit_VERSION}")
    message(STATUS "CUDA compiler: ${CMAKE_CUDA_COMPILER}")
else()
    message(STATUS "CUDA toolkit not found - building CPU-only")
endif()

find_package(Torch REQUIRED)

add_subdirectory(thirdparty/GigaLearnCPP-Leak/GigaLearnCPP)
add_subdirectory(thirdparty/GigaLearnCPP-Leak/RLBotCPP)

file(GLOB CARNAGE_SRC "src/*.cpp" "src/*.h")

add_executable(Carnage ${CARNAGE_SRC})

if(MSVC)
    target_link_options(Carnage PRIVATE /STACK:8388608)
endif()

# Do not link Torch::Torch here: the bundled GigaLearnCPP CMake target already
# links the Torch libraries and the pip wheel may not export Torch::Torch as an
# imported target on all Linux/Colab configurations.
target_link_libraries(Carnage PRIVATE GigaLearnCPP RLBotCPP)
target_include_directories(Carnage PRIVATE "src")

add_custom_command(TARGET Carnage POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "$<TARGET_FILE_DIR:GigaLearnCPP>"
        "$<TARGET_FILE_DIR:${PROJECT_NAME}>"
    COMMENT "Copying GigaLearnCPP libraries to output..."
)

add_custom_command(TARGET Carnage POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${CMAKE_SOURCE_DIR}/collision_meshes"
        "$<TARGET_FILE_DIR:${PROJECT_NAME}>/collision_meshes"
    COMMENT "Copying collision meshes to output..."
)

add_custom_command(TARGET Carnage POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${CMAKE_SOURCE_DIR}/thirdparty/GigaLearnCPP-Leak/GigaLearnCPP/python_scripts"
        "$<TARGET_FILE_DIR:${PROJECT_NAME}>/python_scripts"
    COMMENT "Copying Python scripts to output..."
)