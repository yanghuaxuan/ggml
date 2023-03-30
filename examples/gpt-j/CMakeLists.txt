#
# gpt-j

add_library(libgptj SHARED
    libgptj.cpp
)
set_property(TARGET libgptj PROPERTY POSITION_INDEPENDENT_CODE ON)
target_link_libraries(libgptj PUBLIC ggml ggml_utils)

set(TEST_TARGET gpt-j)
add_executable(${TEST_TARGET} main.cpp)
target_link_libraries(${TEST_TARGET} PRIVATE libgptj)

