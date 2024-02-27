cmake_minimum_required(VERSION 3.10)

# Set the project name and version
project(MyProject VERSION 1.0)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 17)

# Find required packages
find_package(OpenCV REQUIRED)
find_package(Torch REQUIRED)

include(FetchContent)
FetchContent_Declare(dlib
    GIT_REPOSITORY https://github.com/davisking/dlib.git
    GIT_TAG        v19.18
)
FetchContent_MakeAvailable(dlib)

# Add the executable
add_executable(dlib_ex dlib_test.cc)
add_executable(opencv_ex opencv_test.cc)
add_executable(pytorch_ex pytorch_test.cc)

# Link the libraries to the executable
# target_link_libraries(MyProject ${OpenCV_LIBS} ${TORCH_LIBRARIES} dlib::dlib)
# set_property(TARGET MyProject PROPERTY CXX_STANDARD 17)

target_link_libraries(dlib_ex ${OpenCV_LIBS} ${TORCH_LIBRARIES} dlib::dlib)
target_link_libraries(opencv_ex ${OpenCV_LIBS} ${TORCH_LIBRARIES} dlib::dlib)
target_link_libraries(pytorch_ex ${OpenCV_LIBS} ${TORCH_LIBRARIES} dlib::dlib)


# The following code block is suggested to be used on Windows.
# According to https://github.com/pytorch/pytorch/issues/25457,
# the DLLs need to be copied to avoid memory errors.
if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET pytorch_ex
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:pytorch_ex>)
endif (MSVC)

