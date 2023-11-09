cmake_minimum_required(VERSION 3.10)

# Set the project name and version
project(MyProject VERSION 1.0)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 17)

# Add the executable
# add_executable(dlib_ex dlib_test.cc)
# add_executable(opencv_ex opencv_test.cc)
# add_executable(pytorch_ex pytorch_test.cc)

add_executable(docs_opencv docs_opencv.cc)

include(FetchContent)
FetchContent_Declare(dlib
    GIT_REPOSITORY https://github.com/davisking/dlib.git
    GIT_TAG        v19.24
)
FetchContent_MakeAvailable(dlib)

# Link the libraries to the executable
# target_link_libraries(dlib_ex ${OpenCV_LIBS} ${TORCH_LIBRARIES} dlib::dlib)
# target_link_libraries(opencv_ex ${OpenCV_LIBS} ${TORCH_LIBRARIES} dlib::dlib)
# target_link_libraries(pytorch_ex ${OpenCV_LIBS} ${TORCH_LIBRARIES} dlib::dlib)

# Find Package - just for OpenCV
set( OpenCV_DIR "C:/Users/anja.kovacevic/opencv/build" )
option( OpenCV_STATIC OFF )
find_package( OpenCV REQUIRED )

find_package(Torch REQUIRED)

# Set Static Link Runtime Library
if( OpenCV_STATIC )
  foreach( flag_var
           CMAKE_C_FLAGS CMAKE_C_FLAGS_DEBUG CMAKE_C_FLAGS_RELEASE
           CMAKE_C_FLAGS_MINSIZEREL CMAKE_C_FLAGS_RELWITHDEBINFO
           CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE
           CMAKE_CXX_FLAGS_MINSIZEREL CMAKE_CXX_FLAGS_RELWITHDEBINFO )
    if( ${flag_var} MATCHES "/MD" )
      string( REGEX REPLACE "/MD" "/MT" ${flag_var} "${${flag_var}}" )
    endif()
  endforeach()
endif()

if( OpenCV_FOUND )
  # Additional Include Directories
  include_directories( ${OpenCV_INCLUDE_DIRS} )

  # Additional Library Directories
  link_directories( ${OpenCV_LIB_DIR} )

  # Additional Dependencies
  target_link_libraries( docs_opencv ${OpenCV_LIBS} dlib::dlib)
endif()

# The following code block is suggested to be used on Windows.
# According to https://github.com/pytorch/pytorch/issues/25457,
# the DLLs need to be copied to avoid memory errors.
# if (MSVC)
#   file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
#   add_custom_command(TARGET pytorch_ex
#                      POST_BUILD
#                      COMMAND ${CMAKE_COMMAND} -E copy_if_different
#                      ${TORCH_DLLS}
#                      $<TARGET_FILE_DIR:pytorch_ex>)
# endif (MSVC)

