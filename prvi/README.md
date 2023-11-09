# C++ setup

1. [CMake download](https://cmake.org/download/)
   [YT help](https://www.youtube.com/watch?v=A735Y4kMIPM)
2. [OpenCV download](https://opencv.org/releases/)
   - Add OpenCV to PATH as:
     set OPENCV_DIR=\path-to\opencv\build
3. [Build Tensorflow](https://www.tensorflow.org/lite/guide/build_cmake)
4. [PyTorch setup](https://pytorch.org/tutorials/advanced/cpp_export.html)

## Pokretanje koda

```bash
cmake -B /tmp/build
cmake --build /tmp/build
cd /tmp/build/Debug
./executable_name
```
