cmake_minimum_required(VERSION 3.18)
project(P1_Serial)

find_package(Torch REQUIRED)

# OpenCV paths from PyTorch conda env
set(PYTORCH_ROOT "/software/rhel9/manual/install/python/pytorch_251_311_cu124")
set(OPENCV_INCLUDE_DIR "${PYTORCH_ROOT}/include/opencv4")
set(OPENCV_LIB_DIR "${PYTORCH_ROOT}/lib")

# --- cv_wrapper: compiled with ABI=1 to match OpenCV ---
add_library(cv_wrapper STATIC src/cv_wrapper.cpp)
target_include_directories(cv_wrapper PRIVATE ${OPENCV_INCLUDE_DIR})
target_link_libraries(cv_wrapper
    ${OPENCV_LIB_DIR}/libopencv_core.so
    ${OPENCV_LIB_DIR}/libopencv_imgproc.so
    ${OPENCV_LIB_DIR}/libopencv_imgcodecs.so
    ${OPENCV_LIB_DIR}/libopencv_videoio.so
)
# Force ABI=1 for wrapper (matches OpenCV)
target_compile_definitions(cv_wrapper PRIVATE _GLIBCXX_USE_CXX11_ABI=1)
set_property(TARGET cv_wrapper PROPERTY CXX_STANDARD 17)

# --- Main executable: compiled with ABI=0 (forced by LibTorch) ---
add_executable(P1_Serial src/P1_Serial.cpp)
target_include_directories(P1_Serial PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(P1_Serial
    "${TORCH_LIBRARIES}"
    cv_wrapper
    ${OPENCV_LIB_DIR}/libopencv_core.so
    ${OPENCV_LIB_DIR}/libopencv_imgproc.so
    ${OPENCV_LIB_DIR}/libopencv_imgcodecs.so
    ${OPENCV_LIB_DIR}/libopencv_videoio.so
)
set_property(TARGET P1_Serial PROPERTY CXX_STANDARD 17)
