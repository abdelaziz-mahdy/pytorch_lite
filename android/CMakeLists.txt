# cmake_minimum_required(VERSION 3.4.1)
# project(pytorch_lite)

# set(CMAKE_CXX_STANDARD 14)

# add_library(pytorch_lite
#             SHARED
#             ../ios/Classes/pytorch_ffi.cpp)



#   set(build_DIR ${CMAKE_SOURCE_DIR}/build)

  
#   message(FATAL_ERROR "CMAKE_BINARY_DIR: ${build_DIR}")

  
#   file(GLOB PYTORCH_INCLUDE_DIRS "${build_DIR}/pytorch_android*.aar/headers")
#   file(GLOB PYTORCH_LINK_DIRS "${build_DIR}/pytorch_android*.aar/jni/${ANDROID_ABI}")
  

  
#   set(BUILD_SUBDIR ${ANDROID_ABI})
  
#   find_library(PYTORCH_LIBRARY pytorch_jni
#     PATHS ${PYTORCH_LINK_DIRS}
#     NO_CMAKE_FIND_ROOT_PATH)
#   find_library(FBJNI_LIBRARY fbjni
#     PATHS ${PYTORCH_LINK_DIRS}
#     NO_CMAKE_FIND_ROOT_PATH)


#     target_link_libraries(pytorch_lite
#   ${PYTORCH_LIBRARY}
#   ${FBJNI_LIBRARY}
#   log)


  cmake_minimum_required(VERSION 3.4.1)  # for example
  project(pytorch_lite)
  
  set(CMAKE_CXX_STANDARD 14)  # needed for libtorch
  
  add_library( pytorch_lite
  
               # Sets the library as a shared library.
               SHARED
  
               # Provides a relative path to your source file(s).
               ../ios/Classes/pytorch_ffi.cpp )
  
  # libtorch-android archive containing the pre-built shared libraries & headers.
set(LIBTORCH "libtorch-android.zip")

# Download URL of the libtorch-android archive.
set(LIBTORCH_URL "https://github.com/zezo357/pytorch_lite/releases/download/libs/${LIBTORCH}")

# Download location of the libtorch-android archive.
set(LIBTORCH_ARCHIVE "${CMAKE_BINARY_DIR}/${LIBTORCH}")
set(LIBTORCH_SRC "${CMAKE_BINARY_DIR}/libtorch-android")

# Download the libtorch-android archive if it doesn't exist.
if(NOT EXISTS "${LIBTORCH_ARCHIVE}")
  message(STATUS "Downloading ${LIBTORCH}...")
  execute_process(
    COMMAND wget -O "${LIBTORCH_ARCHIVE}" "${LIBTORCH_URL}"
    RESULT_VARIABLE result
  )
  if(result)
    message(FATAL_ERROR "Download of ${LIBTORCH} failed.")
  endif()
  message(STATUS "Downloaded ${LIBTORCH}...")
endif()

# Extract the libtorch-android archive.
if(NOT EXISTS "${LIBTORCH_SRC}")
  message(STATUS "Extracting ${LIBTORCH}...")
  file(MAKE_DIRECTORY "${LIBTORCH_SRC}")
  execute_process(
    COMMAND tar -xvf "${LIBTORCH_ARCHIVE}" -C "${LIBTORCH_SRC}"
    RESULT_VARIABLE result
  )
  if(result)
    message(FATAL_ERROR "Extraction of ${LIBTORCH} failed.")
  endif()
  message(STATUS "Extracted ${LIBTORCH}...")
endif()

  set(LIBTORCH_BASE_DIR "${LIBTORCH_SRC}/libtorch-android")
  file(GLOB PYTORCH_INCLUDE_DIRS "${LIBTORCH_BASE_DIR}/include")
  set(LIBTORCH_LIB_DIR "${LIBTORCH_BASE_DIR}/lib/${ANDROID_ABI}")
  
  add_library(libcplusplus SHARED IMPORTED)
  set_target_properties(libcplusplus PROPERTIES IMPORTED_LOCATION ${LIBTORCH_LIB_DIR}/libc++_shared.so)
  
  add_library(libfbjni SHARED IMPORTED)
  set_target_properties(libfbjni PROPERTIES IMPORTED_LOCATION ${LIBTORCH_LIB_DIR}/libfbjni.so)
  
  add_library(libpytorch_jni SHARED IMPORTED)
  set_target_properties(libpytorch_jni PROPERTIES IMPORTED_LOCATION ${LIBTORCH_LIB_DIR}/libpytorch_jni.so)
  
  
  target_include_directories(pytorch_lite PRIVATE
          ${PYTORCH_INCLUDE_DIRS}
    )
  
  
  target_link_libraries(pytorch_lite
          libcplusplus
          libfbjni
          libpytorch_jni
    )
  