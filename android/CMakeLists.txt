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

# set base directory
set(BASE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

# use BASE_DIR in other variables
set(LIBTORCH_ARCHIVE "${BASE_DIR}/${LIBTORCH}")
set(LIBTORCH_SRC "${BASE_DIR}/libtorch-android")


# Expected MD5 hash
set(EXPECTED_MD5 "61f013bc7371a65b900d14fccb27a013")

# Check if file exists
if(EXISTS "${LIBTORCH_ARCHIVE}")
  # Calculate MD5 hash
  file(MD5 "${LIBTORCH_ARCHIVE}" md5_hash)

  # Check if hash matches
  if(${md5_hash} STREQUAL ${EXPECTED_MD5})
    message(STATUS "Hash matches - no need to download.")
  else()
    message(STATUS "Hash does not match - re-downloading file.")
    file(REMOVE "${LIBTORCH_ARCHIVE}")
    set(FILE_DOES_NOT_EXIST TRUE)
  endif()
endif()

# If the file does not exist or the hash doesn't match, download the file
if(NOT EXISTS "${LIBTORCH_ARCHIVE}" OR FILE_DOES_NOT_EXIST)
  message(STATUS "Downloading ${LIBTORCH}...")
  file(DOWNLOAD "${LIBTORCH_URL}" "${LIBTORCH_ARCHIVE}"
       STATUS status
       SHOW_PROGRESS)

  list(GET status 0 status_code)
  list(GET status 1 status_string)

  if(status_code)
    message(FATAL_ERROR "Download of ${LIBTORCH} failed with the following error: ${status_string}")
  endif()
endif()


# Compare calculated MD5 hash with the expected MD5 hash
if(md5_hash STREQUAL EXPECTED_MD5)
  message(STATUS "MD5 hash verification successful.")
else()
  message(FATAL_ERROR "MD5 hash verification failed. , please run flutter clean and try again")
endif()

message(STATUS "Downloaded ${LIBTORCH}...")

# # Attach MD5 hash of downloaded file (for getting the MD5 hash) 
# message(FATAL_ERROR "MD5 hash of downloaded file: ${md5_hash}")

if(NOT EXISTS "${LIBTORCH_SRC}")
  message(STATUS "Extracting ${LIBTORCH}...")

  if(${CMAKE_HOST_SYSTEM_NAME} STREQUAL "Windows")
    message(STATUS "Detected Windows OS, using tar for extraction...")

    file(MAKE_DIRECTORY "${LIBTORCH_SRC}")
    message(STATUS "Path to LIBTORCH_SRC: ${LIBTORCH_SRC}")

    execute_process(
      COMMAND tar -xvf "${LIBTORCH_ARCHIVE}" -C "${LIBTORCH_SRC}"
      RESULT_VARIABLE result
      OUTPUT_VARIABLE output
      ERROR_VARIABLE error_output
    )
  else()
    message(STATUS "Detected non-Windows OS, using unzip for extraction...")
    
    file(MAKE_DIRECTORY "${LIBTORCH_SRC}")
    message(STATUS "Path to LIBTORCH_SRC: ${LIBTORCH_SRC}")

    execute_process(
      COMMAND unzip -o "${LIBTORCH_ARCHIVE}" -d "${LIBTORCH_SRC}"
      RESULT_VARIABLE result
      OUTPUT_VARIABLE output
      ERROR_VARIABLE error_output
    )
  endif()

  if(result)
    message(FATAL_ERROR "Extraction of ${LIBTORCH} failed with error: ${error_output}\nOutput: ${output}")
  endif()

endif()

message(STATUS "Extracted ${LIBTORCH}...")

# message(FATAL_ERROR "kjasjdlksajd error")


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
  