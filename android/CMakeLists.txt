
# Set minimum required version of cmake, project name and C++ standard
cmake_minimum_required(VERSION 3.4.1)
project(pytorch_lite)
set(CMAKE_CXX_STANDARD 14)  # needed for libtorch
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp -static-openmp")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -static-openmp")
# set(BUILD_WITH_HALF_PRECISION ON)
# Add library source file(s)
add_library(pytorch_lite SHARED ../ios/Classes/pytorch_ffi.cpp)

# Define function to download and extract a file

function(download_and_extract URL EXPECTED_MD5 ARCHIVE_LOCATION EXTRACT_TO)
  # Determine the archive file path
  set(ARCHIVE "${ARCHIVE_LOCATION}")

  # Check if the file exists
  if(EXISTS "${ARCHIVE}")
    # Calculate the MD5 hash
    file(MD5 "${ARCHIVE}" md5_hash)

    # Check if the hash matches
    if(md5_hash STREQUAL EXPECTED_MD5)
      message(STATUS "Hash matches - no need to download.")
    else()
      message(STATUS "Hash does not match - re-downloading file.")
      file(REMOVE "${ARCHIVE}")
    endif()
  endif()

  # If the file does not exist or the hash doesn't match, download the file
  if(NOT EXISTS "${ARCHIVE}")
    message(STATUS "Downloading from URL ${URL}...")
    file(DOWNLOAD "${URL}" "${ARCHIVE}"
         STATUS status
         SHOW_PROGRESS)

    list(GET status 0 status_code)
    list(GET status 1 status_string)

    if(status_code)
      message(FATAL_ERROR "Download failed with the following error: ${status_string}")
    endif()
  endif()

  file(MD5 "${ARCHIVE}" md5_hash)

  # Compare calculated MD5 hash with the expected MD5 hash
  if(md5_hash STREQUAL EXPECTED_MD5)
    message(STATUS "MD5 hash verification successful. Calculated MD5: ${md5_hash}, Expected MD5: ${EXPECTED_MD5}")
  else()
    message(FATAL_ERROR "MD5 hash verification failed. Calculated MD5: ${md5_hash}, Expected MD5: ${EXPECTED_MD5}. Please run flutter clean and try again.")
  endif()

  message(STATUS "Downloaded archive...")

  # If the extraction directory does not exist, create it and extract the archive
  if(NOT EXISTS "${EXTRACT_TO}")
    message(STATUS "Extracting archive...")

    if(${CMAKE_HOST_SYSTEM_NAME} STREQUAL "Windows")
      message(STATUS "Detected Windows OS, using tar for extraction...")

      file(MAKE_DIRECTORY "${EXTRACT_TO}")
      message(STATUS "Path to extraction directory: ${EXTRACT_TO}")

      execute_process(
        COMMAND tar -xvf "${ARCHIVE}" -C "${EXTRACT_TO}"
        RESULT_VARIABLE result
        OUTPUT_VARIABLE output
        ERROR_VARIABLE error_output
      )
    else()
      message(STATUS "Detected non-Windows OS, using unzip for extraction...")
      
      file(MAKE_DIRECTORY "${EXTRACT_TO}")
      message(STATUS "Path to extraction directory: ${EXTRACT_TO}")

      execute_process(
        COMMAND unzip -o "${ARCHIVE}" -d "${EXTRACT_TO}"
        RESULT_VARIABLE result
        OUTPUT_VARIABLE output
        ERROR_VARIABLE error_output
      )
    endif()

    if(result)
      message(FATAL_ERROR "Extraction of the archive failed with error: ${error_output}\nOutput: ${output}")
    endif()

  endif()

  message(STATUS "Extracted archive...")
endfunction()

# Define base directory
set(BASE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

# Define the files to be downloaded and their corresponding URLs
set(LIBTORCH "libtorch-android-13.1.zip")
set(LIBTORCH_URL "https://github.com/zezo357/pytorch_lite/releases/download/libs/${LIBTORCH}")
set(LIBTORCH_Folder_name "libtorch-android-13.1")


set(opencv_archive "opencv-mobile-4.6.0-android.zip")
set(opencv_archive_URL "https://github.com/nihui/opencv-mobile/releases/download/v16/${opencv_archive}")

# Download and extract the files
download_and_extract("${LIBTORCH_URL}" "05491f47535dc6623c823bfcd0e89c75" "${BASE_DIR}/${LIBTORCH}" "${BASE_DIR}/${LIBTORCH_Folder_name}")
download_and_extract("${opencv_archive_URL}" "a9b147c0a5cfdaf1673a55fc4d9a82e9" "${BASE_DIR}/${opencv_archive}" "${BASE_DIR}/opencv")

# Define directories
set(OpenCV_DIR ${BASE_DIR}/opencv/opencv-mobile-4.6.0-android/sdk/native/jni)
# file(GLOB OpenCV_INCLUDE_DIRS "${OpenCV_DIR}/include/opencv2")
find_package(OpenCV REQUIRED)

set(LIBTORCH_SRC "${BASE_DIR}/${LIBTORCH_Folder_name}")
set(LIBTORCH_BASE_DIR "${LIBTORCH_SRC}/${LIBTORCH_Folder_name}")
file(GLOB PYTORCH_INCLUDE_DIRS "${LIBTORCH_BASE_DIR}/headers")

set(LIBTORCH_LIB_DIR "${LIBTORCH_BASE_DIR}/jni/${ANDROID_ABI}")
# set(OPENCV_LIB_DIR "${OpenCV_DIR}/libs/${ANDROID_ABI}")

# Function to check if a file exists
function(check_file_exists FILE)
    if(NOT EXISTS "${FILE}")
        message(FATAL_ERROR "File ${FILE} does not exist.")
    endif()
endfunction()

# Import libraries
add_library(libcplusplus SHARED IMPORTED)
set_target_properties(libcplusplus PROPERTIES IMPORTED_LOCATION ${LIBTORCH_LIB_DIR}/libc++_shared.so)
check_file_exists(${LIBTORCH_LIB_DIR}/libc++_shared.so)

add_library(libfbjni SHARED IMPORTED)
set_target_properties(libfbjni PROPERTIES IMPORTED_LOCATION ${LIBTORCH_LIB_DIR}/libfbjni.so)
check_file_exists(${LIBTORCH_LIB_DIR}/libfbjni.so)

add_library(libpytorch_jni SHARED IMPORTED)
set_target_properties(libpytorch_jni PROPERTIES IMPORTED_LOCATION ${LIBTORCH_LIB_DIR}/libpytorch_jni.so)
check_file_exists(${LIBTORCH_LIB_DIR}/libpytorch_jni.so)

# add_library(opencv_jni SHARED IMPORTED)
# set_target_properties(opencv_jni PROPERTIES IMPORTED_LOCATION ${OPENCV_LIB_DIR}/libopencv_java4.so)
# check_file_exists(${OPENCV_LIB_DIR}/libopencv_java4.so)

# Include directories
target_include_directories(pytorch_lite PRIVATE
  ${OpenCV_INCLUDE_DIRS}
  ${PYTORCH_INCLUDE_DIRS}
)
# message(FATAL_ERROR ${OpenCV_INCLUDE_DIRS})

# message(FATAL_ERROR ${OpenCV_LIBS})
# Link libraries
target_link_libraries(pytorch_lite
  libcplusplus
  libfbjni
  libpytorch_jni
  ${OpenCV_LIBS}
)
# target_link_libraries(pytorch_lite -fopenmp)
# target_link_libraries(pytorch_lite dl)