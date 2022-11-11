# Copyright (c) 2020, ILUVATAR. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met: *
# Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer. * Redistributions in binary
# form must reproduce the above copyright notice, this list of conditions and
# the following disclaimer in the documentation and/or other materials provided
# with the distribution. * Neither the name of the NVIDIA CORPORATION nor the
# names of its contributors may be used to endorse or promote products derived
# from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY DIRECT,
# INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TOR (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
# EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# Copyright (c) 2017-2020, NVIDIA CORPORATION.  All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met: *
# Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer. * Redistributions in binary
# form must reproduce the above copyright notice, this list of conditions and
# the following disclaimer in the documentation and/or other materials provided
# with the distribution. * Neither the name of the NVIDIA CORPORATION nor the
# names of its contributors may be used to endorse or promote products derived
# from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY DIRECT,
# INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TOR (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
# EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

set(CUDA_ARCH
    ivcore11
    CACHE STRING "The SM architectures to build code for.")

list(APPEND CUDA_CUDA_CLANG_FLAGS -Wno-unused-command-line-argument)
list(APPEND CUDA_CUDA_CLANG_FLAGS -Wl,--disable-new-dtags)

# find cudart

find_library(
  CUDART_LIBRARY cudart
  PATHS ${CUDA_TOOLKIT_ROOT_DIR}
  PATH_SUFFIXES lib/x64 lib64 lib
  NO_DEFAULT_PATH
  # We aren't going to search any system paths. We want to find the runtime in
  # the CUDA toolkit we're building against.
)

if(NOT TARGET cudart AND CUDART_LIBRARY)

  message(STATUS "CUDART: ${CUDART_LIBRARY}")

  if(WIN32)
    add_library(cudart STATIC IMPORTED GLOBAL)
    # Even though we're linking against a .dll, in Windows you statically link
    # against the .lib file found under lib/x64. The .dll will be loaded at
    # runtime automatically from the PATH search.
  else()
    add_library(cudart SHARED IMPORTED GLOBAL)
  endif()

  add_library(nvidia::cudart ALIAS cudart)

  set_property(TARGET cudart PROPERTY IMPORTED_LOCATION ${CUDART_LIBRARY})

elseif(TARGET cudart)

  message(STATUS "CUDART: Already Found")

else()

  message(STATUS "CUDART: Not Found")

endif()

# end cudart

find_library(
  CUDA_DRIVER_LIBRARY cuda
  PATHS ${CUDA_TOOLKIT_ROOT_DIR}
  PATH_SUFFIXES lib/x64 lib64 lib lib64/stubs lib/stubs
  NO_DEFAULT_PATH
  # We aren't going to search any system paths. We want to find the runtime in
  # the CUDA toolkit we're building against.
)

if(NOT TARGET cuda_driver AND CUDA_DRIVER_LIBRARY)

  message(STATUS "CUDA Driver: ${CUDA_DRIVER_LIBRARY}")

  if(WIN32)
    add_library(cuda_driver STATIC IMPORTED GLOBAL)
    # Even though we're linking against a .dll, in Windows you statically link
    # against the .lib file found under lib/x64. The .dll will be loaded at
    # runtime automatically from the PATH search.
  else()
    add_library(cuda_driver SHARED IMPORTED GLOBAL)
  endif()

  add_library(nvidia::cuda_driver ALIAS cuda_driver)

  set_property(TARGET cuda_driver PROPERTY IMPORTED_LOCATION
                                           ${CUDA_DRIVER_LIBRARY})

elseif(TARGET cuda_driver)

  message(STATUS "CUDA Driver: Already Found")

else()

  message(STATUS "CUDA Driver: Not Found")

endif()

find_library(
  NVRTC_LIBRARY nvrtc
  PATHS ${CUDA_TOOLKIT_ROOT_DIR}
  PATH_SUFFIXES lib/x64 lib64 lib
  NO_DEFAULT_PATH
  # We aren't going to search any system paths. We want to find the runtime in
  # the CUDA toolkit we're building against.
)

if(NOT TARGET nvrtc AND NVRTC_LIBRARY)

  message(STATUS "NVRTC: ${NVRTC_LIBRARY}")

  if(WIN32)
    add_library(nvrtc STATIC IMPORTED GLOBAL)
    # Even though we're linking against a .dll, in Windows you statically link
    # against the .lib file found under lib/x64. The .dll will be loaded at
    # runtime automatically from the PATH search.
  else()
    add_library(nvrtc SHARED IMPORTED GLOBAL)
  endif()

  add_library(nvidia::nvrtc ALIAS nvrtc)

  set_property(TARGET nvrtc PROPERTY IMPORTED_LOCATION ${NVRTC_LIBRARY})

elseif(TARGET nvrtc)

  message(STATUS "NVRTC: Already Found")

else()

  message(STATUS "NVRTC: Not Found")

endif()

find_library(
  CURAND_LIBRARY curand
  PATHS ${CUDA_TOOLKIT_ROOT_DIR}
  PATH_SUFFIXES lib/x64 lib64 lib
  NO_DEFAULT_PATH
  # We aren't going to search any system paths. We want to find the runtime in
  # the CUDA toolkit we're building against.
)

if(NOT TARGET curand AND CURAND_LIBRARY)

  message(STATUS "CURAND: ${CURAND_LIBRARY}")

  if(WIN32)
    add_library(curand STATIC IMPORTED GLOBAL)
    # Even though we're linking against a .dll, in Windows you statically link
    # against the .lib file found under lib/x64. The .dll will be loaded at
    # runtime automatically from the PATH search.
  else()
    add_library(curand SHARED IMPORTED GLOBAL)
  endif()

  add_library(nvidia::curand ALIAS curand)

  set_property(TARGET curand PROPERTY IMPORTED_LOCATION ${CURAND_LIBRARY})

elseif(TARGET curand)

  message(STATUS "CURAND: Already Found")

else()

  message(STATUS "CURAND: Not Found")

endif()

find_library(
  IXTHUNK_LIBRARY ixthunk
  PATHS ${CUDA_TOOLKIT_ROOT_DIR}
  PATH_SUFFIXES lib/x64 lib64 lib
  NO_DEFAULT_PATH
  # We aren't going to search any system paths. We want to find the runtime in
  # the CUDA toolkit we're building against.
)

if(NOT TARGET ixthunk AND CURAND_LIBRARY)

  message(STATUS "IXTHUNK: ${IXTHUNK_LIBRARY}")

  if(WIN32)
    add_library(ixthunk STATIC IMPORTED GLOBAL)
    # Even though we're linking against a .dll, in Windows you statically link
    # against the .lib file found under lib/x64. The .dll will be loaded at
    # runtime automatically from the PATH search.
  else()
    add_library(ixthunk SHARED IMPORTED GLOBAL)
  endif()

  add_library(nvidia::ixthunk ALIAS ixthunk)

  set_property(TARGET ixthunk PROPERTY IMPORTED_LOCATION ${IXTHUNK_LIBRARY})

elseif(TARGET ixthunk)

  message(STATUS "IXTHUNK: Already Found")

else()

  message(STATUS "IXTHUNK: Not Found")

endif()

find_library(
  IXLOGGER_LIBRARY ixlogger
  PATHS ${CUDA_TOOLKIT_ROOT_DIR}
  PATH_SUFFIXES lib/x64 lib64 lib
  NO_DEFAULT_PATH
  # We aren't going to search any system paths. We want to find the runtime in
  # the CUDA toolkit we're building against.
)

if(NOT TARGET ixlogger AND IXLOGGER_LIBRARY)

  message(STATUS "IXLOGGER: ${IXLOGGER_LIBRARY}")

  if(WIN32)
    add_library(ixlogger STATIC IMPORTED GLOBAL)
    # Even though we're linking against a .dll, in Windows you statically link
    # against the .lib file found under lib/x64. The .dll will be loaded at
    # runtime automatically from the PATH search.
  else()
    add_library(ixlogger SHARED IMPORTED GLOBAL)
  endif()

  add_library(nvidia::ixlogger ALIAS ixlogger)

  set_property(TARGET ixlogger PROPERTY IMPORTED_LOCATION ${IXLOGGER_LIBRARY})

elseif(TARGET ixlogger)

  message(STATUS "IXLOGGER: Already Found")

else()

  message(STATUS "IXLOGGER: Not Found")

endif()

function(cuda_apply_standard_compile_options TARGET)

  set(CUDA_COMPILE_LANGUAGE CXX)
  set(_FLAGS ${CUDA_CUDA_FLAGS} ${CUDA_CUDA_CLANG_FLAGS})
  set(_FLAGS_RELEASE ${CUDA_CUDA_FLAGS_RELEASE}
                     ${CUDA_CUDA_CLANG_FLAGS_RELEASE})
  set(_FLAGS_RELWITHDEBINFO ${CUDA_CUDA_FLAGS_RELWITHDEBINFO}
                            ${CUDA_CUDA_CLANG_FLAGS_RELWITHDEBINFO})
  set(_FLAGS_DEBUG ${CUDA_CUDA_FLAGS_DEBUG} ${CUDA_CUDA_CLANG_FLAGS_DEBUG})

  target_compile_options(
    ${TARGET}
    PRIVATE
      $<$<COMPILE_LANGUAGE:${CUDA_COMPILE_LANGUAGE}>:${_FLAGS}>
      $<$<COMPILE_LANGUAGE:${CUDA_COMPILE_LANGUAGE}>:$<$<CONFIG:RELEASE>:${_FLAGS_RELEASE}>>
      $<$<COMPILE_LANGUAGE:${CUDA_COMPILE_LANGUAGE}>:$<$<CONFIG:RELWITHDEBINFO>:${_FLAGS_RELWITHDEBINFO}>>
      $<$<COMPILE_LANGUAGE:${CUDA_COMPILE_LANGUAGE}>:$<$<CONFIG:DEBUG>:${_FLAGS_DEBUG}>>
  )

  # target_include_directories( ${TARGET} PRIVATE
  # ${CUDA_TOOLKIT_ROOT_DIR}/include )

endfunction()

function(cuda_apply_cuda_gencode_flags TARGET)
  set(NVCC_FLAGS)
  set(CLANG_FLAGS)
  foreach(ARCH ${CUDA_ARCH})
    list(APPEND CLANG_FLAGS --cuda-gpu-arch=${ARCH})
  endforeach()

  target_compile_options(${TARGET}
                         PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${CLANG_FLAGS}>)
endfunction()

function(cuda_correct_source_file_language_property)
  foreach(File ${ARGN})
    if(File MATCHES ".*\.cu$")
      set_source_files_properties(${File} PROPERTIES LANGUAGE CXX)
    endif()
  endforeach()
endfunction()

function(cuda_unify_source_files TARGET_ARGS_VAR)

  set(options)
  set(oneValueArgs BATCH_SOURCES BATCH_SIZE)
  set(multiValueArgs)
  cmake_parse_arguments(_ "${options}" "${oneValueArgs}" "${multiValueArgs}"
                        ${ARGN})

  if(NOT DEFINED TARGET_ARGS_VAR)
    message(FATAL_ERROR "TARGET_ARGS_VAR parameter is required")
  endif()

  set(TARGET_SOURCE_ARGS ${__UNPARSED_ARGUMENTS})

  set(${TARGET_ARGS_VAR}
      ${TARGET_SOURCE_ARGS}
      PARENT_SCOPE)

endfunction()

function(cuda_add_library NAME)

  set(options)
  set(oneValueArgs EXPORT_NAME)
  set(multiValueArgs)
  cmake_parse_arguments(_ "${options}" "${oneValueArgs}" "${multiValueArgs}"
                        ${ARGN})

  cuda_unify_source_files(TARGET_SOURCE_ARGS ${__UNPARSED_ARGUMENTS})

  cuda_correct_source_file_language_property(${TARGET_SOURCE_ARGS})
  add_library(${NAME} ${TARGET_SOURCE_ARGS})

  cuda_apply_standard_compile_options(${NAME})
  cuda_apply_cuda_gencode_flags(${NAME})

  target_compile_features(${NAME} INTERFACE cxx_std_11)

endfunction()

function(cuda_add_executable NAME)

  set(options)
  set(oneValueArgs)
  set(multiValueArgs)
  cmake_parse_arguments(_ "${options}" "${oneValueArgs}" "${multiValueArgs}"
                        ${ARGN})

  cuda_unify_source_files(TARGET_SOURCE_ARGS ${__UNPARSED_ARGUMENTS})

  cuda_correct_source_file_language_property(${TARGET_SOURCE_ARGS})
  add_executable(${NAME} ${TARGET_SOURCE_ARGS})

  cuda_apply_standard_compile_options(${NAME})
  cuda_apply_cuda_gencode_flags(${NAME})

  target_compile_features(${NAME} INTERFACE cxx_std_11)

endfunction()
