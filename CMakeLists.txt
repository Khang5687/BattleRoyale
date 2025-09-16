cmake_minimum_required(VERSION 3.20)
project(BattleRoyale5 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(PkgConfig REQUIRED)

# Vulkan loader (MoltenVK via loader on macOS)
find_package(Vulkan REQUIRED)

# GLFW via pkg-config to be robust with Homebrew
pkg_check_modules(GLFW3 REQUIRED IMPORTED_TARGET glfw3)

add_executable(battleroyale5
  src/main.cpp
)

target_link_libraries(battleroyale5 PRIVATE Vulkan::Vulkan PkgConfig::GLFW3)

target_compile_definitions(battleroyale5 PRIVATE GLFW_INCLUDE_VULKAN)

# Compile shaders if a compiler is available
find_program(GLSLC_EXECUTABLE NAMES glslc HINTS ENV VULKAN_SDK PATH_SUFFIXES Bin bin)
find_program(GLSLANG_VALIDATOR_EXECUTABLE NAMES glslangValidator HINTS ENV VULKAN_SDK PATH_SUFFIXES Bin bin)

set(SHADER_SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/shaders/circle.vert
  ${CMAKE_CURRENT_SOURCE_DIR}/shaders/circle.frag
)

set(COMPILED_SHADERS)
foreach(SRC ${SHADER_SOURCES})
  get_filename_component(FNAME ${SRC} NAME)
  set(SPIRV ${CMAKE_CURRENT_BINARY_DIR}/shaders/${FNAME}.spv)
  list(APPEND COMPILED_SHADERS ${SPIRV})
  if (GLSLC_EXECUTABLE)
    add_custom_command(
      OUTPUT ${SPIRV}
      COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/shaders
      COMMAND ${GLSLC_EXECUTABLE} -fshader-stage=$<IF:$<STREQUAL:$<LOWER_CASE:${FNAME}>,circle.vert>,vert,frag> ${SRC} -o ${SPIRV}
      DEPENDS ${SRC}
      VERBATIM
    )
  elseif (GLSLANG_VALIDATOR_EXECUTABLE)
    add_custom_command(
      OUTPUT ${SPIRV}
      COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/shaders
      COMMAND ${GLSLANG_VALIDATOR_EXECUTABLE} -V ${SRC} -o ${SPIRV}
      DEPENDS ${SRC}
      VERBATIM
    )
  endif()
endforeach()

if (COMPILED_SHADERS)
  add_custom_target(battleroyale5_shaders DEPENDS ${COMPILED_SHADERS})
  add_dependencies(battleroyale5 battleroyale5_shaders)
  target_compile_definitions(battleroyale5 PRIVATE BR5_SHADER_DIR="${CMAKE_CURRENT_BINARY_DIR}/shaders")
else()
  message(WARNING "No shader compiler (glslc or glslangValidator) found. Expect runtime to fail if shaders are missing.")
  target_compile_definitions(battleroyale5 PRIVATE BR5_SHADER_DIR="${CMAKE_CURRENT_BINARY_DIR}/shaders")
endif()

if(APPLE)
  set_target_properties(battleroyale5 PROPERTIES
    MACOSX_BUNDLE FALSE
    BUILD_WITH_INSTALL_RPATH TRUE
    BUILD_RPATH "/opt/homebrew/lib"
    INSTALL_RPATH "/opt/homebrew/lib"
  )
endif()

