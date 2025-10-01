cmake_minimum_required(VERSION 3.20)
project(BattleRoyale5 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Vulkan loader
find_package(Vulkan REQUIRED)

# GLFW: use vcpkg on Windows, pkg-config on Unix
if(WIN32)
	find_package(glfw3 CONFIG REQUIRED)
	set(GLFW_LIBRARIES glfw)
else()
	find_package(PkgConfig REQUIRED)
	pkg_check_modules(GLFW3 REQUIRED IMPORTED_TARGET glfw3)
	set(GLFW_LIBRARIES PkgConfig::GLFW3)
endif()

# Main simulation executable
add_executable(battleroyale5
  src/main.cpp
  src/damage_curve.cpp
  src/font_loader.cpp
)

target_link_libraries(battleroyale5 PRIVATE Vulkan::Vulkan ${GLFW_LIBRARIES})

# Config tool executable
add_executable(battleroyale5-config
  src/config_main.cpp
  src/damage_curve.cpp
  src/font_loader.cpp
)

target_link_libraries(battleroyale5-config PRIVATE ${GLFW_LIBRARIES})

# OpenGL for config tool
find_package(OpenGL REQUIRED)
target_link_libraries(battleroyale5-config PRIVATE OpenGL::GL)

target_compile_definitions(battleroyale5 PRIVATE GLFW_INCLUDE_VULKAN)

# Compile shaders if a compiler is available
find_program(GLSLC_EXECUTABLE NAMES glslc HINTS ENV VULKAN_SDK PATH_SUFFIXES Bin bin)
find_program(GLSLANG_VALIDATOR_EXECUTABLE NAMES glslangValidator HINTS ENV VULKAN_SDK PATH_SUFFIXES Bin bin)

set(SHADER_SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/shaders/circle.vert
  ${CMAKE_CURRENT_SOURCE_DIR}/shaders/circle.frag
  ${CMAKE_CURRENT_SOURCE_DIR}/shaders/circle_optimized.frag
  ${CMAKE_CURRENT_SOURCE_DIR}/shaders/health_bar.vert
  ${CMAKE_CURRENT_SOURCE_DIR}/shaders/health_bar.frag
  ${CMAKE_CURRENT_SOURCE_DIR}/shaders/text.vert
  ${CMAKE_CURRENT_SOURCE_DIR}/shaders/text.frag
  ${CMAKE_CURRENT_SOURCE_DIR}/shaders/circle_cull.comp
  ${CMAKE_CURRENT_SOURCE_DIR}/shaders/frustum_cull.comp
  ${CMAKE_CURRENT_SOURCE_DIR}/shaders/hiz_build.comp
  ${CMAKE_CURRENT_SOURCE_DIR}/shaders/texture_stream.comp
  ${CMAKE_CURRENT_SOURCE_DIR}/shaders/procedural_texture.comp
)

set(COMPILED_SHADERS)
foreach(SRC ${SHADER_SOURCES})
  get_filename_component(FNAME ${SRC} NAME)
  get_filename_component(EXT ${SRC} EXT)
  string(TOLOWER ${EXT} EXT_LOWER)
  if (EXT_LOWER STREQUAL ".vert")
    set(SHADER_STAGE vert)
  elseif (EXT_LOWER STREQUAL ".frag")
    set(SHADER_STAGE frag)
  elseif (EXT_LOWER STREQUAL ".comp")
    set(SHADER_STAGE comp)
  else()
    message(FATAL_ERROR "Unsupported shader extension: ${EXT}")
  endif()
  set(SPIRV ${CMAKE_CURRENT_BINARY_DIR}/shaders/${FNAME}.spv)
  list(APPEND COMPILED_SHADERS ${SPIRV})
  if (GLSLC_EXECUTABLE)
    add_custom_command(
      OUTPUT ${SPIRV}
      COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/shaders
      COMMAND ${GLSLC_EXECUTABLE} -fshader-stage=${SHADER_STAGE} ${SRC} -o ${SPIRV}
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
