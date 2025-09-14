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

if(APPLE)
  set_target_properties(battleroyale5 PROPERTIES
    MACOSX_BUNDLE FALSE
    BUILD_WITH_INSTALL_RPATH TRUE
    BUILD_RPATH "/opt/homebrew/lib"
    INSTALL_RPATH "/opt/homebrew/lib"
  )
endif()

