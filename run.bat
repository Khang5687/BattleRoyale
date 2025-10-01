@echo off
setlocal enabledelayedexpansion

REM BattleRoyale5 - Build and Run Script (Windows)
REM This script builds the Vulkan C++ battle royale simulation and runs it

echo [94m^
 BattleRoyale5 - Build and Run Script[0m
echo ============================================

REM Get script directory (project root)
set "PROJECT_ROOT=%~dp0"
set "PROJECT_ROOT=%PROJECT_ROOT:~0,-1%"
set "BUILD_DIR=%PROJECT_ROOT%\build"

echo [93mProject root: %PROJECT_ROOT%[0m

REM Check for required tools
echo [94mChecking dependencies...[0m

where cmake >nul 2>&1
if %errorlevel% neq 0 (
    echo [91mError: cmake not found. Please install CMake.[0m
    exit /b 1
)

REM Check for Visual Studio or Ninja build tools
where cl >nul 2>&1
if %errorlevel% neq 0 (
    echo [93mWarning: MSVC compiler not found in PATH. Checking for Visual Studio...[0m
    if not defined VSCMD_VER (
        echo [91mError: Visual Studio build tools not detected.[0m
        echo Please run this script from a Visual Studio Developer Command Prompt,[0m
        echo or install CMake with MinGW/Ninja build tools.[0m
        exit /b 1
    )
)

REM Check for Vulkan SDK
if not defined VULKAN_SDK (
    echo [91mError: VULKAN_SDK environment variable not set. Please install Vulkan SDK.[0m
    exit /b 1
)

echo [92mDependencies check passed[0m

REM Create build directory if it doesn't exist
if not exist "%BUILD_DIR%" (
    echo [93mCreating build directory...[0m
    mkdir "%BUILD_DIR%"
)

cd /d "%BUILD_DIR%"

REM Check if we need to run cmake configure
set NEED_CONFIGURE=0
if not exist "CMakeCache.txt" (
    set NEED_CONFIGURE=1
    echo [93mNo CMakeCache.txt found, will configure...[0m
)

REM Configure with CMake if needed
if !NEED_CONFIGURE! equ 1 (
    echo [94mConfiguring project with CMake...[0m
    cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_PREFIX_PATH="%PROJECT_ROOT%\vcpkg_installed\x64-windows" "%PROJECT_ROOT%"
    if %errorlevel% neq 0 (
        echo [91mError: CMake configuration failed[0m
        exit /b 1
    )
    echo [92mCMake configuration complete[0m
)

REM Build the project
echo [94mBuilding project...[0m
cmake --build . --config RelWithDebInfo --target battleroyale5
if %errorlevel% neq 0 (
    echo [91mError: Build failed[0m
    exit /b 1
)

REM Check if build was successful
if not exist "%BUILD_DIR%\RelWithDebInfo\battleroyale5.exe" (
    if not exist "%BUILD_DIR%\battleroyale5.exe" (
        echo [91mError: Build failed, executable not found[0m
        exit /b 1
    )
)

echo [92mBuild complete[0m

REM Copy assets and configuration files
echo [94mCopying assets...[0m

REM Copy assets directory
if exist "%PROJECT_ROOT%\assets" (
    xcopy /E /I /Y /Q "%PROJECT_ROOT%\assets" "%BUILD_DIR%\assets" >nul
    echo [92mAssets copied[0m
) else (
    echo [93mWarning: No assets directory found[0m
)

REM Copy stb directory if it exists
if exist "%PROJECT_ROOT%\stb" (
    xcopy /E /I /Y /Q "%PROJECT_ROOT%\stb" "%BUILD_DIR%\stb" >nul
    echo [92mSTB headers copied[0m
)

REM Copy bias.txt if it exists
if exist "%PROJECT_ROOT%\bias.txt" (
    copy /Y "%PROJECT_ROOT%\bias.txt" "%BUILD_DIR%\" >nul
    echo [92mbias.txt copied[0m
)

REM Copy size_factors.txt if it exists
if exist "%PROJECT_ROOT%\size_factors.txt" (
    copy /Y "%PROJECT_ROOT%\size_factors.txt" "%BUILD_DIR%\" >nul
    echo [92msize_factors.txt copied[0m
)

REM Copy vcpkg DLLs to executable directory
if exist "%PROJECT_ROOT%\vcpkg_installed\x64-windows\bin\glfw3.dll" (
    copy /Y "%PROJECT_ROOT%\vcpkg_installed\x64-windows\bin\*.dll" "%BUILD_DIR%\RelWithDebInfo\" >nul 2>&1
    copy /Y "%PROJECT_ROOT%\vcpkg_installed\x64-windows\bin\*.dll" "%BUILD_DIR%\" >nul 2>&1
    echo [92mvcpkg DLLs copied[0m
)

REM Check if shaders were compiled
if exist "%BUILD_DIR%\shaders\circle.vert.spv" (
    if exist "%BUILD_DIR%\shaders\circle.frag.spv" (
        echo [92mShaders compiled successfully[0m
    ) else (
        echo [93mWarning: Shader compilation may have failed[0m
    )
) else (
    echo [93mWarning: Shader compilation may have failed[0m
)

echo ============================================
echo [92mStarting BattleRoyale5...[0m
echo ============================================

REM Run the executable from the project root so relative paths work correctly
cd /d "%PROJECT_ROOT%"

REM Try RelWithDebInfo directory first (MSVC), then build root (MinGW/Ninja)
if exist "%BUILD_DIR%\RelWithDebInfo\battleroyale5.exe" (
    "%BUILD_DIR%\RelWithDebInfo\battleroyale5.exe"
) else (
    "%BUILD_DIR%\battleroyale5.exe"
)