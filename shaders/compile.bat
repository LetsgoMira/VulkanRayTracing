@echo off
setlocal
if not defined VULKAN_SDK (
    echo VULKAN_SDK is not set. Install the Vulkan SDK and open a new terminal.
    exit /b 1
)
if not exist "%VULKAN_SDK%\Bin\glslangValidator.exe" (
    echo Cannot find glslangValidator.exe under VULKAN_SDK.
    exit /b 1
)
pushd "%~dp0"
rem Use new names so the application cannot silently load legacy SPIR-V.
rem Compile to temporary files first; invalidate stale outputs on failure.
"%VULKAN_SDK%\Bin\glslangValidator.exe" -V --target-env vulkan1.0 shader.vert -o pathtrace.vert.spv.tmp
if errorlevel 1 goto failed
"%VULKAN_SDK%\Bin\glslangValidator.exe" -V --target-env vulkan1.0 shader.frag -o pathtrace.frag.spv.tmp
if errorlevel 1 goto failed
move /y pathtrace.vert.spv.tmp pathtrace.vert.spv >nul
if errorlevel 1 goto failed
move /y pathtrace.frag.spv.tmp pathtrace.frag.spv >nul
if errorlevel 1 goto failed
popd
exit /b 0
:failed
del /q pathtrace.vert.spv.tmp pathtrace.frag.spv.tmp pathtrace.vert.spv pathtrace.frag.spv 2>nul
popd
exit /b 1
