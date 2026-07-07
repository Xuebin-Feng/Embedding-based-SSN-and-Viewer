@echo off
REM =========================================================================
REM Installation and Shortcut Generation Script for SSN Viewer & Tools
REM =========================================================================
setlocal EnableDelayedExpansion

:: Move to the directory containing this batch script (project root)
cd /d "%~dp0"
set "PROJECT_ROOT=%CD%"

echo Setting up shortcuts for SSN Viewer and SSN Tools...
echo Project root: !PROJECT_ROOT!

:: 1. Define icon paths
set "VIEWER_ICON=!PROJECT_ROOT!\src\bin\logos\viewer_logo_large.ico"
set "TOOL_ICON=!PROJECT_ROOT!\src\bin\logos\tool_logo_large.ico"

:: 2. Create Windows Shortcut for SSN_Viewer.bat in the root folder
echo Creating shortcut for SSN_Viewer...
powershell -ExecutionPolicy Bypass -Command "$WshShell = New-Object -ComObject WScript.Shell; $Shortcut = $WshShell.CreateShortcut('!PROJECT_ROOT!\SSN_Viewer.lnk'); $Shortcut.TargetPath = '!PROJECT_ROOT!\src\bin\SSN_Viewer.bat'; $Shortcut.WorkingDirectory = '!PROJECT_ROOT!'; $Shortcut.IconLocation = '!VIEWER_ICON!'; $Shortcut.Save();"
if exist "SSN_Viewer.lnk" (
    echo [OK] Created SSN_Viewer.lnk in project root.
)

:: 3. Create Windows Shortcut for SSN_Tools.bat in the root folder
echo Creating shortcut for SSN_Tools...
powershell -ExecutionPolicy Bypass -Command "$WshShell = New-Object -ComObject WScript.Shell; $Shortcut = $WshShell.CreateShortcut('!PROJECT_ROOT!\SSN_Tools.lnk'); $Shortcut.TargetPath = '!PROJECT_ROOT!\src\bin\SSN_Tools.bat'; $Shortcut.WorkingDirectory = '!PROJECT_ROOT!'; $Shortcut.IconLocation = '!TOOL_ICON!'; $Shortcut.Save();"
if exist "SSN_Tools.lnk" (
    echo [OK] Created SSN_Tools.lnk in project root.
)

:: 4. Optional: Copy shortcuts to Desktop
echo.
choice /M "Would you like to copy these shortcuts to your Desktop"
if %ERRORLEVEL% equ 1 (
    for /f "usebackq tokens=*" %%i in (`powershell -NoProfile -Command "[Environment]::GetFolderPath('CommonDesktopDirectory')"`) do set "COMMON_DESKTOP=%%i"
    for /f "usebackq tokens=*" %%i in (`powershell -NoProfile -Command "[Environment]::GetFolderPath('Desktop')"`) do set "USER_DESKTOP=%%i"
    
    echo Copying shortcuts to Public Desktop at !COMMON_DESKTOP!...
    copy "!PROJECT_ROOT!\SSN_Viewer.lnk" "!COMMON_DESKTOP!\SSN_Viewer.lnk" /Y >nul 2>nul
    
    if !ERRORLEVEL! neq 0 (
        echo [INFO] Writing to Public Desktop requires Administrator privileges - Access Denied.
        echo Falling back: Copying shortcuts to your personal Desktop at !USER_DESKTOP!...
        copy "!PROJECT_ROOT!\SSN_Viewer.lnk" "!USER_DESKTOP!\SSN_Viewer.lnk" /Y >nul
        copy "!PROJECT_ROOT!\SSN_Tools.lnk" "!USER_DESKTOP!\SSN_Tools.lnk" /Y >nul
        echo [OK] Shortcuts successfully copied to your personal Desktop!
    ) else (
        copy "!PROJECT_ROOT!\SSN_Tools.lnk" "!COMMON_DESKTOP!\SSN_Tools.lnk" /Y >nul 2>nul
        echo [OK] Shortcuts successfully copied to the Public Desktop!
    )
)

echo.
echo Setup Complete! You can now run SSN Viewer and Tools using the root-level shortcuts.
pause
