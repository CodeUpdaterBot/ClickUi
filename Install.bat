@echo off
REM ------------------------------------------------------------------------
REM  1. Move to the folder where this script is located.
REM ------------------------------------------------------------------------
cd /d "%~dp0"
set "ENV_NAME=click_ui"
set "INSTALL_MARKER=install_complete.txt"

REM ------------------------------------------------------------------------
REM  2. Check if ClickUi folder exists; if not, download (ZIP) from GitHub.
REM ------------------------------------------------------------------------
IF NOT EXIST "ClickUi" (
    echo [INFO] "ClickUi" folder not found. Attempting to download from GitHub...
    
    REM 1) Download the ZIP (using -L to follow redirects)
    curl -L "https://github.com/CodeUpdaterBot/ClickUi/archive/refs/heads/main.zip" -o "ClickUi.zip"
    IF ERRORLEVEL 1 (
        echo [ERROR] Failed to download repository ZIP from GitHub.
        echo Please manually download from:
        echo   https://github.com/CodeUpdaterBot/ClickUi/archive/refs/heads/main.zip
        pause
        goto end
    )
    echo [OK] Successfully downloaded "ClickUi.zip".

    REM 2) Extract the ZIP using PowerShell's Expand-Archive command
    echo [INFO] Extracting archive...
    powershell -Command "Expand-Archive -Path 'ClickUi.zip' -DestinationPath '.'" 
    IF ERRORLEVEL 1 (
        echo [ERROR] Failed to extract "ClickUi.zip".
        pause
        goto end
    )
    echo [OK] Archive extracted.

    REM 3) Rename "ClickUi-main" to "ClickUi"
    rename "ClickUi-main" "ClickUi"
    del "ClickUi.zip"

    echo [OK] Successfully fetched repository into "ClickUi" folder.
) ELSE (
    echo [INFO] "ClickUi" folder already exists. Skipping download...
)


REM ------------------------------------------------------------------------
REM  3. Change directory into ClickUi where conda_packages.txt, etc. exist.
REM ------------------------------------------------------------------------
cd ClickUi

REM ------------------------------------------------------------------------
REM  4. If we are NOT in post-install mode, do the normal installer steps.
REM ------------------------------------------------------------------------
if NOT "%1"=="--post-install" (

    echo.
    echo ===================================================
    echo   Checking if installation has already occurred...
    echo ===================================================
    echo.

    IF EXIST "%INSTALL_MARKER%" (
        echo [INFO] Installation marker "%INSTALL_MARKER%" found.
        echo [INFO] Skipping installer steps...
    ) ELSE (
        echo [INFO] No installation marker found. Proceeding with installer steps...
        echo.

        REM -----------------------------------------------------
        REM STEP 1: Check for Conda Installation
        REM -----------------------------------------------------
        echo [STEP 1] Checking for Conda...
        conda --version >nul 2>&1
        IF ERRORLEVEL 1 (
            echo [ERROR] Conda is not found or not on PATH.
            echo Please install Anaconda first (for all users, add to PATH) and re-run this script: https://www.anaconda.com/download/success.
            pause
            goto end
        ) ELSE (
            echo [OK] Conda is installed.
        )
        echo.

        REM -----------------------------------------------------
        REM STEP 2: Check or Create the Environment
        REM -----------------------------------------------------
        echo [STEP 2] Checking for conda environment "%ENV_NAME%"...
        conda env list | findstr /i "%ENV_NAME%" >nul 2>&1
        IF ERRORLEVEL 1 (
            echo [INFO] Environment "%ENV_NAME%" not found; creating now...
            conda create -n "%ENV_NAME%" --file conda_packages.txt -y
            IF ERRORLEVEL 1 (
                echo [ERROR] Failed to create environment "%ENV_NAME%".
                pause
                goto end
            )
            echo [OK] Environment "%ENV_NAME%" created.
        ) ELSE (
            echo [OK] Environment "%ENV_NAME%" already exists.
        )
        echo.

        REM -----------------------------------------------------
        REM STEP 3: Create Installation Marker
        REM -----------------------------------------------------
        echo [INFO] Creating installation marker file.
        echo Installation Complete > "%INSTALL_MARKER%"
    )

    REM ---------------------------------------------------------
    REM Restart in post-install mode (in the SAME cmd window).
    REM ---------------------------------------------------------
    echo [INFO] Proceeding to post-install steps...
    call "%~f0" --post-install
    goto end
)

REM ------------------------------------------------------------------------
REM 5. Post-Install Phase (actual environment activation & run).
REM ------------------------------------------------------------------------
:post_install
echo [INFO] Running in post-install mode.

REM Name of the conda environment we used above:

echo [INFO] Activating environment "%ENV_NAME%"...
call conda activate "%ENV_NAME%"
IF "%CONDA_PREFIX%"=="" (
    echo [ERROR] Failed to activate environment "%ENV_NAME%" in post-install.
    pause
    goto end
)
echo [INFO] Environment activated: %CONDA_PREFIX%
echo.

REM -----------------------------------------------------
REM STEP 4: Installing required pip packages via requirements.txt
REM -----------------------------------------------------
echo [STEP 4] Installing required packages via pip...
IF EXIST requirements.txt (
    pip install -r requirements.txt
    IF ERRORLEVEL 1 (
        echo [ERROR] Failed to install packages from requirements.txt.
        pause
        goto end
    )
) ELSE (
    echo [WARNING] No requirements.txt found. Skipping pip package installation.
)
echo [OK] Pip packages installed.
echo.

REM -----------------------------------------------------
REM STEP 5: Launch the Application
REM -----------------------------------------------------
echo ===================================================
echo   Starting the application...
echo ===================================================
echo.

call python clickui.py
IF ERRORLEVEL 1 (
    echo [ERROR] The application encountered an error during execution.
    pause
    goto end
)

echo.
echo [OK] Application exited successfully.

:end
echo.
echo [INFO] Press any key to continue...
pause
REM Prevents window from closing:
cmd /k
