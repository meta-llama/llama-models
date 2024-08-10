@echo off
setlocal enabledelayedexpansion

:: Copyright (c) Meta Platforms, Inc. and affiliates.
:: All rights reserved.
::
:: This source code is licensed under the terms described in the LICENSE file in
:: top-level folder for each specific model found within the models/ directory at
:: the top-level of this source tree.

:: Copyright (c) Meta Platforms, Inc. and affiliates.
:: This software may be used and distributed according to the terms of the Llama 3.1 Community License Agreement.

set /p PRESIGNED_URL="Enter the URL from email: "
set ALL_MODELS_LIST=meta-llama-3.1-405b,meta-llama-3.1-70b,meta-llama-3.1-8b,meta-llama-guard-3-8b,prompt-guard
echo.
echo  **** Model list ***
for %%a in (%ALL_MODELS_LIST:,= %) do (
    echo  -  %%a
)
set /p SELECTED_MODEL="Choose the model to download: "
echo.
echo  Selected model: %SELECTED_MODEL%

set SELECTED_MODELS=
if "%SELECTED_MODEL%"=="meta-llama-3.1-405b" (
    set MODEL_LIST=meta-llama-3.1-405b-instruct-mp16,meta-llama-3.1-405b-instruct-mp8,meta-llama-3.1-405b-instruct-fp8,meta-llama-3.1-405b-mp16,meta-llama-3.1-405b-mp8,meta-llama-3.1-405b-fp8
) else if "%SELECTED_MODEL%"=="meta-llama-3.1-70b" (
    set MODEL_LIST=meta-llama-3.1-70b-instruct,meta-llama-3.1-70b
) else if "%SELECTED_MODEL%"=="meta-llama-3.1-8b" (
    set MODEL_LIST=meta-llama-3.1-8b-instruct,meta-llama-3.1-8b
) else if "%SELECTED_MODEL%"=="meta-llama-guard-3-8b" (
    set MODEL_LIST=meta-llama-guard-3-8b-int8-hf,meta-llama-guard-3-8b
) else if "%SELECTED_MODEL%"=="prompt-guard" (
    set SELECTED_MODELS=prompt-guard
    set MODEL_LIST=
)

if "%SELECTED_MODELS%"=="" (
    echo.
    echo  **** Available models to download: ***
    for %%a in (%MODEL_LIST:,= %) do (
        echo  -  %%a
    )
    set /p SELECTED_MODELS="Enter the list of models to download without spaces or press Enter for all: "
)

set TARGET_FOLDER=.
if not exist %TARGET_FOLDER% mkdir %TARGET_FOLDER%

if "%SELECTED_MODELS%"=="" set SELECTED_MODELS=%MODEL_LIST%

echo Downloading LICENSE and Acceptable Usage Policy
powershell -Command "$url = '%PRESIGNED_URL%'.Replace('*', 'LICENSE'); Invoke-WebRequest -Uri $url -OutFile '%TARGET_FOLDER%\LICENSE'"
powershell -Command "$url = '%PRESIGNED_URL%'.Replace('*', 'USE_POLICY.md'); Invoke-WebRequest -Uri $url -OutFile '%TARGET_FOLDER%\USE_POLICY.md'"

for %%m in (%SELECTED_MODELS:,= %) do (
    set ADDITIONAL_FILES=
    set TOKENIZER_MODEL=1
    set PTH_FILE_CHUNK_COUNT=0
    if "%%m"=="meta-llama-3.1-405b-instruct-mp16" (
        set PTH_FILE_COUNT=15
        set PTH_FILE_CHUNK_COUNT=2
        set MODEL_PATH=Meta-Llama-3.1-405B-Instruct-MP16
    ) else if "%%m"=="meta-llama-3.1-405b-instruct-mp8" (
        set PTH_FILE_COUNT=7
        set PTH_FILE_CHUNK_COUNT=4
        set MODEL_PATH=Meta-Llama-3.1-405B-Instruct-MP8
    ) else if "%%m"=="meta-llama-3.1-405b-instruct-fp8" (
        set PTH_FILE_COUNT=7
        set PTH_FILE_CHUNK_COUNT=3
        set MODEL_PATH=Meta-Llama-3.1-405B-Instruct
        set ADDITIONAL_FILES=fp8_scales_0.pt,fp8_scales_1.pt,fp8_scales_2.pt,fp8_scales_3.pt,fp8_scales_4.pt,fp8_scales_5.pt,fp8_scales_6.pt,fp8_scales_7.pt
    ) else if "%%m"=="meta-llama-3.1-405b-mp16" (
        set PTH_FILE_COUNT=15
        set PTH_FILE_CHUNK_COUNT=2
        set MODEL_PATH=Meta-Llama-3.1-405B-MP16
    ) else if "%%m"=="meta-llama-3.1-405b-mp8" (
        set PTH_FILE_COUNT=7
        set PTH_FILE_CHUNK_COUNT=4
        set MODEL_PATH=Meta-Llama-3.1-405B-MP8
    ) else if "%%m"=="meta-llama-3.1-405b-fp8" (
        set PTH_FILE_COUNT=7
        set PTH_FILE_CHUNK_COUNT=3
        set MODEL_PATH=Meta-Llama-3.1-405B
        set ADDITIONAL_FILES=fp8_scales_0.pt,fp8_scales_1.pt,fp8_scales_2.pt,fp8_scales_3.pt,fp8_scales_4.pt,fp8_scales_5.pt,fp8_scales_6.pt,fp8_scales_7.pt
    ) else if "%%m"=="meta-llama-3.1-70b-instruct" (
        set PTH_FILE_COUNT=7
        set MODEL_PATH=Meta-Llama-3.1-70B-Instruct
    ) else if "%%m"=="meta-llama-3.1-70b" (
        set PTH_FILE_COUNT=7
        set MODEL_PATH=Meta-Llama-3.1-70B
    ) else if "%%m"=="meta-llama-3.1-8b-instruct" (
        set PTH_FILE_COUNT=0
        set MODEL_PATH=Meta-Llama-3.1-8B-Instruct
    ) else if "%%m"=="meta-llama-3.1-8b" (
        set PTH_FILE_COUNT=0
        set MODEL_PATH=Meta-Llama-3.1-8B
    ) else if "%%m"=="meta-llama-guard-3-8b-int8-hf" (
        set PTH_FILE_COUNT=-1
        set MODEL_PATH=Meta-Llama-Guard-3-8B-INT8-HF
        set ADDITIONAL_FILES=generation_config.json,model-00001-of-00002.safetensors,model-00002-of-00002.safetensors,model.safetensors.index.json,special_tokens_map.json,tokenizer_config.json,tokenizer.json
        set TOKENIZER_MODEL=0
    ) else if "%%m"=="meta-llama-guard-3-8b" (
        set PTH_FILE_COUNT=0
        set MODEL_PATH=Meta-Llama-Guard-3-8B
    ) else if "%%m"=="prompt-guard" (
        set PTH_FILE_COUNT=-1
        set MODEL_PATH=Prompt-Guard
        set ADDITIONAL_FILES=model.safetensors,special_tokens_map.json,tokenizer_config.json,tokenizer.json
        set TOKENIZER_MODEL=0
    )

    echo.
    echo ***Downloading !MODEL_PATH!***
    if not exist %TARGET_FOLDER%\!MODEL_PATH! mkdir %TARGET_FOLDER%\!MODEL_PATH!

    if !TOKENIZER_MODEL! equ 1 (
        echo Downloading tokenizer
        powershell -Command "$url = '%PRESIGNED_URL%'.Replace('*', '!MODEL_PATH!/tokenizer.model'); Invoke-WebRequest -Uri $url -OutFile '%TARGET_FOLDER%\!MODEL_PATH!\tokenizer.model'"
    )

    if !PTH_FILE_COUNT! geq 0 (
        for /L %%s in (0,1,!PTH_FILE_COUNT!) do (
            set "s=0%%s"
            set "s=!s:~-2!"
            echo Downloading consolidated.!s!.pth
            if !PTH_FILE_CHUNK_COUNT! gtr 0 (
                set /a start=0
                set /a chunk_size=27000000001
                for /L %%c in (1,1,!PTH_FILE_CHUNK_COUNT!) do (
                    set /a end=!start!+!chunk_size!-1
                    powershell -Command "$ProgressPreference = 'SilentlyContinue'; $url = '%PRESIGNED_URL%'.Replace('*', '!MODEL_PATH!/consolidated.!s!.pth'); Invoke-WebRequest -Uri $url -Headers @{'Range'='bytes=!start!-!end!'} -OutFile '%TARGET_FOLDER%\!MODEL_PATH!\part.%%c.pth'"
                    type %TARGET_FOLDER%\!MODEL_PATH!\part.%%c.pth >> %TARGET_FOLDER%\!MODEL_PATH!\consolidated.!s!.pth
                    del %TARGET_FOLDER%\!MODEL_PATH!\part.%%c.pth
                    set /a start=!end!+1
                )
            ) else (
                powershell -Command "$url = '%PRESIGNED_URL%'.Replace('*', '!MODEL_PATH!/consolidated.!s!.pth'); Invoke-WebRequest -Uri $url -OutFile '%TARGET_FOLDER%\!MODEL_PATH!\consolidated.!s!.pth'"
            )
        )
    )

    for %%a in (!ADDITIONAL_FILES:,= !) do (
        echo Downloading %%a...
        powershell -Command "$url = '%PRESIGNED_URL%'.Replace('*', '!MODEL_PATH!/%%a'); Invoke-WebRequest -Uri $url -OutFile '%TARGET_FOLDER%\!MODEL_PATH!\%%a'"
    )

    if "%%m" neq "prompt-guard" if "%%m" neq "meta-llama-guard-3-8b-int8-hf" (
        echo Downloading params.json...
        powershell -Command "$url = '%PRESIGNED_URL%'.Replace('*', '!MODEL_PATH!/params.json'); Invoke-WebRequest -Uri $url -OutFile '%TARGET_FOLDER%\!MODEL_PATH!\params.json'"
    )
)

endlocal