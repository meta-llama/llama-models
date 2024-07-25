<#
.SYNOPSIS
This script downloads selected models from a presigned URL.
#>

# Ensure the script stops on errors
$ErrorActionPreference = "Stop"

# Read the presigned URL from the user
$PRESIGNED_URL = Read-Host -Prompt "Enter the URL from email"

$ALL_MODELS_LIST = "meta-llama-3.1-405b,meta-llama-3.1-70b,meta-llama-3.1-8b,meta-llama-guard-3-8b,prompt-guard"

Write-Output "`n **** Model list ***"
$ALL_MODELS_LIST.Split(',') | ForEach-Object { Write-Output " -  $_" }

$SELECTED_MODEL = Read-Host -Prompt "Choose the model to download"
Write-Output "`n Selected model: $SELECTED_MODEL `n"

$SELECTED_MODELS = ""
$MODEL_LIST = ""

switch ($SELECTED_MODEL) {
    "meta-llama-3.1-405b" { $MODEL_LIST = "meta-llama-3.1-405b-instruct-mp16,meta-llama-3.1-405b-instruct-mp8,meta-llama-3.1-405b-instruct-fp8,meta-llama-3.1-405b-mp16,meta-llama-3.1-405b-mp8,meta-llama-3.1-405b-fp8" }
    "meta-llama-3.1-70b" { $MODEL_LIST = "meta-llama-3.1-70b-instruct,meta-llama-3.1-70b" }
    "meta-llama-3.1-8b" { $MODEL_LIST = "meta-llama-3.1-8b-instruct,meta-llama-3.1-8b" }
    "meta-llama-guard-3-8b" { $MODEL_LIST = "meta-llama-guard-3-8b-int8-hf,meta-llama-guard-3-8b" }
    "prompt-guard" { $SELECTED_MODELS = "prompt-guard" }
}

if ([string]::IsNullOrEmpty($SELECTED_MODELS)) {
    Write-Output "`n **** Available models to download: ***"
    $MODEL_LIST.Split(',') | ForEach-Object { Write-Output " -  $_" }
    $SELECTED_MODELS = Read-Host -Prompt "Enter the list of models to download without spaces or press Enter for all"
}

$TARGET_FOLDER = "."  # where all files should end up
New-Item -ItemType Directory -Path $TARGET_FOLDER -Force | Out-Null

if ([string]::IsNullOrEmpty($SELECTED_MODELS)) {
    $SELECTED_MODELS = $MODEL_LIST
}

if ($SELECTED_MODEL -eq "meta-llama-3.1-405b") {
    Write-Output "`nModel requires significant storage and computational resources, occupying approximately 750GB of disk storage space and necessitating two nodes on MP16 for inferencing.`n"
    $ACK = Read-Host -Prompt "Enter Y to continue"
    if ($ACK -ne 'Y' -and $ACK -ne 'y') {
        Write-Output "Exiting..."
        exit 1
    }
}

Write-Output "Downloading LICENSE and Acceptable Usage Policy"
Invoke-WebRequest -Uri $PRESIGNED_URL.Replace('*', "LICENSE") -OutFile "$TARGET_FOLDER/LICENSE"
Invoke-WebRequest -Uri $PRESIGNED_URL.Replace('*', "USE_POLICY.md") -OutFile "$TARGET_FOLDER/USE_POLICY.md"

$SELECTED_MODELS.Split(',') | ForEach-Object {
    $m = $_
    $ADDITIONAL_FILES = ""
    $TOKENIZER_MODEL = $true
    $PTH_FILE_COUNT = 0
    $MODEL_PATH = ""

    switch ($m) {
        "meta-llama-3.1-405b-instruct-mp16" { $PTH_FILE_COUNT = 15; $MODEL_PATH = "Meta-Llama-3.1-405B-Instruct-MP16" }
        "meta-llama-3.1-405b-instruct-mp8" { $PTH_FILE_COUNT = 7; $MODEL_PATH = "Meta-Llama-3.1-405B-Instruct-MP8" }
        "meta-llama-3.1-405b-instruct-fp8" { $PTH_FILE_COUNT = 7; $MODEL_PATH = "Meta-Llama-3.1-405B-Instruct"; $ADDITIONAL_FILES = "fp8_scales_0.pt,fp8_scales_1.pt,fp8_scales_2.pt,fp8_scales_3.pt,fp8_scales_4.pt,fp8_scales_5.pt,fp8_scales_6.pt,fp8_scales_7.pt" }
        "meta-llama-3.1-405b-mp16" { $PTH_FILE_COUNT = 15; $MODEL_PATH = "Meta-Llama-3.1-405B-MP16" }
        "meta-llama-3.1-405b-mp8" { $PTH_FILE_COUNT = 7; $MODEL_PATH = "Meta-Llama-3.1-405B-MP8" }
        "meta-llama-3.1-405b-fp8" { $PTH_FILE_COUNT = 7; $MODEL_PATH = "Meta-Llama-3.1-405B" }
        "meta-llama-3.1-70b-instruct" { $PTH_FILE_COUNT = 7; $MODEL_PATH = "Meta-Llama-3.1-70B-Instruct" }
        "meta-llama-3.1-70b" { $PTH_FILE_COUNT = 7; $MODEL_PATH = "Meta-Llama-3.1-70B" }
        "meta-llama-3.1-8b-instruct" { $PTH_FILE_COUNT = 0; $MODEL_PATH = "Meta-Llama-3.1-8B-Instruct" }
        "meta-llama-3.1-8b" { $PTH_FILE_COUNT = 0; $MODEL_PATH = "Meta-Llama-3.1-8B" }
        "meta-llama-guard-3-8b-int8-hf" { $PTH_FILE_COUNT = -1; $MODEL_PATH = "Meta-Llama-Guard-3-8B-INT8-HF"; $ADDITIONAL_FILES = "generation_config.json,model-00001-of-00002.safetensors,model-00002-of-00002.safetensors,model.safetensors.index.json,special_tokens_map.json,tokenizer_config.json,tokenizer.json"; $TOKENIZER_MODEL = $false }
        "meta-llama-guard-3-8b" { $PTH_FILE_COUNT = 0; $MODEL_PATH = "Meta-Llama-Guard-3-8B" }
        "prompt-guard" { $PTH_FILE_COUNT = -1; $MODEL_PATH = "Prompt-Guard"; $ADDITIONAL_FILES = "model.safetensors,special_tokens_map.json,tokenizer_config.json,tokenizer.json"; $TOKENIZER_MODEL = $false }
    }

    Write-Output "`n***Downloading $MODEL_PATH***`n"
    New-Item -ItemType Directory -Path "$TARGET_FOLDER/$MODEL_PATH" -Force | Out-Null

    if ($TOKENIZER_MODEL) {
        Write-Output "Downloading tokenizer"
        Invoke-WebRequest -Uri $PRESIGNED_URL.Replace('*', "$MODEL_PATH/tokenizer.model") -OutFile "$TARGET_FOLDER/$MODEL_PATH/tokenizer.model"
    }

    if ($PTH_FILE_COUNT -ge 0) {
        for ($s = 0; $s -le $PTH_FILE_COUNT; $s++) {
            $fileNumber = "{0:D2}" -f $s
            Write-Output "Downloading consolidated.$fileNumber.pth"
            Invoke-WebRequest -Uri $PRESIGNED_URL.Replace('*', "$MODEL_PATH/consolidated.$fileNumber.pth") -OutFile "$TARGET_FOLDER/$MODEL_PATH/consolidated.$fileNumber.pth"
        }
    }

    if ($ADDITIONAL_FILES) {
        $ADDITIONAL_FILES.Split(',') | ForEach-Object {
            Write-Output "Downloading $_..."
            Invoke-WebRequest -Uri $PRESIGNED_URL.Replace('*', "$MODEL_PATH/$_") -OutFile "$TARGET_FOLDER/$MODEL_PATH/$_"
        }
    }

    if ($m -ne "prompt-guard" -and $m -ne "meta-llama-guard-3-8b-int8-hf") {
        Write-Output "Downloading params.json..."
        Invoke-WebRequest -Uri $PRESIGNED_URL.Replace('*', "$MODEL_PATH/params.json") -OutFile "$TARGET_FOLDER/$MODEL_PATH/params.json"
    }
}
