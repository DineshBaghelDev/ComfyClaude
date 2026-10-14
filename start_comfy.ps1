# Starts the ComfyUI server headless (no Desktop window) using the Desktop install
# and the shared models folder. UI: http://127.0.0.1:8188
$install = 'E:\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI'
$shared  = 'E:\Comfy-Desktop\ComfyUI-Shared'
Set-Location $install
& "$install\.venv\Scripts\python.exe" main.py `
    --port 8188 `
    --enable-manager `
    --extra-model-paths-config "$PSScriptRoot\comfy_paths.yaml" `
    --input-directory "$shared\input" `
    --output-directory "$shared\output" @args
