$root = Split-Path -Parent $PSScriptRoot
$backend = Start-Process -FilePath "python" -ArgumentList "$root\demo_backend\server.py" -PassThru
Write-Host "Backend running (pid $($backend.Id))."

Set-Location "$root\demo_app"
if (-not (Test-Path node_modules)) {
  npm install
}

npm run dev

Stop-Process -Id $backend.Id
