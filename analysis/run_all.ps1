# Full re-computation: tuning (5 models in parallel) -> curve fits -> pipeline (4 chains) -> parameter accuracy -> full-size sensitivity.
# usage: powershell -File analysis\run_all.ps1   (from the project root; XGBoost 3.2.0 isolated copy required)
$root = Split-Path -Parent $PSScriptRoot
$py = "C:\Users\seonh\AppData\Local\Programs\Python\Python312\python.exe"
$env:PYTHONPATH = "C:\Users\seonh\AppData\Local\xgboost320"; $env:PYTHONIOENCODING = "utf-8"; $env:PYTHONUTF8 = "1"
$log = "$root\results\logs"; New-Item -ItemType Directory -Force $log | Out-Null
"START tuning $(Get-Date -Format 'MM-dd HH:mm:ss')" | Out-File "$log\run_all.txt" -Append
$procs = foreach ($j in @(@("XGB-direct", 60), @("XGB-direct-mono", 60), @("tanh-XGB", 30), @("NN-direct", 30), @("tanh-NN", 30))) {
  Start-Process -FilePath $py -ArgumentList "analysis\09_tune.py", $j[0], $j[1] -WorkingDirectory $root -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput "$log\tune_$($j[0]).txt" -RedirectStandardError "$log\tune_$($j[0]).err"
}
$procs | Wait-Process
"END tuning $(Get-Date -Format 'MM-dd HH:mm:ss')" | Out-File "$log\run_all.txt" -Append
Set-Location $root
& $py analysis\01_fit_curves.py > "$log\01_fit_curves_py.txt" 2>&1
& powershell -NoProfile -ExecutionPolicy Bypass -File "$root\analysis\run_pipeline.ps1" > "$log\pipeline_stdout.txt" 2>&1
"END pipeline $(Get-Date -Format 'MM-dd HH:mm:ss')" | Out-File "$log\run_all.txt" -Append
& $py analysis\16_param_accuracy.py > "$log\16_param_accuracy.txt" 2>&1
& $py analysis\13_supp_cv.py fullsize > "$log\13_supp_cv_py_fullsize.txt" 2>&1
"ALL DONE $(Get-Date -Format 'MM-dd HH:mm:ss')" | Out-File "$log\run_all.txt" -Append
