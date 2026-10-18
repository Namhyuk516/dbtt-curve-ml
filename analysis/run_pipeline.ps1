# Re-runs the whole analysis after tuning (results/tuned_*.json must exist).  Four parallel chains; logs in results/logs.
# usage (PowerShell):  .\analysis\run_pipeline.ps1        (from the project root)
$ErrorActionPreference = "Continue"
$root = Split-Path -Parent $PSScriptRoot
$py = "C:\Users\seonh\AppData\Local\Programs\Python\Python312\python.exe"
$env:PYTHONPATH = "C:\Users\seonh\AppData\Local\xgboost320"      # XGBoost 3.2.0 (isolated copy) - see results/README_results.md
$env:PYTHONIOENCODING = "utf-8"; $env:PYTHONUTF8 = "1"
New-Item -ItemType Directory -Force "$root\results\logs" | Out-Null
$chains = @{
  A = @("03_compare.py", "06_figures.py", "11_final_vs_heldout.py", "14_class_errors.py")
  B = @("04_cast.py", "05_shap.py", "12_shap_composite.py", "08_finetune.py 1")
  C = @("13_supp_cv.py ablation_tree", "15_nested_cv.py XGB-direct 20", "15_nested_cv.py XGB-direct-mono 20", "15_nested_cv.py tanh-XGB 20")
  D = @("13_supp_cv.py ablation_nn", "13_supp_cv.py multi_temp")
}
$jobs = foreach ($k in $chains.Keys) {
  $steps = $chains[$k]
  Start-Job -Name "chain$k" -ScriptBlock {
    param($root, $py, $steps, $k)
    $env:PYTHONPATH = "C:\Users\seonh\AppData\Local\xgboost320"; $env:PYTHONIOENCODING = "utf-8"; $env:PYTHONUTF8 = "1"
    Set-Location $root
    foreach ($s in $steps) {
      $name = ($s -replace '[ .]', '_')
      "START $s $(Get-Date -Format HH:mm:ss)" | Out-File "$root\results\logs\chain$k.txt" -Append
      & $py ("analysis\" + $s.Split(' ')[0]) ($s.Split(' ') | Select-Object -Skip 1) > "$root\results\logs\$name.txt" 2> "$root\results\logs\$name.err"
      "END   $s exit=$LASTEXITCODE $(Get-Date -Format HH:mm:ss)" | Out-File "$root\results\logs\chain$k.txt" -Append
    }
  } -ArgumentList $root, $py, $steps, $k
}
Wait-Job $jobs | Out-Null
Get-Content "$root\results\logs\chain*.txt"
