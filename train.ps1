Param(
  [string]$TrainingFile = "$PSScriptRoot\training_data.txt",
  [int]$ParallelJobs = 5,
  [string]$Endpoint = "http://localhost:3000/api/chat",
  [int]$DelayBetweenRequests = 1000,
  [int]$Repetitions = 1000
)
if (!(Test-Path $TrainingFile)) {
  Write-Host "Training file not found: $TrainingFile"
  exit
}
$allLines = Get-Content $TrainingFile | Where-Object { $_.Trim() -ne "" }
for ($r = 1; $r -le $Repetitions; $r++) {
  Write-Host "Starting training cycle $r"
  $jobGroups = @()
  $linesPerJob = [math]::Ceiling($allLines.Count / $ParallelJobs)
  for ($i = 0; $i -lt $ParallelJobs; $i++) {
    $startIndex = $i * $linesPerJob
    $endIndex = [math]::Min($allLines.Count - 1, $startIndex + $linesPerJob - 1)
    $jobLines = $allLines[$startIndex..$endIndex]
    $job = Start-Job -ScriptBlock {
      param($jobLines, $Endpoint, $Delay)
      foreach ($line in $jobLines) {
        # Simulate multiple users by prepending a random user identifier
        $userId = "User" + (Get-Random -Minimum 1 -Maximum 10)
        $modifiedLine = "$userId: $line"
        $body = @{ message = $modifiedLine } | ConvertTo-Json
        try {
          $response = Invoke-WebRequest -Uri $Endpoint -Method Post -Body $body -ContentType "application/json" -UseBasicParsing
          $result = $response.Content | ConvertFrom-Json
          Write-Host "Sent: $modifiedLine"
          Write-Host "Received: $($result.reply)"
        } catch {
          Write-Host "Error sending: $modifiedLine - $_"
        }
        Start-Sleep -Milliseconds $Delay
      }
    } -ArgumentList $jobLines, $Endpoint, $DelayBetweenRequests
    $jobGroups += $job
  }
  Wait-Job -Job $jobGroups
  Receive-Job -Job $jobGroups | Out-Null
  Start-Sleep -Seconds 5
}
