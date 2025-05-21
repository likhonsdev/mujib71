# Configuration
$API_KEY = "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"
$MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
$AGENT_COUNT = 2

$WORKDIR = Join-Path $PSScriptRoot "ai-agent"
$LOGDIR = Join-Path $WORKDIR "outputs\logs"
$PROMPT_FILE = Join-Path $WORKDIR "system_prompt.mdx"
$TASK_FILE = Join-Path $WORKDIR "task_context.md"

# Create directories
New-Item -ItemType Directory -Force -Path $LOGDIR | Out-Null
New-Item -ItemType Directory -Force -Path $WORKDIR | Out-Null

# Initialize prompt file if missing
if (-not (Test-Path $PROMPT_FILE)) {
    $initialPrompt = '<Plan>' + [Environment]::NewLine
    $initialPrompt += 'You are AI coding agents focused on building a Bengali code + NLP LLM.' + [Environment]::NewLine
    $initialPrompt += 'Output commands inside <Actions> blocks, analyses inside <Task> blocks.' + [Environment]::NewLine
    $initialPrompt += 'After command execution, output results inside <TaskResult> blocks.' + [Environment]::NewLine
    $initialPrompt += '</Plan>' + [Environment]::NewLine + [Environment]::NewLine
    $initialPrompt += '<Actions>' + [Environment]::NewLine
    $initialPrompt += 'echo "Starting initial training setup..."' + [Environment]::NewLine
    $initialPrompt += '# Dummy start command for training' + [Environment]::NewLine
    $initialPrompt += 'echo "Training started."' + [Environment]::NewLine
    $initialPrompt += '</Actions>' + [Environment]::NewLine + [Environment]::NewLine
    $initialPrompt += '<Task>' + [Environment]::NewLine
    $initialPrompt += 'Review output and plan next steps to create a Bengali LLM focused on code + Bangla NLP.' + [Environment]::NewLine
    $initialPrompt += '</Task>'
    
    Set-Content -Path $PROMPT_FILE -Value $initialPrompt
}

# Initialize task file if missing
if (-not (Test-Path $TASK_FILE)) {
    "" | Set-Content $TASK_FILE
}

# Copy training script if missing
$TRAIN_SCRIPT = Join-Path $WORKDIR "train.py"
if (-not (Test-Path $TRAIN_SCRIPT)) {
    Copy-Item -Path (Join-Path $PSScriptRoot "train.py") -Destination $TRAIN_SCRIPT
}

# Function to call Groq API with streaming
function Invoke-GroqAPI {
    param (
        [string]$Prompt,
        [string]$AgentId
    )
    
    $headers = @{
        "Authorization" = "Bearer " + $API_KEY
        "Content-Type" = "application/json"
    }
    
    $body = @{
        model = $MODEL
        messages = @(
            @{
                role = "system"
                content = $Prompt
            }
        )
        temperature = 1
        max_completion_tokens = 1024
        top_p = 1
        stream = $true
    } | ConvertTo-Json

    try {
        $apiUrl = "https://api.groq.com/openai/v1/chat/completions"
        $response = Invoke-RestMethod -Uri $apiUrl -Method Post -Headers $headers -Body $body -ContentType "application/json"
        
        # Process streaming response
        $fullResponse = ""
        foreach ($chunk in $response.choices[0].delta.content) {
            if ($null -ne $chunk) {
                $fullResponse += $chunk
                Write-Host ("🤖 Agent " + $AgentId + ": " + $chunk) -NoNewline
            }
        }
        Write-Host ""
        return $fullResponse
    }
    catch {
        Write-Host "❌ Error calling Groq API: $_" -ForegroundColor Red
        return $null
    }
}

# Function to extract and run actions
function Invoke-Actions {
    param (
        [string]$Response,
        [string]$AgentId
    )
    
    if ($Response -match '(?s)<Actions>(.*?)</Actions>') {
        $actions = $matches[1].Trim()
        if ($actions) {
            Write-Host ("⚡ Agent " + $AgentId + " executing <Actions>...")
            $actionScriptName = "run_actions_" + $AgentId + ".ps1"
            $actionScript = Join-Path $WORKDIR $actionScriptName
            $actions | Set-Content $actionScript
            
            $logFileName = "actions_agent_" + $AgentId + ".log"
            $logFile = Join-Path $LOGDIR $logFileName
            & $actionScript *>&1 | Tee-Object -Path $logFile
        }
    }
    else {
        Write-Host ("ℹ️ Agent " + $AgentId + " found no <Actions>.")
        $logFileName = "actions_agent_" + $AgentId + ".log"
        "" | Set-Content (Join-Path $LOGDIR $logFileName)
    }
}

# Function to append task result
function Add-TaskResult {
    param (
        [string]$AgentId
    )
    
    $logFileName = "actions_agent_" + $AgentId + ".log"
    $logFile = Join-Path $LOGDIR $logFileName
    if (Test-Path $logFile) {
        $result = Get-Content $logFile -Tail 50 | Out-String
        $taskResult = [Environment]::NewLine + '<TaskResult>' + [Environment]::NewLine
        $taskResult += $result
        $taskResult += '</TaskResult>'
        
        Add-Content -Path $TASK_FILE -Value $taskResult
        Write-Host ("✍️ Agent " + $AgentId + " appended <TaskResult>.")
    }
}

# Main loop with multi-agent coordination
Write-Host "🚀 Starting multi-agent AI loop with $AGENT_COUNT agents..."

$stopLoop = $false
while (-not $stopLoop) {
    $promptCombined = Get-Content $PROMPT_FILE, $TASK_FILE | Out-String
    
    # Create array to hold jobs
    $jobs = @()
    
    # Start agents in parallel
    1..$AGENT_COUNT | ForEach-Object {
        $agentId = $_
        $workdir = $WORKDIR
        $logdir = $LOGDIR
        $apiKey = $API_KEY
        $model = $MODEL
        
        $jobs += Start-Job -ScriptBlock {
            param($promptCombined, $agentId, $workdir, $logdir, $apiKey, $model)
            
            # Recreate functions in job scope
            function Invoke-GroqAPI {
                param($Prompt, $AgentId)
                $headers = @{
                    "Authorization" = "Bearer " + $apiKey
                    "Content-Type" = "application/json"
                }
                
                $body = @{
                    model = $model
                    messages = @(
                        @{
                            role = "system"
                            content = $Prompt
                        }
                    )
                    temperature = 1
                    max_completion_tokens = 1024
                    top_p = 1
                    stream = $true
                } | ConvertTo-Json

                try {
                    # Add hosts entry
                    $hostsPath = "$env:SystemRoot\System32\drivers\etc\hosts"
                    $hostEntry = "104.198.40.119 groq-api.local"
                    
                    # Check if entry exists
                    $hostsContent = Get-Content $hostsPath
                    if ($hostsContent -notcontains $hostEntry) {
                        Add-Content -Path $hostsPath -Value "`n$hostEntry" -Force
                    }
                    
                    # Configure TLS
                    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
                    [Net.ServicePointManager]::ServerCertificateValidationCallback = {$true}
                    
                    # Make request
                    $headers = @{
                        "Authorization" = "Bearer $apiKey"
                        "Content-Type" = "application/json"
                        "Host" = "api.groq.com"
                    }
                    
                    $apiUrl = "https://groq-api.local/v1/chat/completions"
                    $response = Invoke-RestMethod -Uri $apiUrl -Method Post -Headers $headers -Body $body -ContentType "application/json"
                    
                    $fullResponse = ""
                    foreach ($chunk in $response.choices[0].delta.content) {
                        if ($null -ne $chunk) {
                            $fullResponse += $chunk
                            Write-Host ("🤖 Agent " + $AgentId + ": " + $chunk) -NoNewline
                        }
                    }
                    Write-Host ""
                    return $fullResponse
                }
                catch {
                    Write-Host ("❌ Error calling Groq API: " + $_.Exception.Message) -ForegroundColor Red
                    # Stop the loop on API errors
                    return "<Done>"
                }
            }

            function Invoke-Actions {
                param($Response, $AgentId)
                if ($Response -match '(?s)<Actions>(.*?)</Actions>') {
                    $actions = $matches[1].Trim()
                    if ($actions) {
                        Write-Host ("⚡ Agent " + $AgentId + " executing <Actions>...")
                        $actionScriptName = "run_actions_" + $AgentId + ".ps1"
                        $actionScript = Join-Path $workdir $actionScriptName
                        $actions | Set-Content $actionScript
                        
                        $logFileName = "actions_agent_" + $AgentId + ".log"
                        $logFile = Join-Path $logdir $logFileName
                        & $actionScript *>&1 | Tee-Object -Path $logFile
                    }
                }
                else {
                    Write-Host ("ℹ️ Agent " + $AgentId + " found no <Actions>.")
                    $logFileName = "actions_agent_" + $AgentId + ".log"
                    "" | Set-Content (Join-Path $logdir $logFileName)
                }
            }

            function Add-TaskResult {
                param($AgentId)
                $logFile = Join-Path $logdir ('actions_agent_' + $AgentId + '.log')
                if (Test-Path $logFile) {
                    $result = Get-Content $logFile -Tail 50 | Out-String
                    $taskResult = [Environment]::NewLine + '<TaskResult>' + [Environment]::NewLine
                    $taskResult += $result
                    $taskResult += '</TaskResult>'
                    
                    Add-Content -Path (Join-Path $workdir 'task_context.md') -Value $taskResult
                    Write-Host ('✍️ Agent ' + $AgentId + ' appended <TaskResult>.')
                }
            }
            
            Write-Host ("🤖 Agent " + $agentId + " sending prompt to Groq API...")
            $response = Invoke-GroqAPI -Prompt $promptCombined -AgentId $agentId
            
            if ($response) {
                $responseFileName = "agent_" + $agentId + "_response.txt"
                $response | Set-Content (Join-Path $logdir $responseFileName)
                
                Invoke-Actions -Response $response -AgentId $agentId
                Add-TaskResult -AgentId $agentId
                
                # Check for completion
                if ($response -match '<Done>') {
                    Write-Host ("✅ Agent " + $agentId + " indicated completion.")
                    return $true
                }
            }
            return $false
        } -ArgumentList $promptCombined, $agentId, $workdir, $logdir, $apiKey, $model
    }
    
    # Wait for all jobs and get results
    $results = $jobs | Wait-Job | Receive-Job
    $jobs | Remove-Job
    
    # Check if any agent indicated completion
    if ($results -contains $true) {
        Write-Host "🚀 Stopping AI loop as <Done> was detected."
        $stopLoop = $true
    }
    
    Start-Sleep -Seconds 2
}

Write-Host "🎉 All agents completed."
