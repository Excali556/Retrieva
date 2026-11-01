<#
.SYNOPSIS
    Workflow Task Runner for Windows PowerShell
.EXAMPLE
    .\tasks.ps1 test
    .\tasks.ps1 eval
    .\tasks.ps1 up
#>

param(
    [Parameter(Position=0)]
    [ValidateSet("install", "ingest", "eval", "benchmark", "test", "validate", "up", "down", "logs", "help")]
    [string]$Command = "help"
)

switch ($Command) {
    "install" {
        Write-Host "Installing requirements..." -ForegroundColor Cyan
        python -m pip install -r requirements.txt
    }
    "ingest" {
        Write-Host "Running ingestion pipeline..." -ForegroundColor Cyan
        python scripts/ingest.py
    }
    "eval" {
        Write-Host "Running retrieval evaluation benchmark (held-out test split)..." -ForegroundColor Cyan
        python mlops/eval/evaluate.py --split test
    }
    "benchmark" {
        Write-Host "Running search performance benchmark..." -ForegroundColor Cyan
        python mlops/benchmarks/benchmark_search.py
    }
    "test" {
        Write-Host "Running test suite..." -ForegroundColor Cyan
        python -m pytest tests/ -v
    }
    "validate" {
        Write-Host "Validating data integrity..." -ForegroundColor Cyan
        python mlops/data_validation.py
    }
    "up" {
        Write-Host "Starting Docker Compose services..." -ForegroundColor Cyan
        docker compose up -d --build
    }
    "down" {
        Write-Host "Stopping Docker Compose services..." -ForegroundColor Cyan
        docker compose down
    }
    "logs" {
        docker compose logs -f
    }
    default {
        Write-Host "Retrieva Developer Commands (PowerShell):" -ForegroundColor Green
        Write-Host "  .\tasks.ps1 install    - Install Python dependencies"
        Write-Host "  .\tasks.ps1 ingest     - Ingest PDFs and generate BM25 + dense index"
        Write-Host "  .\tasks.ps1 eval       - Run benchmark evaluation on held-out test split"
        Write-Host "  .\tasks.ps1 benchmark  - Run search latency benchmark"
        Write-Host "  .\tasks.ps1 test       - Run pytest test suite"
        Write-Host "  .\tasks.ps1 validate   - Validate document chunk integrity"
        Write-Host "  .\tasks.ps1 up         - Start Docker stack"
        Write-Host "  .\tasks.ps1 down       - Stop Docker stack"
    }
}
