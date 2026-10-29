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
        Write-Host "Running retrieval evaluation benchmark..." -ForegroundColor Cyan
        python mlops/eval/evaluate.py
    }
    "benchmark" {
        Write-Host "Running search performance benchmark..." -ForegroundColor Cyan
        python mlops/benchmarks/benchmark_search.py
    }
    "test" {
        Write-Host "Running test suite..." -ForegroundColor Cyan
        python -m unittest discover tests/ -v
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
        Write-Host "Retrieva Task Runner" -ForegroundColor Yellow
        Write-Host "Usage: .\tasks.ps1 [command]"
        Write-Host "Commands:"
        Write-Host "  install   - Install dependencies"
        Write-Host "  ingest    - Ingest documents from data/ and create BM25 and vector indexes"
        Write-Host "  eval      - Run retrieval evaluation against test dataset"
        Write-Host "  benchmark - Run retrieval latency and throughput benchmark"
        Write-Host "  test      - Run automated unit and integration tests"
        Write-Host "  validate  - Run data validation and sanity checks"
        Write-Host "  up        - Start FastAPI, Streamlit, Redis, Prometheus, and Grafana in Docker"
        Write-Host "  down      - Stop all Docker containers"
        Write-Host "  logs      - Tail Docker container logs"
    }
}
