.PHONY: help install ingest eval benchmark test validate up down logs

help:
	@echo "Retrieva Project Commands:"
	@echo "  make install    - Install python dependencies"
	@echo "  make ingest     - Run document ingestion pipeline"
	@echo "  make eval       - Run retrieval evaluation against test dataset"
	@echo "  make benchmark  - Run search latency and QPS benchmark"
	@echo "  make test       - Run test suite"
	@echo "  make validate   - Validate data integrity"
	@echo "  make up         - Start Docker stack (API, UI, Redis, Prometheus, Grafana)"
	@echo "  make down       - Stop Docker stack"
	@echo "  make logs       - Tail Docker logs"

install:
	pip install -r requirements.txt

ingest:
	python scripts/ingest.py

eval:
	python mlops/eval/evaluate.py

benchmark:
	python mlops/benchmarks/benchmark_search.py

test:
	python -m unittest discover tests/ -v

validate:
	python mlops/data_validation.py

up:
	docker compose up -d --build

down:
	docker compose down

logs:
	docker compose logs -f
