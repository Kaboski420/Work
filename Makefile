.PHONY: help install test lint format run docker-build docker-up docker-down clean test-api

help:
	@echo "Available commands:"
	@echo "  make install     - Install dependencies"
	@echo "  make test        - Run tests"
	@echo "  make test-api    - Test API endpoints (requires server running)"
	@echo "  make lint        - Run linters"
	@echo "  make format      - Format code"
	@echo "  make run         - Run the API server"
	@echo "  make docker-up   - Start Docker services"
	@echo "  make docker-down - Stop Docker services"
	@echo "  make clean       - Clean temporary files"

install:
	pip install -r requirements.txt

test:
	pytest tests/ -v --cov=src --cov-report=html

test-api:
	python test_api.py

lint:
	flake8 src/ tests/
	mypy src/

format:
	black src/ tests/
	isort src/ tests/

run:
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

# Docker Compose - prefer modern 'docker compose' (v2), fallback to 'docker-compose' (v1)
DOCKER_COMPOSE := $(shell docker compose version >/dev/null 2>&1 && echo "docker compose" || echo "")
ifeq ($(DOCKER_COMPOSE),)
	DOCKER_COMPOSE := $(shell command -v docker-compose 2>/dev/null || echo "")
endif
ifeq ($(DOCKER_COMPOSE),)
	DOCKER_COMPOSE := $(shell [ -f /Users/hamzashaheen/Library/Python/3.9/bin/docker-compose ] && echo "/Users/hamzashaheen/Library/Python/3.9/bin/docker-compose" || echo "")
endif

docker-build:
	@if [ -z "$(DOCKER_COMPOSE)" ]; then \
		echo "Error: Docker Compose not found. Install Docker Desktop or: pip3 install docker-compose"; \
		exit 1; \
	fi
	$(DOCKER_COMPOSE) build

docker-up:
	@if [ -z "$(DOCKER_COMPOSE)" ]; then \
		echo "Error: Docker Compose not found. Install Docker Desktop or: pip3 install docker-compose"; \
		exit 1; \
	fi
	@echo "Starting Docker services..."
	$(DOCKER_COMPOSE) up -d --pull=missing

docker-down:
	@if [ -z "$(DOCKER_COMPOSE)" ]; then \
		echo "Error: Docker Compose not found. Install Docker Desktop or: pip3 install docker-compose"; \
		exit 1; \
	fi
	$(DOCKER_COMPOSE) down

clean:
	find . -type d -name __pycache__ -exec rm -r {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	rm -rf .pytest_cache
	rm -rf .coverage
	rm -rf htmlcov
	rm -rf dist
	rm -rf build
	rm -rf *.egg-info


