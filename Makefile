.PHONY: help build up down logs clean dev prod test health

# Default target
help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Targets:'
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-15s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

build: ## Build the Docker images
	docker-compose build

up: ## Start the services
	docker-compose up -d

down: ## Stop the services
	docker-compose down

logs: ## Show logs from all services
	docker-compose logs -f

logs-app: ## Show logs from app service only
	docker-compose logs -f app

logs-db: ## Show logs from postgres service only
	docker-compose logs -f postgres

clean: ## Remove containers, networks, and volumes
	docker-compose down -v --remove-orphans
	docker system prune -f

dev: ## Start in development mode with hot reload
	docker-compose -f docker-compose.yml -f docker-compose.override.yml up -d
	@echo "Development environment started. App running on http://localhost:3000"
	@echo "PostgreSQL running on localhost:5432"

prod: ## Start in production mode
	docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
	@echo "Production environment started. App running on http://localhost:3000"
	@echo "PostgreSQL running on localhost:5433"

test: ## Run tests
	docker-compose exec app npm test

docker-test: ## Run Docker integration tests
	node docker-test.js

health: ## Check health of all services
	@echo "Checking application health..."
	@curl -s http://localhost:3000/health | jq . || echo "App health check failed"
	@echo "Checking database connection..."
	@docker-compose exec postgres pg_isready -U solvedbook || echo "Database health check failed"

restart: ## Restart all services
	docker-compose restart

restart-app: ## Restart app service only
	docker-compose restart app

shell: ## Open shell in app container
	docker-compose exec app sh

db-shell: ## Open PostgreSQL shell
	docker-compose exec postgres psql -U solvedbook -d solvedbook

backup: ## Backup database
	docker-compose exec postgres pg_dump -U solvedbook solvedbook > backup_$(shell date +%Y%m%d_%H%M%S).sql

restore: ## Restore database (usage: make restore FILE=backup.sql)
	@if [ -z "$(FILE)" ]; then echo "Usage: make restore FILE=backup.sql"; exit 1; fi
	docker-compose exec -T postgres psql -U solvedbook -d solvedbook < $(FILE)
