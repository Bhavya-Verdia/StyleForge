.PHONY: help dev-server dev-frontend dev lint test docker-build docker-up clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

dev-server: ## Start the mock server
	cd server && USE_MOCK=true uvicorn main:app --reload --port 8000

dev-frontend: ## Start the frontend dev server
	cd frontend && npm run dev

dev: ## Start both server and frontend (requires tmux or two terminals)
	@echo "Run 'make dev-server' and 'make dev-frontend' in separate terminals"

lint: ## Run all linters
	ruff check pipeline/ server/ --ignore E501
	cd frontend && npx oxlint .

docker-build: ## Build Docker images
	docker-compose build

docker-up: ## Start Docker containers
	docker-compose up -d

docker-down: ## Stop Docker containers
	docker-compose down

clean: ## Clean build artifacts
	rm -rf frontend/dist frontend/node_modules
	rm -rf server/__pycache__ pipeline/__pycache__
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
