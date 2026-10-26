.PHONY: install lint types test check eval up down
install:
	pip install -e ".[dev]"
	pre-commit install
lint:
	ruff check .
	ruff format --check .
types:
	mypy src
test:
	pytest
check: lint types test
eval:
	python -m eval.run
up:
	docker compose up -d
down:
	docker compose down
