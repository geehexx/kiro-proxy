.PHONY: test test-unit test-integration lint fmt typecheck

VENV := .venv/bin

test:
	$(VENV)/pytest tests/ -q

test-unit:
	$(VENV)/pytest tests/unit/ -q

test-integration:
	$(VENV)/pytest tests/integration/ -q

lint:
	$(VENV)/ruff check kiro/ tests/

fmt:
	$(VENV)/ruff format kiro/ tests/

typecheck:
	$(VENV)/pyright kiro/
