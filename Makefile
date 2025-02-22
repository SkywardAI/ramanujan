bundoora:
	@git submodule update --remote


linter:
	@uv run --active ruff check .

sync:
	@uv sync --active

 