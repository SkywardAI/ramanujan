bundoora:
	@git submodule update --remote


linter:
	@uv run ruff check .

sync:
	@uv sync --active

 