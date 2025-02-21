bundoora:
	@git submodule update --remote


linter:
	@uv run ruff check .