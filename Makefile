template-Makefile:
	curl https://raw.githubusercontent.com/quintenroets/package-dev-tools/refs/heads/main/template-Makefile -o template-Makefile

include template-Makefile

.venv/bin/python:
	command -v uv >/dev/null 2>&1 || (curl -LsSf https://astral.sh/uv/install.sh | sh)
	python_version=$$(grep "requires-python" pyproject.toml | sed -E 's/.*>=(3\.[0-9]+).*/\1/') ; \
	uv sync --dev --all-extras --python python$$python_version
