.PHONY: publish
publish: clean build
	uv publish -u __token__ -p `dotenv -f .env get PYPI_TOKEN`

.PHONY: build
build:
	uv build
	uv pip install -e .

.PHONY: clean
clean:
	rm -rf dist
	rm -rf argo_ai.egg-info
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -rf {} +
