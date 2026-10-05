.PHONY: install data train evaluate all test api docker clean

install:
	pip install -e ".[dev]"

data:
	python -m blastograde.cli generate

train:
	python -m blastograde.cli train

evaluate:
	python -m blastograde.cli evaluate

all:
	python -m blastograde.cli all

test:
	pytest -q

api:
	uvicorn blastograde.api.main:app --host 0.0.0.0 --port 8000

docker:
	docker compose up --build

clean:
	rm -rf data/synthetic reports/history.json .pytest_cache
