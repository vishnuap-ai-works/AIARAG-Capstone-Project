install:
	pip install -e .

test:
	pytest tests/

run-ui:
	streamlit run ui/app.py

run-api:
	uvicorn api.main:app --reload

format:
	black src tests api
	isort src tests api
