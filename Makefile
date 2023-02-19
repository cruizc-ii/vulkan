.DEFAULT_GOAL := run
run: 
	@streamlit run lab.py;
# .PHONY: lint
# lint:
#     python3 -m pylint --version
#     python3 -m pylint src## Run tests using pytest
# .PHONY: test
test:
	python3 -m pytest --version
	python3 -m pytest tests
# .PHONY: black
# black:
#     python3 -m black --version
#     python3 -m black .## Run ci part
# .PHONY: ci
#     ci: precommit lint test
