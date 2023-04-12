.DEFAULT_GOAL := run
run: 
	@streamlit run lab.py;
# .PHONY: lint
# lint:
#     python3 -m pylint --version
#     python3 -m pylint src## Run tests using pytest
.PHONY: test
test:
	python3 -m pytest --version
	python3 -m pytest tests
.PHONY: black
black:
	python3 -m black --version
	python3 -m black .
.PHONY: ci
ci: black test
.PHONY: clean
clean:
	rm -rf models/design/**;
	rm -rf models/hazard/**;
	rm -rf models/strana/**;
	rm -rf models/loss/**;
	rm -rf models/compare/**;
	rm -rf models/loss_csvs/**;
	rm -rf models/rate_csvs/**;
	rm -rf results/**;
	git restore models/;
