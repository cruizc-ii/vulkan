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
.PHONY: clean-results
clean-results:
	rm -rf results/**;
.PHONY: clean
clean:
	rm -rf models/design/**;
	rm -rf models/strana/**;
	rm -rf models/loss/**;
	rm -rf models/compare/**;
	find models/loss_csvs/ -delete
	find models/rate_csvs/ -delete
	find results/** -name ".csv" -delete
	git restore models/;
clean-results:
	rm -rf results/**;
docker: build docker_run
build:
	podman build -t vulkan .
docker_run:
	podman run -p 8501:8501 -t vulkan

