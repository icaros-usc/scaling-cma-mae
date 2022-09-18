help: ## Print this message.
	@echo "\033[0;1mCommands\033[0m"
	@grep -E '^[.a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[34;1m%-30s\033[0m %s\n", $$1, $$2}'
.PHONY: help

container.sif: container.def requirements.txt dask_config.yml ## The Singularity container. Requires sudo to run.
	singularity build $@ $<

# The value of DISPLAY may depend on your system.
shell: ## Start a shell in the container.
	SINGULARITYENV_DISPLAY=:0 singularity shell --cleanenv --nv --no-home --bind $(PWD) container.sif
shell-bind: ## Start a shell with GUI and with ./results bound to /results.
	SINGULARITYENV_DISPLAY=:0 singularity shell --cleanenv --nv --no-home --bind $(PWD),./results:/results container.sif
.PHONY: shell shell-gui

test: ## Run unit tests.
	pytest src/
.PHONY: test
