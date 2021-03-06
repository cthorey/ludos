CUDAGL_TAG='10.1-devel-ubuntu18.04'
SHELL := bash
.SHELLFLAGS := -eu -o pipefail -c
.DEFAULT_GOAL := help
.DELETE_ON_ERROR:
.SUFFIXES:

REPO = ludos
DOCKERFILE = ./build/Dockerfile
DOCKER_BUILD_ARGS = --build-arg CUDAGL_TAG=$(CUDAGL_TAG)
VERSION = latest

.PHONY: help
help:
	$(info Available make targets:)
	@egrep '^(.+)\:\ ##\ (.+)' ${MAKEFILE_LIST} | column -t -c 2 -s ':#'

.PHONY: build
build: ## Build docker image
	$(info *** Building docker image: $(REPO):$(VERSION))
	@docker build \
    $(DOCKER_BUILD_ARGS) \
		--tag $(REPO):$(VERSION) \
		--file $(DOCKERFILE) \
		.

.PHONY: notebook
notebook: ## Launch a notebook
	$(info *** Launch a serving server on requested port)
	@docker run --rm -ti \
    --name ludos_notebook \
		--volume ~/.aws:/root/.aws \
    --volume ~/workdir/$(REPO)/.pgpass:/root/.pgpass \
    --volume ~/.clearml.conf:/root/clearml.conf \
		--volume ~/workdir/$(REPO):/workdir \
		--volume ~/workdir/$(REPO)/training_config:/workdir/training_config \
    --volume /mnt/hdd/data:/workdir/data \
    --volume /mnt/hdd/models:/workdir/models \
		--volume /mnt/hdd/models/cache:/root/.cache \
		--detach \
		--shm-size "32G" \
		--publish 8887:8888 \
		$(REPO):$(VERSION) /workdir/scripts/run_jupyter.sh

.PHONY: serve
serve: ## Launch a serving server on requested port
	$(info *** Launch a serving server on requested port)
	@docker run --rm -ti \
		--volume /mnt/hdd/omatai/models:/workdir/models \
		--volume ~/workdir/$(REPO):/workdir \
		--volume ~/.aws:/root/.aws \
    --volume ~/workdir/$(REPO)/.pgpass:/root/.pgpass \
		--publish 5557:5557 \
		--detach \
    --name ludos_server \
		$(REPO):$(VERSION) python3 scripts/serve.py --host=0.0.0.0 --port=5557
