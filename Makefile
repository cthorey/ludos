CUDAGL_TAG='10.1-devel-ubuntu18.04'
SHELL := bash
.SHELLFLAGS := -eu -o pipefail -c
.DEFAULT_GOAL := help
.DELETE_ON_ERROR:
.SUFFIXES:

REPO = hubmap
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
		--volume ~/.aws:/root/.aws \
    --volume ~/.pgpass:/root/.pgpass \
    --volume ~/.trains.conf:/root/trains.conf \
		--volume ~/workdir/competitions/$(REPO):/workdir \
		--volume ~/workdir/training_config:/workdir/$(REPO)/training_config \
    --volume /mnt/hdd/data:/workdir/data \
    --volume /mnt/hdd/models:/workdir/models \
		--detach \
		--shm-size "32G" \
		--publish 8887:8888 \
		$(REPO):$(VERSION) /workdir/scripts/run_jupyter.sh

