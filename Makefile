.PHONY: release build test test_gpt2 copy_course_file

current_version_number := $(shell git tag --list "v*" | sort -V | tail -n 1 | cut -c 2-)
next_version_number := $(shell echo $$(($(current_version_number)+1)))

release:
	git tag v$(next_version_number)
	git push origin main v$(next_version_number)

build:
	go build -o dist/main.out ./cmd/tester

test:
	go test -v ./internal/...

test_gpt2:
	scripts/test_gpt2_stage.sh

copy_course_file:
	cp ../build-your-own-inference-engine/course-definition.yml internal/test_helpers/course_definition.yml
