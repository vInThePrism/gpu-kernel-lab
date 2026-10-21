# Default target is the free Colab / Kaggle T4 (Turing, sm_75).
# Override for other GPUs:  make ARCH=sm_80   (A100)   make ARCH=sm_89 (L4)
ARCH    ?= sm_75
NVCC    ?= nvcc
FLAGS   := -O2 -std=c++17 -arch=$(ARCH) -I. -lineinfo

SRCS := $(wildcard [0-9][0-9]_*.cu)
BINS := $(SRCS:.cu=)

all: $(BINS)

%: %.cu common/check.cuh common/primitives.cuh
	$(NVCC) $(FLAGS) $< -o $@

# Builds and runs every test; stops at the first FAIL.
test: all
	@for b in $(BINS); do echo "== $$b"; ./$$b || exit 1; done

# Race / barrier checking. Slow; run on the small cases.
sanitize: all
	@for b in $(filter-out 11_async_pipeline,$(BINS)); do \
	  compute-sanitizer --tool racecheck ./$$b >/dev/null && \
	  compute-sanitizer --tool synccheck ./$$b >/dev/null && echo "$$b clean" || exit 1; done

clean:
	rm -f $(BINS)

.PHONY: all test sanitize clean
