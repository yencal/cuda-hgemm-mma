NVCC = nvcc
ARCH = sm_80
CFLAGS = -O3 -arch=$(ARCH) --std=c++17

profile: src/profile.cu src/*.cuh
	$(NVCC) $(CFLAGS) -lcurand -lcublas -o $@ src/profile.cu

clean:
	rm -f profile *.csv

.PHONY: clean
