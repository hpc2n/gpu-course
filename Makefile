#CPU_LIBS=-lblas
CPU_LIBS=$(LIBBLAS_MT)
CUDA_LIBS=-lcublas

all: hello.cuda ax.cuda axpy.cpu axpy.cuda gemm.cpu gemm.cuda

%.cpu: %.c
	gcc -o $@ $< $(CFLAGS) $(CPU_LIBS)

%.cuda: %.cu
	nvcc -o $@ $< $(CUDA_LIBS)

.PHONY: clean
clean:
	rm -f *.cuda *.cpu
