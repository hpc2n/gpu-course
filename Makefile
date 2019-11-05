CPU_LIBS=$(LIBBLAS_MT)
CUDA_LIBS=-lcublas

all: hello.cuda ax.cuda axpy.cpu axpy.cuda gemm.cpu gemm.cuda

%.cpu: %.c
	$(CC) -o $@ $< $(CFLAGS) $(CPU_LIBS)

%.cuda: %.cu
	$(CUDA_CC) -o $@ $< $(CUDA_LIBS)

.PHONY: clean
clean:
	rm -f *.cuda *.cpu