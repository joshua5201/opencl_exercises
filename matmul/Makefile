all: matmul_cl
CC = gcc
CFLAGS = -g -Wall -lOpenCL -std=c99

matmul: matmul_cl.c 
	$(CC) $(CFLAGS) matmul_cl.c

clean: 
	rm matmul_cl *.out 

diff:
	diff global.cl.out local.cl.out
