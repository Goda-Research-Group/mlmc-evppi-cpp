PROGRAM=mlmc
SRCS=mlmc.c
OBJS=$(SRCS:.c=.o)
CC=g++
CFLAGS=-Wall

$(PROGRAM):$(OBJS)
	$(CC) $(CFLAGS) -o $(PROGRAM) $(OBJS)

clean:
	rm -rf $(PROGRAM) *.o