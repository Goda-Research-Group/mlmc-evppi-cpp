PROGRAM=mlmc
SRCS=mlmc.c
OBJS=$(SRCS:.c=.o)
CC=g++
CFLAGS=-Wall

$(PROGRAM):$(OBJS)
	$(CC) $(CFLAGS) -o $(PROGRAM) $(OBJS) -lstdc++ -lpython2.7 -std=c++11

clean:
	rm -rf $(PROGRAM) *.o