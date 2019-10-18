PROGRAM = test
SRCS = test.cpp ../evppi.cpp
OBJS = $(SRCS:.cpp = .o)
CC = g++
CFLAGS = -Wall

$(PROGRAM) : $(OBJS)
	$(CC) $(CFLAGS) -o $(PROGRAM) $(OBJS)

clean:
	rm -rf $(PROGRAM) *.o