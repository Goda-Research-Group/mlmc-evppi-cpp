PROGRAM = evppi
SRCS = bkoc.cpp ../evppi.cpp
OBJS = $(SRCS:.cpp = .o)
CC = g++
CFLAGS = -Wall -Wextra

$(PROGRAM) : $(OBJS)
	$(CC) $(CFLAGS) -o $(PROGRAM) $(OBJS)

clean:
	rm -rf $(PROGRAM) *.o