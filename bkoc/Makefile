PROGRAM = evppi
SRCS = bkoc1.cpp ../evppi.cpp ../matrix.cpp ../util.cpp
OBJS = $(SRCS:.cpp = .o)
CC = g++
CFLAGS = -Wall -Wextra -std=c++11 -o3

$(PROGRAM) : $(OBJS)
	$(CC) $(CFLAGS) -o $(PROGRAM) $(OBJS)

clean:
	rm -rf $(PROGRAM) *.o