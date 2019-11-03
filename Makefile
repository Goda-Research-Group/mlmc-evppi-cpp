PROGRAM = test
SRCS = test.cpp matrix.cpp
OBJS = $(SRCS:.cpp = .o)
CC = g++
CFLAGS = -Wall -Wextra -std=c++11

$(PROGRAM) : $(OBJS)
	$(CC) $(CFLAGS) -o $(PROGRAM) $(OBJS)

clean:
	rm -rf $(PROGRAM) *.o