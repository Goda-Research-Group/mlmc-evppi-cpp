PROGRAM=eig
SRCS=eig.cpp
OBJS=$(SRCS:.cpp=.o)
CXX=g++
CXXFLAGS=-Wall -O3 -std=c++11 -pthread

$(PROGRAM):$(OBJS)
	$(CXX) $(CXXFLAGS) -o $(PROGRAM) $(OBJS)

clean:
	rm -rf $(PROGRAM) *.o