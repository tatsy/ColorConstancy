CC			= g++
BINDIR		= ./bin/
SRCS		= ./clcnst/clcnst.cpp
SOBJ		= $(BINDIR)libclcnst.so
OBJ			= main.o
INC			= -I/usr/local/include -I./clcnst
CFLAGS		= -fPIC
LDFLAGS		= -L/usr/local/lib/opencv -L./bin
LIBS		= -lopencv_core -lopencv_highgui -lopencv_imgproc
LIB_CLCNST	= -lclcnst
HORN		= $(BINDIR)HornAlrogithm
BLAKE		= $(BINDIR)BlakeAlgorithm
MOORE		= $(BINDIR)MooreAlgorithm
RAHMAN		= $(BINDIR)RahmanAlgorithm
HOMO		= $(BINDIR)HomomorphicFilter
FAUG		= $(BINDIR)FaugerasAlgorithm
PROGRAMS	= $(SOBJ) $(HORN) $(BLAKE) $(MOORE) $(RAHMAN) $(HOMO) $(FAUG)

all: $(PROGRAMS)

# build .so
$(SOBJ): $(SRCS)
	mkdir -p $(BINDIR)
	$(CC) -shared $(CFLAGS) -o $(SOBJ) $(SRCS) $(INC) $(LDFLAGS) $(LIBS)

# Horn's algorithm
SRC_HORN			= ./HornAlgorithm/main.cpp
$(HORN): $(SRC_HORN)
	$(CC) -o $(HORN) $(SRC_HORN) $(INC) $(LDFLAGS) $(LIBS) $(LIB_CLCNST)

# Blake's algorithm
SRC_BLAKE			= ./BlakeAlgorithm/main.cpp
$(BLAKE): $(SRC_BLAKE)
	$(CC) -o $(BLAKE) $(SRC_BLAKE) $(INC) $(LDFLAGS) $(LIBS) $(LIB_CLCNST)

# Moore's algorithm
SRC_MOORE			= ./MooreAlgorithm/main.cpp
$(MOORE): $(SRC_MOORE)
	$(CC) -o $(MOORE) $(SRC_MOORE) $(INC) $(LDFLAGS) $(LIBS) $(LIB_CLCNST)

# Rahman's algorithm
SRC_RAHMAN			= ./RahmanAlgorithm/main.cpp
$(RAHMAN): $(SRC_RAHMAN)
	$(CC) -o $(RAHMAN) $(SRC_RAHMAN) $(INC) $(LDFLAGS) $(LIBS) $(LIB_CLCNST)

# Homomorphic Filter
SRC_HOMO			= ./HomomorphicFilter/main.cpp
$(HOMO): $(SRC_HOMO)
	$(CC) -o $(HOMO) $(SRC_HOMO) $(INC) $(LDFLAGS) $(LIBS) $(LIB_CLCNST)

# Faugeras Filter
SRC_FAUG			= ./FaugerasAlgorithm/main.cpp
$(FAUG): $(SRC_FAUG)
	$(CC) -o $(FAUG) $(SRC_FAUG) $(INC) $(LDFLAGS) $(LIBS) $(LIB_CLCNST)

clean:
	rm -rf ./bin