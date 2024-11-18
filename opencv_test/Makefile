INCLUDE_DIRS = -I/usr/include/opencv4
LIB_DIRS = 
TARGET ?= capture
CROSS_COMPILE = #aarch64-none-linux-gnu-
CXX ?= $(CROSS_COMPILE)g++

CDEFS=
CXXFLAGS ?= -O3 -g -Wall -Werror $(INCLUDE_DIRS) $(CDEFS)
# LIBS= -lpthread
LDFLAGS ?= -lpthread -lrt -L/usr/lib -lopencv_core -lopencv_flann -lopencv_video

HFILES= $(wildcard *.hpp)
CFILES= $(wildcard *.cpp)

#SRCS= ${HFILES} #${CFILES}
OBJS= ${CFILES:.c=.o}

.PHONY: default all clean

default: $(TARGET)
all:	default

clean:
	-rm -f *.o *.d
	-rm -f $(TARGET)
 
%.o: %.c $(HFILES)
	$(CC) $(CFLAGS) -c $< -o $@


$(TARGET): $(OBJS)
	$(CXX) $(OBJS) $(CXXFLAGS) $(LDFLAGS) -o $@ `pkg-config --libs opencv4`
#	$(CC) $(OBJS) $(CFLAGS) $(LIBS) -o $@