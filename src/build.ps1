

$CFLAGS = @("-std=c++20", "-O3", "-fopenmp")
$LIBS = @("-lws2_32")

g++ main.cpp $CFLAGS -o main.exe $LIBS
