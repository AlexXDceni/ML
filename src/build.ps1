param (
    [switch]$raylib
)

$CFLAGS = @("-std=c++20", "-O3", "-fopenmp")

if ($Raylib) {
    $RAYLIB_PATH = "D:/Sibrary/raylib/raylib/src"
    $LIBS = @("-I$RAYLIB_PATH", "-L$RAYLIB_PATH", "-lraylib", "-lopengl32", "-lgdi32", "-lwinmm")
} 
else {
    $LIBS = @("-lws2_32")
}

g++ main.cpp $CFLAGS -o main.exe $LIBS
