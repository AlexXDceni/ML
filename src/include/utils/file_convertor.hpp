#pragma once

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include <ctime>
#include <cstdint>
#include <sys/stat.h>

    // FILE FROM OTHER PROJECT ON GITHUB: https://github.com/AlexXDceni/File_convertor_to_PPM

// #define STB_IMAGE_IMPLEMENTATION
#include "../external/stb_image.h"

#ifdef _WIN32
    #include <direct.h>
    #define mkdir _mkdir
#else
    #include <unistd.h>
#endif


using namespace std;


struct Pixel {
    int r, g, b;
};

struct Arguments {
    string inputFile;
    string outputFile;
    int width = 28;
    int height = 28;
    string type = "ppm";
};

// Input file type
string getFileExtension(const string& filename) {
    size_t pos = filename.find_last_of('.');
    if (pos != string::npos) {
        return filename.substr(pos + 1);
    }
    return "";
}

// Input file name without path
string getBaseName(const string& path) {
    size_t pos = path.find_last_of("/\\");
    if (pos != string::npos) {
        return path.substr(pos + 1);
    }
    return path;
}

string getCurrentTimestamp() {
    time_t now = time(nullptr);
    tm* localTime = localtime(&now);
    char buffer[32];
    strftime(buffer, sizeof(buffer), "%Y%m%d_%H%M%S", localTime);
    return string(buffer);
}


void createDirectory(const string& path) {
    mkdir(path.c_str());
}


// Creating output file name and folder if needed
void ensureOutputPath(Arguments& args) {
    const string outputDir = "images";
    createDirectory(outputDir);

    if (args.outputFile.empty()) {
        args.outputFile = outputDir + "/output_" + getCurrentTimestamp() + "." + args.type;
        cout << "No output specified. Using: " << args.outputFile << "\n";
        return;
    }

    string extension = getFileExtension(args.outputFile);
    string baseName = getBaseName(args.outputFile);

    if (extension.empty()) {
        args.outputFile += "." + args.type;
    }


    // Check if output file has a path , if not, put in folder
    if (args.outputFile.find('/') == string::npos && 
        args.outputFile.find('\\') == string::npos) {
        args.outputFile = outputDir + "/" + args.outputFile;
    }

    size_t lastSlash = args.outputFile.find_last_of("/\\");
    if (lastSlash != string::npos) {
        string dir = args.outputFile.substr(0, lastSlash);
        createDirectory(dir);
    }
}

void printHelp() {
    cout << "Image to PPM/PGM/PBM Converter\n";
    cout << "Usage:\n";
    cout << "  main.exe -i <input> [options]\n";
    cout << "  main.exe -i <input> -o <output> [options]\n\n";
    cout << "Options:\n";
    cout << "  --help, -help, /?, help   Show this help message\n";
    cout << "  -i, --input <file>    Input image file (required)\n";
    cout << "  -o, --output <file>   Output file (optional, default: images/timestamp.type)\n";
    cout << "  -w, --width <px>      Target width in pixels (default: 28)\n";
    cout << "  -h, --height <px>     Target height in pixels (default: 28)\n";
    cout << "  -t, --type <type>     Output type: ppm (color), pgm (grayscale) (default: ppm)\n";
    cout << "\nIf only width or height is specified, the other will be set to match (square).\n";
    cout << "Quality is calculated automatically based on output/input dimensions.\n";
    cout << "Output files are saved to 'images/' folder.\n";
}

bool parseArguments(int argc, char* argv[], Arguments& args) {
    for (int i = 1; i < argc; i++) {
        string arg = argv[i];

        if (arg == "-i" || arg == "--input") {
            if (i + 1 < argc) args.inputFile = argv[++i];
            else { cerr << "Error: -i requires a filename\n"; return false; }
        }
        else if (arg == "-o" || arg == "--output") {
            if (i + 1 < argc) args.outputFile = argv[++i];
            else { cerr << "Error: -o requires a filename\n"; return false; }
        }
        else if (arg == "-w" || arg == "--width") {
            if (i + 1 < argc) args.width = stoi(argv[++i]);
            else { cerr << "Error: -w requires a number\n"; return false; }
        }
        else if (arg == "-h" || arg == "--height") {
            if (i + 1 < argc) args.height = stoi(argv[++i]);
            else { cerr << "Error: -h requires a number\n"; return false; }
        }
        else if (arg == "-t" || arg == "--type") {
            if (i + 1 < argc) args.type = argv[++i];
            else { cerr << "Error: -t requires a type\n"; return false; }
        }
        else if (arg == "--help" || arg == "-help" || arg == "/?" || arg == "help") {
            printHelp();
            exit(0);
        }
        else {
            cerr << "Unknown argument: " << arg << "\n";
            printHelp();
            return false;
        }
    }

    if (args.inputFile.empty()) {
        cerr << "Error: Input file is required (-i)\n";
        return false;
    }

    if (args.type != "ppm" && args.type != "pgm" ) {
        cerr << "Error: Type must be 'ppm' or 'pgm'\n";
        return false;
    }

    if (args.width <= 0 || args.height <= 0) {
        cerr << "Error: Width and height must be positive\n";
        return false;
    }

    return true;
}

Pixel getPixelAt(const unsigned char* imageData, int x, int y, int width, int channels) {
    Pixel p = {0, 0, 0};
    if (x < 0) x = 0;
    if (y < 0) y = 0;
    if (x >= width) x = width - 1;
    if (y >= width) y = width - 1;

    int index = (y * width + x) * channels;
    if (channels >= 3) {
        p.r = imageData[index];
        p.g = imageData[index + 1];
        p.b = imageData[index + 2];
    } else if (channels == 1) {
        p.r = p.g = p.b = imageData[index];
    }
    return p;
}

Pixel getBlockAverage(const unsigned char* imageData, int startX, int startY, int blockWidth, int blockHeight, int imgWidth, int imgHeight, int channels)               
{
    int64_t totalR = 0, totalG = 0, totalB = 0;
    int count = 0;

    for (int y = 0; y < blockHeight; y++) {
        for (int x = 0; x < blockWidth; x++) {
            Pixel p = getPixelAt(imageData, startX + x, startY + y, imgWidth, channels);
            totalR += p.r;
            totalG += p.g;
            totalB += p.b;
            count++;
        }
    }

    Pixel result;
    if (count > 0) {
        result.r = static_cast<int>(totalR / count);
        result.g = static_cast<int>(totalG / count);
        result.b = static_cast<int>(totalB / count);
    } else {
        result = {0, 0, 0};
    }
    return result;
}

int getGrayscale(const Pixel& p) {
    return static_cast<int>(0.299 * p.r + 0.587 * p.g + 0.114 * p.b);
}

bool writePPM(const vector<vector<Pixel>>& pixels, int width, int height, const string& filename) {
    ofstream file(filename, ios::binary);
    if (!file.is_open()) {
        cerr << "Error: Cannot create output file\n";
        return false;
    }

    file << "P6\n";
    file << width << " " << height << "\n";
    file << "255\n";

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            const Pixel& p = pixels[y][x];
            file << static_cast<unsigned char>(min(255, max(0, p.r)));
            file << static_cast<unsigned char>(min(255, max(0, p.g)));
            file << static_cast<unsigned char>(min(255, max(0, p.b)));
        }
    }

    file.close();
    return true;
}

bool writePGM(const vector<vector<Pixel>>& pixels, int width, int height, const string& filename) {
    ofstream file(filename, ios::binary);
    if (!file.is_open()) {
        cerr << "Error: Cannot create output file\n";
        return false;
    }

    file << "P5\n";
    file << width << " " << height << "\n";
    file << "255\n";

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int gray = getGrayscale(pixels[y][x]);
            file << static_cast<unsigned char>(gray);
        }
    }

    file.close();
    return true;
}

bool convertImage(const Arguments& args) {
    int imgWidth, imgHeight, channels;
    unsigned char* imageData = stbi_load(args.inputFile.c_str(), &imgWidth, &imgHeight, &channels, 0);

    if (!imageData) {
        cerr << "Error: Cannot load image: " << args.inputFile << "\n";
        cerr << "Make sure the file exists and is a valid image format.\n";
        return false;
    }

    cout << "Input image: " << imgWidth << "x" << imgHeight << " (" << channels << " channels)\n";
    cout << "Output size: " << args.width << "x" << args.height << "\n";

    int qualityX = imgWidth / args.width;
    int qualityY = imgHeight / args.height;
    int blockWidth = max(1, qualityX);
    int blockHeight = max(1, qualityY);

    cout << "Quality: block size " << blockWidth << "x" << blockHeight << "\n";

    vector<vector<Pixel>> outputPixels(args.height, vector<Pixel>(args.width));

    for (int y = 0; y < args.height; y++) {
        for (int x = 0; x < args.width; x++) {
            int startX = x * blockWidth;
            int startY = y * blockHeight;
            outputPixels[y][x] = getBlockAverage(imageData, startX, startY, blockWidth, blockHeight, imgWidth, imgHeight, channels);
        }
    }

    stbi_image_free(imageData);

    cout << "Writing " << args.type << " file...\n";

    bool success = false;
    if (args.type == "ppm") {
        success = writePPM(outputPixels, args.width, args.height, args.outputFile);
    } else if (args.type == "pgm") {
        success = writePGM(outputPixels, args.width, args.height, args.outputFile);
    } 

    if (success) {
        cout << "Success! File saved to: " << args.outputFile << "\n";
    }

    return success;
}
