#pragma once

#include <iostream>
#include <fstream>

#include "../neural_network.hpp"

#include "file_convertor.hpp"

namespace MNIST {

        void addPhotoToTraining(NeuralNetwork& nn, const string &filename, double target){ 

            ifstream file(filename, ios::binary);
            if (!file.is_open())
            {
                cerr << "Error: " << filename << " not found!" << endl;
                return;
            }
            string magic;
            file >> magic;
            file >> ws;
            while (file.peek() == '#')
            {
                string dummy;
                getline(file, dummy);
                file >> ws;
            }

            int w, h, maxVal;
            file >> w >> h >> maxVal;
            file.ignore();

            vector<double> pixels;

            if (magic == "P6")
            {
                for (int i = 0; i < w * h; ++i)
                {
                    unsigned char r, g, b;
                    file.read((char *)&r, 1);
                    file.read((char *)&g, 1);
                    file.read((char *)&b, 1);
                    double gray = (0.299 * r + 0.587 * g + 0.114 * b) / 255.0;
                    pixels.push_back(gray);
                }
            }
            else if (magic == "P5")
            {
                for (int i = 0; i < w * h; ++i)
                {
                    unsigned char pixel;
                    file.read((char *)&pixel, 1);
                    pixels.push_back((double)pixel / 255.0);
                }
            }
            else if (magic == "P2")
            {
                for (int i = 0; i < w * h; ++i)
                {
                    int val;
                    file >> val;
                    pixels.push_back((double)val / 255.0);
                }
            }
            else
            {
                cerr << "Format error: " << filename << " is " << magic << ". Not supported!" << endl;
                return;
            }

            if (pixels.size() == 784)
            {
                vector<double> targetVector(10, 0.0);
                if (target >= 0 && target < 10)
                    targetVector[(int)target] = 1.0;
                nn.data.inputs.push_back(pixels);
                nn.data.targets.push_back(targetVector);
                cout << "Loaded " << filename << " as Grayscale (" << magic << ")" << endl;
            }
            else
            {
                cerr << "Error: " << filename << " size mismatch. Got " << pixels.size() << " values, expected 784." << endl;
            }
        }

        int reverseInt(int i) {
            unsigned char c1, c2, c3, c4;
            c1 = i & 255;
            c2 = (i >> 8) & 255;
            c3 = (i >> 16) & 255; 
            c4 = (i >> 24) & 255;
            return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
        }

        void loadMnist(NeuralNetwork& nn, const string &image_path, const string &label_path, int max_samples = -1) {

            
            cout << "Loading MNIST dataset..." << endl;

            ifstream img_file(image_path, ios::binary);
            ifstream lbl_file(label_path, ios::binary);

            if (!img_file.is_open() || !lbl_file.is_open()) {
                cerr << "Error: Check your paths! Could not open MNIST raw binary files." << endl;
                return;
            }

            int magic, num_items, rows, cols, magic_lbl, num_lbls;
            
            img_file.read((char*)&magic, 4);
            img_file.read((char*)&num_items, 4);
            img_file.read((char*)&rows, 4);
            img_file.read((char*)&cols, 4);
            
            lbl_file.read((char*)&magic_lbl, 4);
            lbl_file.read((char*)&num_lbls, 4);

            num_items = reverseInt(num_items);
            rows = reverseInt(rows);
            cols = reverseInt(cols);
            num_lbls = reverseInt(num_lbls);

            if( max_samples != -1 && num_items > max_samples ) {
                num_items = max_samples;
            }

            for (int i = 0; i < num_items; ++i) {
                vector<double> pixels;
                for (int p = 0; p < rows * cols; ++p) {
                    unsigned char temp = 0;
                    img_file.read((char*)&temp, 1);
                    pixels.push_back((double)temp / 255.0);
                }
                nn.data.inputs.push_back(pixels);

                unsigned char label = 0;
                lbl_file.read((char*)&label, 1);
                vector<double> target(10, 0.0);
                target[(int)label] = 1.0;
                nn.data.targets.push_back(target);
            }

            cout<< "MNIST dataset loaded." << endl; 
        }

        void displayMnistImage(const vector<double> &pixels, int width = 28, int height = 28) {
            cout << "\nMNIST Image (ID-based):" << endl;
            if (pixels.size() != width * height) {
                cerr << "Error: Invalid pixel count for " << width << "x" << height << " image" << endl;
                return;
            }
            
            for (int y = 0; y < height; ++y) {
                for (int x = 0; x < width; ++x) {
                    double pixel = pixels[y * width + x];
                    if (pixel > 0.5) {
                        cout << "##";
                    } else {
                        cout << "  ";
                    }
                }
                cout << endl;
            }
        }

        void displayMnistImageById(NeuralNetwork& nn, int imageId) {
            if (imageId < 0 || imageId >= nn.data.inputs.size()) {
                cerr << "Error: Image ID " << imageId << " out of range. Available: 0-" << (nn.data.inputs.size() - 1) << endl;
                return;
            }
            cout << "Image ID: " << imageId << " | Label: ";
            
            for (int i = 0; i < nn.data.targets[imageId].size(); ++i) {
                if (nn.data.targets[imageId][i] > 0.5) {
                    cout << i << endl;
                    break;
                }
            }
            
            displayMnistImage(nn.data.inputs[imageId]);
        }
        
        void MnistTest(NeuralNetwork& nn, const string &input_file, const string &target_file, int max_samples, int tests_number) {
            
            
            loadMnist(nn, input_file, target_file, max_samples);
            cout<< "Training samples: " << (nn.data.inputs.size() < max_samples ? nn.data.inputs.size() : max_samples) << endl;
    
    
            cout << "Testing on " << tests_number << " samples..." << endl;
    
            int correctPredictions = 0;
            vector<int> testedIndices;
    
            // Generate random indices                  // Verify so it doesn't repeat and test on unique samples, and also verify that it doesn't go out of bounds if tests_number > data.inputs.size()
            srand(time(0));

            for (int i = 0; i < tests_number && i < nn.data.inputs.size(); i++) {

                int randomIdx = rand() % nn.data.inputs.size();

                testedIndices.push_back(randomIdx);
            }
    
            for (int i = 0; i < testedIndices.size(); i++)
            {
                int idx = testedIndices[i];
                vector<double> prediction = nn.predictBiggest(nn.data.inputs[idx]);
    
                int actualLabel = -1;
                for (int j = 0; j < nn.data.targets[idx].size(); j++) {
                if (nn.data.targets[idx][j] > 0.9) { 
                actualLabel = j;
                break;
                }
                }
    
                bool isCorrect = (prediction[0] == actualLabel);
                if (isCorrect) correctPredictions++;
    
                cout << setw(4) << (i + 1) 
                << ". Input " << setw(4) << idx 
                << ": Real: " << setw(2) << actualLabel 
                << " | AI: " << setw(2) << prediction[0] 
                << (isCorrect ? " [CORRECT]" : " [WRONG]") << endl;
    
                if (!isCorrect) {
                    displayMnistImage(nn.data.inputs[idx]);
                }
            }
    
            double accuracy = (double)correctPredictions / tests_number * 100.0;
            cout << "\n-----------------------------" << endl;
            cout << "Total correct: " << correctPredictions << "/" << tests_number << endl;
            cout << "Accuracy: " << accuracy << "%" << endl;
            cout << "-----------------------------" << endl;
        }

}