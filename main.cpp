#include <iostream>
#include <string>
#include <iomanip>
#include "neural_network.hpp"
using namespace std;

int main( int argc , char* argv[] )
{

    string SAVE_FILE = "tests/tests.txt";

    int W = 28;  // for images
    int H = 28;

    int input = 3;
    int output = 1;

    int activation_func = 0;

    // 0 - SIGMOID  <- By default if no param passsed through
    // 1 - RELU
    // 2 - LEAKY_RELU
    // 3 - LINEAR
    // 4 - TANH
    // 5 - ELU

    // Network nn = Network( { input , 8 , 7 , 5 , output } , activation_func );    // for normal data
    // Network::TrainingData data = nn.loadData(input, output);                     // comment for images


    Network nn = Network( { W*H , 128 , 10 } , activation_func );                   // for images

    Network::TrainingData data;                                                     // for images

    // nn.addPhotoToTraining("images/img_3.ppm", 9, data);
    // nn.addPhotoToTraining("images/img_1.ppm", 2, data);
    // nn.addPhotoToTraining("images/img_2.ppm", 0, data);

        cout << "Loading MNIST dataset..." << endl;
        nn.loadMnist("mnist/train-images.idx3-ubyte", "mnist/train-labels.idx1-ubyte", data);
        
        int max_samples = 1000; 
        if (data.inputs.size() > max_samples) {
            data.inputs.resize(max_samples);
            data.targets.resize(max_samples);
        }
        cout<< "MNIST dataset loaded. Training samples: " << data.inputs.size() << endl;

    int task = 2; // 0 - Learn, 1 - Predict, 2 - Predict mnist
    bool training_mode = 0; // 0 - Backpropagation, 1 - Evolve

    int epochs = 500;
    double learning_rate = 0.05;
    int population_size = 20;
    int tests_number = 1000;

    switch (task)   // make separate functions and task cases for every type  , and way to use args for cli use
    {

    case 0: // Learn

        nn.loadModal(SAVE_FILE);

        if(training_mode == 0){
            nn.backpropagate(learning_rate, epochs, data); 
        }
        else {
            nn.evolve(epochs, population_size, learning_rate, data); 
        }

        nn.saveModal(SAVE_FILE);
        break;

    case 1: // Predict

        nn.loadModal(SAVE_FILE);

        for (int i = 0; i < data.inputs.size(); i++)
        {
            double prediction = nn.predictBiggest(data.inputs[i]);  //  <- for ppm images

            // vector<double> prediction = nn.predict(data.inputs[i]);   // for normal data

            cout << "Input " << i << ": ";

            // for (double val : data.inputs[i])  // comment for images
            //     cout << val << " ";

            cout << "-> AI says: " << prediction << endl;

                // prediction[0]    //  <- for normal
                // (prediction[0] == 1 ? "1" : "0")  //  <- true/false or first/second
                // prediction       //  <- for images
        }
        break;



    case 2: // Predict mnist   // made by ai, read , understand, and rewrite

        nn.loadModal(SAVE_FILE);
        cout << "Testing on " << data.inputs.size() << " samples..." << endl;

        int correctPredictions = 0;
        vector<int> testedIndices;

        // Generate random indices                  // Verify so it doesn't repeat and test on unique samples, and also verify that it doesn't go out of bounds if tests_number > data.inputs.size()
        srand(time(0));
        for (int i = 0; i < tests_number && i < data.inputs.size(); i++) {
            int randomIdx = rand() % data.inputs.size();
            testedIndices.push_back(randomIdx);
        }

        for (int i = 0; i < testedIndices.size(); i++)
        {
            int idx = testedIndices[i];
            int prediction = (int)nn.predictBiggest(data.inputs[idx]);

            int actualLabel = -1;
            for (int j = 0; j < data.targets[idx].size(); j++) {
            if (data.targets[idx][j] > 0.9) { 
            actualLabel = j;
            break;
            }
            }

            bool isCorrect = (prediction == actualLabel);
            if (isCorrect) correctPredictions++;

            cout << setw(4) << (i + 1) 
            << ". Input " << setw(4) << idx 
            << ": Real: " << setw(2) << actualLabel 
            << " | AI: " << setw(2) << prediction 
            << (isCorrect ? " [CORRECT]" : " [WRONG]") << endl;

            if (!isCorrect) {
                nn.displayMnistImage(data.inputs[idx]);
            }
        }

        double accuracy = (double)correctPredictions / tests_number * 100.0;
        cout << "\n-----------------------------" << endl;
        cout << "Total correct: " << correctPredictions << "/" << tests_number << endl;
        cout << "Accuracy: " << accuracy << "%" << endl;
        cout << "-----------------------------" << endl;
        
        break;
    }
    return 0;
}