#include <iostream>
#include <vector>
#include <string>
#include "neural_network.hpp"
using namespace std;

int main()
{

    string SAVE_FILE = "tests/images.txt";
 
    int W = 28;  // for images
    int H = 28;

    int input = W*H;
    int output = 10;

    int activation_func = 1;

    // 0 - SIGMOID  <- By default if no param passed through
    // 1 - RELU
    // 2 - LEAKY_RELU
    // 3 - LINEAR
    // 4 - TANH
    // 5 - ELU

    Network nn = Network( { input , 16 , 32 , 16 , output } , activation_func ); // W*H     , 16 , 32 , 16 ,      10  // for images
    // Network::TrainingData data = nn.loadData(input, output); // comment for images
    
    Network::TrainingData data; // for images
    nn.addPhotoToTraining("images/img_3.ppm", 9, data);
    nn.addPhotoToTraining("images/img_1.ppm", 2, data);
    nn.addPhotoToTraining("images/img_2.ppm", 0, data);

    bool task = 1;

    switch (task)
    {

    case 0: // Learn

        nn.evolve(4000, 20, 0.05, data);
        for (int i = 0; i < data.inputs.size(); ++i)
        {
            nn.feedForward(data.inputs[i]);
            nn.printResults();
        }
        nn.saveModal(SAVE_FILE);
        break;

    case 1: // Predict

        nn.loadModal(SAVE_FILE);
        // nn.displayFullNetwork(data.targets[0]);

        for (int i = 0; i < data.inputs.size(); i++)
        {
            double prediction = nn.predictBiggest(data.inputs[i]);  //  <- for ppm images
            //vector<double> prediction = nn.predict(data.inputs[i]);   // for normal data

            cout << "Input " << i << ": ";

            // for (double val : data.inputs[i])  // comment for images
            //     cout << val << " ";

            cout << "-> AI says: " << prediction << endl; 
                // prediction[0]   <- for normal
                //  (prediction[0] == 1 ? "1" : "0")  <- true/false or first/second
                // prediction // 
        }
        break;
    }
    return 0;
}