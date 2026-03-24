#include <iostream>
#include <vector>
#include <string>
#include "neural_network.hpp"
using namespace std;

int main()
{

    string SAVE_FILE = "tests/tests.txt";

    int W = 28;
    int H = 28;

    int input = 1;
    int output = 1;

    Network nn = Network({input, 8, output}); // W*H, 16, 32, 16, 10
    Network::TrainingData data = nn.loadData(input, output);

    // TrainingData data;
    // nn.addPhotoToTraining("images/img_1.ppm", 2, data);
    // nn.addPhotoToTraining("images/img_2.ppm", 0, data);
    // nn.addPhotoToTraining("images/img_3.ppm", 9, data);

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
            vector<double> prediction = nn.predict(data.inputs[i]); // double prediction = nn.predictBiggest(data.inputs[i]);   //vector<double> prediction = nn.predict(data.inputs[i]);
            cout << "Input " << i << ": ";
            for (double val : data.inputs[i])
                cout << val << " ";
            cout << "-> AI says: " << prediction[0] << endl; // prediction[0]  //  (prediction[0] == 1 ? "1" : "0")
        }
        break;
    }
    return 0;
}