#include <iostream>
#include <string>
#include <iomanip>
#include "include/neural_network.hpp"
#include <fstream>
// #include <nlohmann/json.json>

using namespace std;

int main( int argc , char* argv[] )
{
    
                        // TODOS

        // 1. Refactor the code to make it my own 
        // 2. Add checkpoint system to save intermediate models during training, and also add option to load specific checkpoint for continued training or evaluation.
        // 3. Add encoder, decoder, tokenizer     
        // 4. Implement a more efficient training algorithm, such as mini-batch gradient descent or Adam optimizer, to speed up the learning process and improve convergence.


    const string model_name = "mnist_model";
    const string SAVE_FILE = "../models/" + model_name + ".json";
    const string INPUT_FILE = "../data/io/inputs.in";
    const string TARGET_FILE = "../data/io/targets.in";
    const string MINST_INPUTS_FILE = "../data/mnist/train-images.idx3-ubyte";
    const string MINST_LABELS_FILE = "../data/mnist/train-labels.idx1-ubyte";

    // For MNIST dataset
    const int W = 28; 
    const int H = 28;

    // NN architecture
    const int INPUT = 3;
    const int OUTPUT = 1;

    // Training parameters
    const int epochs = 500;
    const double learning_rate = 0.05;
    const int task = 2;             
    // 0 - Learn
    // 1 - Predict
    // 2 - Test mnist
    const string activation_func = "sigmoid";
    // sigmoid  <- By default if no param passsed through
    // relu
    // leaky_relu 
    // linear 
    // tanh 
    // elu 



    switch (task)   
    {

    case 0: // Learn
        {   
            Network nn = Network( nn.get_sizes(SAVE_FILE) );    
            Network::TrainingData data;

            nn.load_modal(SAVE_FILE);
    
    
            data = nn.loadData(INPUT, OUTPUT, INPUT_FILE, TARGET_FILE);
    
      
            nn.backpropagate(data, learning_rate, epochs); 
    
            nn.save_modal(SAVE_FILE);
            break;
        }

    case 1: // Predict
        {
            Network nn = Network( nn.get_sizes(SAVE_FILE) );    
            Network::TrainingData data;
            
            nn.load_modal(SAVE_FILE);
            
            data = nn.loadData(INPUT, OUTPUT, INPUT_FILE, TARGET_FILE);
    
            for (int i = 0; i < data.inputs.size(); i++)
            {
                vector<double> prediction = nn.predictBiggest(data.inputs[i]);  //  <- for ppm images
    
                // vector<double> prediction = nn.predict(data.inputs[i]);   // for normal data
    
                cout << "Input " << i << ": ";
    
                // for (double val : data.inputs[i])  // comment for images
                //     cout << val << " ";
    
                cout << "-> AI says: " << prediction[0] << endl;
    
                    // prediction[0]    //  <- for normal
                    // (prediction[0] == 1 ? "1" : "0")  //  <- true/false or first/second
            }
        break;
        }


    case 2: 
        {
            Network nn = Network( nn.get_sizes(SAVE_FILE) );    

            cout << "Loading model from " << SAVE_FILE << "..." <<endl;
            nn.load_modal(SAVE_FILE);
            cout << "Model loaded successfully!" << endl;

            const int max_samples = 1000; 
            const int tests_number = 1000;

            nn.MnistTest(MINST_INPUTS_FILE, MINST_LABELS_FILE, max_samples, tests_number);

            break;
        }
    }   
    return 0;
}