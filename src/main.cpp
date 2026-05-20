#include <iostream>
#include <string>
#include "neural_network.hpp"
// #include <nlohmann/json.json>

using namespace std;

int main( int argc , char* argv[] )
{

    // TODOS

    // 1. Refactor the code to make it my own 
    // 2. add model comparation function

    // 3. Implement a more efficient training algorithm, such as mini-batch gradient descent or Adam optimizer, to speed up the learning process and improve convergence.
    // 4. Add support for more complex neural network architectures, such as convolutional neural networks (CNNs) for image processing tasks or recurrent neural networks (RNNs) for sequential data.
    // 5. Implement a more robust evaluation system, including metrics like precision, recall, and F1-score, to better assess the performance of the model on different types of data and tasks.
    // 6. Add functionality for hyperparameter tuning, allowing users to easily experiment with different learning rates, activation functions, and network architectures to find the best configuration for their specific problem.

    
    // NN architecture
    const int INPUT = 3;
    const int OUTPUT = 1;
    
    const string activation_func = "sigmoid";
    // sigmoid  <- By default if no param passsed through
    // relu
    // leaky_relu 
    // linear 
    // tanh 
    // elu 
    
    // Model metadata
    const string model_name = "TestModel";
    
    Network::metadata meta;
    meta.model_name = model_name;
    meta.model_version = "1.0.0";
    meta.author = "Alex";
    meta.creation_timestamp = "2024-06-01 12:00:00";
    meta.timestamp = "2024-06-01 12:00:00";

    // Files
    const string SAVE_FILE = "../models/" + model_name + ".json";
    const string INPUT_FILE = "../data/io/inputs.in";
    const string TARGET_FILE = "../data/io/targets.in";
    const string MINST_INPUTS_FILE = "../data/mnist/train-images.idx3-ubyte";
    const string MINST_LABELS_FILE = "../data/mnist/train-labels.idx1-ubyte";

    // Training parameters
    const int epochs = 50000;
    const double learning_rate = 0.05;
    const int checkpointInterval = 100;
    const int displayInterval = 1;
    const int task = 1;             
    // 0 - Create
    // 1 - Learn
    // 2 - Predict
    // 3 - Test mnist


    switch (task)   
    {
    
    case 0: // Create
        {   
            Network nn = Network( {INPUT, 5, 5, OUTPUT}, activation_func , meta );  
            nn.save_modal(SAVE_FILE);
            break; 
        }

    case 1: // Learn
        {   
            Network nn = Network( nn.get_params(SAVE_FILE) );  

            nn.load_modal(SAVE_FILE);
            nn.loadData(INPUT, OUTPUT, INPUT_FILE, TARGET_FILE);
    
            nn.backpropagate(learning_rate, epochs, checkpointInterval, displayInterval); 
    
            nn.save_modal(SAVE_FILE);
            break;
        }

    case 2: // Predict
        {
            Network nn = Network( nn.get_params(SAVE_FILE) );    
            Network::TrainingData data;
            
            nn.load_modal(SAVE_FILE);
            nn.loadData(INPUT, OUTPUT, INPUT_FILE, TARGET_FILE);
    
            for (int i = 0; i < data.inputs.size(); i++)
            {
                vector<double> prediction =  nn.predict(data.inputs[i]);  

                cout << "Input " << i << ": ";
                for (double val : data.inputs[i]) 
                    cout << val << " ";
                cout << "-> AI says: " << prediction[0] << endl;
            }
        break;
        }


    case 3: // Test mnist
        {
            Network nn = Network( nn.get_params(SAVE_FILE) );    

            nn.load_modal(SAVE_FILE);

            const int max_samples = 1000; 
            const int tests_number = 1000;

            nn.MnistTest(MINST_INPUTS_FILE, MINST_LABELS_FILE, max_samples, tests_number);

            break;
        }
    }   
    return 0;
}