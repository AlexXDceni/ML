#include <iostream>
#include <string>
#include "neural_network.hpp"
// #include <nlohmann/json.json>

using namespace std;

int main( int argc , char* argv[] )
{
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
    const string model_name = "TEST_MODEL";

    // Files
    const string SAVE_FILE = "../models/" + model_name + ".json";
    const string INPUT_FILE = "../data/io/inputs.in";
    const string TARGET_FILE = "../data/io/targets.in";
    const string TRAINING_MINST_INPUTS_FILE = "../data/mnist/train-images.idx3-ubyte";
    const string TRAINING_MINST_LABELS_FILE = "../data/mnist/train-labels.idx1-ubyte";
    const string MINST_INPUTS_FILE = "../data/mnist/t10k-images.idx3-ubyte";
    const string MINST_LABELS_FILE = "../data/mnist/t10k-labels.idx1-ubyte";


    // Training parameters
    const int epochs = 50000;
    const double learning_rate = 0.05;

    const int checkpointInterval = 10;
    const int displayInterval = 1;
    const int mnist_max_samples = -1;

    const int task = 0;             
    // 0 - Create
    // 1 - Change metadata
    // 2 - Learn
    // 3 - Predict
    // 4 - Test mnist
    // 5 - Learn mnist


    switch (task)   
    {
    
    case 0: // Create
        {           
            Network::metadata meta;
            meta.model_name = model_name;
            meta.model_version = "1.0.0";
            meta.author = "Alex";
            meta.creation_timestamp = get_time();
            meta.timestamp = get_time();

            Network nn = Network( {INPUT, 5, 5, OUTPUT}, activation_func , meta );  
            nn.save_modal(SAVE_FILE);
            break; 
        }
        
        case 1: // Change metadata   
        {
            Network nn = Network( nn.get_params(SAVE_FILE) );  
            nn.load_modal(SAVE_FILE);
            
            nn.meta.model_version = "1.0.0";
            nn.meta.timestamp = get_time();
    
            nn.save_modal(SAVE_FILE);
            break; 
        }

    case 2: // Learn
        {   
            Network nn = Network( nn.get_params(SAVE_FILE) );  
            nn.load_modal(SAVE_FILE);
            nn.loadData(INPUT, OUTPUT, INPUT_FILE, TARGET_FILE);
            nn.backpropagate(learning_rate, epochs, checkpointInterval, displayInterval); 
            nn.save_modal(SAVE_FILE);
            break;
        }

    case 3: // Predict
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


        case 4: // Test mnist
            {
                Network nn = Network( nn.get_params(SAVE_FILE) );
                nn.load_modal(SAVE_FILE);
                const int max_samples = 1000; 
                const int tests_number = 1000;
                nn.MnistTest(MINST_INPUTS_FILE, MINST_LABELS_FILE, max_samples, tests_number);
                break;
            }
        case 5: // Learn mnist
            {
                Network nn = Network( nn.get_params(SAVE_FILE) );  
                nn.load_modal(SAVE_FILE);
                nn.loadMnist(TRAINING_MINST_INPUTS_FILE, TRAINING_MINST_LABELS_FILE, mnist_max_samples); 
                nn.backpropagate(learning_rate, epochs, checkpointInterval, displayInterval); 
                nn.save_modal(SAVE_FILE);
                break;
            }
    }   
    return 0;
}