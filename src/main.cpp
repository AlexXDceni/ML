#include <iostream>
#include <string>
#include "neural_network.hpp"

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
    const string model_name = "test_model";


    // Directries
    const string MODELS_DIR = "../models/";
    const string DATA_DIR = "../data/";
    const string IO_DIR = DATA_DIR + "io/";
    const string MNIST_DIR = DATA_DIR + "mnist/";
    const string IMAGES_DIR = DATA_DIR + "images/";

    // Files
    const string SAVE_FILE = MODELS_DIR + model_name + ".json";
    const string INPUT_FILE = IO_DIR + "inputs.in";
    const string TARGET_FILE = IO_DIR + "targets.in";
    const string TRAINING_MINST_INPUTS_FILE = MNIST_DIR + "train-images.idx3-ubyte";
    const string TRAINING_MINST_LABELS_FILE = MNIST_DIR + "train-labels.idx1-ubyte";
    const string MINST_INPUTS_FILE = MNIST_DIR + "t10k-images.idx3-ubyte";
    const string MINST_LABELS_FILE = MNIST_DIR + "t10k-labels.idx1-ubyte";


    // Training parameters
    const int epochs = 50000;
    const double learning_rate = 0.05;

    const int displayInterval = 1;
    const bool saveOnInterrupt = true;
    const int checkpointInterval = -1;  // -1 means no checkpoints, otherwise it will save the model every checkpointInterval epochs
    const int mnist_max_samples = -1;   // -1 means load all samples, otherwise it will load only the specified number of samples from the mnist dataset

    const int task = -1;             
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
            NeuralNetwork::metadata meta;
            meta.model_name = model_name;
            meta.model_version = "1.0.0";
            meta.author = "Alex";
            meta.creation_timestamp = get_time();
            meta.timestamp = get_time();

            NeuralNetwork nn = NeuralNetwork( {INPUT, 5, 5, OUTPUT}, activation_func , meta );  
            nn.save_model(SAVE_FILE);
            break; 
        }
        
        case 1: // Change metadata   
        {
            NeuralNetwork nn = NeuralNetwork( nn.get_params(SAVE_FILE) );  
            nn.load_model(SAVE_FILE);
            
            nn.meta.model_version = "1.0.0";
            nn.meta.timestamp = get_time();
    
            nn.save_model(SAVE_FILE);
            break; 
        }

    case 2: // Learn
        {   
            NeuralNetwork nn = NeuralNetwork( nn.get_params(SAVE_FILE) );  
            nn.signalHandler(saveOnInterrupt);
            nn.load_model(SAVE_FILE);
            nn.loadData(INPUT, OUTPUT, INPUT_FILE, TARGET_FILE);
            nn.backpropagate(learning_rate, epochs, checkpointInterval, displayInterval); 
            nn.save_model(SAVE_FILE);
            break;
        }

    case 3: // Predict
        {
            NeuralNetwork nn = NeuralNetwork( nn.get_params(SAVE_FILE) );    
            NeuralNetwork::TrainingData data;
            nn.load_model(SAVE_FILE);
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
            NeuralNetwork nn = NeuralNetwork( nn.get_params(SAVE_FILE) );
            nn.load_model(SAVE_FILE);
            const int max_samples = 1000; 
            const int tests_number = 1000;
            nn.MnistTest(MINST_INPUTS_FILE, MINST_LABELS_FILE, max_samples, tests_number);
            break;
        }

    case 5: // Learn mnist
        {
            NeuralNetwork nn = NeuralNetwork( nn.get_params(SAVE_FILE) );  
            nn.signalHandler(saveOnInterrupt);
            nn.load_model(SAVE_FILE);
            nn.loadMnist(TRAINING_MINST_INPUTS_FILE, TRAINING_MINST_LABELS_FILE, mnist_max_samples); 
            nn.backpropagate(learning_rate, epochs, checkpointInterval, displayInterval); 
            nn.save_model(SAVE_FILE);
            break;
        }

    default:
        {
            cout<<"No test case selected, change task value.";
            break;
        }
    }   
    return 0;
}