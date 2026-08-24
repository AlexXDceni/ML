#define NOBYTE    
#define NOMINMAX   

#include <iostream>
#include <string>

#include "include/neural_network.hpp"
#include "include/utils/nn_signal_handler_util.hpp"
#include "include/utils/mnist_util.hpp"
#include "include/utils/functions_util.hpp"

using namespace std;

const string model_name = "test_model";

const int task = -1;             
// 0 - Create
// 1 - Change metadata
// 2 - Learn <- with input and target files
// 3 - Predict
// 4 - Test mnist
// 5 - Learn mnist
// 6 - Classify points
// 7 - Generate Math Function

namespace Tasks{
    void create(){
        NeuralNetwork::metadata meta;
        meta.model_name = model_name;
        meta.model_version = "1.0.0";
        meta.author = "Alex";
        meta.creation_timestamp = get_time();
        
        
        // NN architecture
        const int INPUT = 3;
        const int OUTPUT = 1;
        
        vector<string> activations = {
            "none",
            "sigmoid",
            "sigmoid",
            "sigmoid"
        };
        // sigmoid 
        // relu
        // leaky_relu 
        // linear 
        // tanh 
        // elu 
        
        NeuralNetwork nn = NeuralNetwork( {INPUT, 5, 5, OUTPUT}, activations, meta ); 
        
        nn.update_timestamp(); 
        nn.save_model(model_name);
    }
    void change_metadata(){
        NeuralNetwork nn = NeuralNetwork( nn.get_params(model_name) );  
        nn.load_model(model_name);
        
        
        nn.meta.model_version = "1.0.0";
        nn.update_timestamp();
        
        
        nn.save_model(model_name);
    }
    void learn(){
        const bool saveOnInterrupt = true;  // If true, the model will be saved when the program receives an interrupt signal (Ctrl+C)
    
        NeuralNetwork nn = NeuralNetwork( nn.get_params(model_name) );  
        NeuralNetworkSignalHandler::signalHandler(&nn, saveOnInterrupt);
        nn.load_model(model_name);
        
        nn.loadData(DIRS::INPUT_FILE, DIRS::TARGET_FILE);
        
        const int epochs = 50000;
        const double learning_rate = 0.05;
        const int checkpointInterval = -1; // -1 means no checkpoints, otherwise it will save the model every checkpointInterval epochs
        const int displayInterval = 1;  // Display loss every displayInterval epochs, -1 means no display
        
        nn.backpropagate(learning_rate, epochs, checkpointInterval, displayInterval); 
        
        
        nn.save_model(model_name);
    }
    void predict(){
        NeuralNetwork nn = NeuralNetwork( nn.get_params(model_name) );    
        nn.load_model(model_name);
        nn.loadData(DIRS::INPUT_FILE, DIRS::TARGET_FILE);


        for (int i = 0; i < nn.data.inputs.size(); i++)
        {
            vector<double> prediction =  nn.predict(nn.data.inputs[i]); 
            cout << "Input " << i << ": ";
            for (double val : nn.data.inputs[i]) 
                cout << val << " ";
            cout << "-> AI says: " << prediction[0] << endl;
        }
    }
    void test_mnist(){
        NeuralNetwork nn = NeuralNetwork( nn.get_params(DIRS::MNIST_MODEL_NAME) );
        nn.load_model(DIRS::MNIST_MODEL_NAME);

        const int max_samples = 1000; 
        const int tests_number = 1000;

        MNIST::MnistTest(nn, DIRS::MNIST_INPUTS_FILE, DIRS::MNIST_LABELS_FILE, max_samples, tests_number);
    }
    void learn_mnist(){
        const bool saveOnInterrupt = true;  // If true, the model will be saved when the program receives an interrupt signal (Ctrl+C)

        NeuralNetwork nn = NeuralNetwork( nn.get_params(DIRS::MNIST_MODEL_NAME) );  
        NeuralNetworkSignalHandler::signalHandler(&nn, saveOnInterrupt);
        nn.load_model(DIRS::MNIST_MODEL_NAME);

        const int mnist_max_samples = -1;  // -1 means load all samples, otherwise it will load only the specified number of samples from the mnist dataset


        MNIST::loadMnist(nn, DIRS::TRAINING_MNIST_INPUTS_FILE, DIRS::TRAINING_MNIST_LABELS_FILE, mnist_max_samples); 

        const int epochs = 50000;
        const double learning_rate = 0.05;
        const int checkpointInterval = -1; // -1 means no checkpoints, otherwise it will save the model every checkpointInterval epochs
        const int displayInterval = 1;  // Display loss every displayInterval epochs, -1 means no display

        nn.backpropagate(learning_rate, epochs, checkpointInterval, displayInterval); 


        nn.save_model(DIRS::MNIST_MODEL_NAME);
    }
    void classify(){
        const int RESOLUTION = 800; 

        vector<string> activations = { "none", "sigmoid", "sigmoid", "sigmoid" };

        NeuralNetwork nn({2, 16, 16, 1}, activations); 
        classify_points(nn,RESOLUTION,true);
    }
    void generate_functions(){
        const int RESOLUTION = 800; 
        
        vector<string> activations = { "none", "sigmoid", "sigmoid", "sigmoid" };
        NeuralNetwork nn({2, 64, 64, 1}, activations); ;
        // generate_and_learn_map(nn, screen, RESOLUTION, GUI_HUD::julia_generator);
        old_generation(nn, RESOLUTION);
    }
}

int main( int argc , char* argv[] )
{
    switch (task)   
    {
        case 0:   
            Tasks::create();    
            break; 
        case 1:
            Tasks::change_metadata();
            break;
        case 2: 
            Tasks::learn();
            break;
        case 3: 
            Tasks::predict();
            break;
        case 4: 
            Tasks::test_mnist();
            break;
        case 5:
            Tasks::learn_mnist();
            break;
        case 6:
            Tasks::classify();
            break;
        case 7:
            Tasks::generate_functions();
            break;
        default:
            cerr<<"No test case selected, change task value.";
            break;
    }   
    return 0;
}