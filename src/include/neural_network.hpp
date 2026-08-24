#pragma once
#include <iostream>
#include <vector>
#include <initializer_list>
#include <fstream>
#include <random>
#include <cmath>
#include <string>
#include <iomanip>
#include <tuple>
#include <chrono>
#include <sstream>
#include <omp.h>
#include <filesystem>

#include "external/json.hpp"

using namespace std;
using json = nlohmann::json;
using namespace std::chrono;



namespace DIRS {
    //? Directries
    const string MODELS_DIR = "../models/";
    const string DATA_DIR = "../data/";
    const string IO_DIR = DATA_DIR + "io/";
    const string MNIST_DIR = DATA_DIR + "mnist/";
    const string IMAGES_DIR = DATA_DIR + "images/";

    //? Files
    const string INPUT_FILE = IO_DIR + "inputs.in";
    const string TARGET_FILE = IO_DIR + "targets.in";

    //? MNIST dataset files
    const string MNIST_MODEL_NAME = "mnist_model"; 
    const string TRAINING_MNIST_INPUTS_FILE = MNIST_DIR + "train-images.idx3-ubyte";
    const string TRAINING_MNIST_LABELS_FILE = MNIST_DIR + "train-labels.idx1-ubyte";
    const string MNIST_INPUTS_FILE = MNIST_DIR + "t10k-images.idx3-ubyte";
    const string MNIST_LABELS_FILE = MNIST_DIR + "t10k-labels.idx1-ubyte";
}

double get_random()
{
    thread_local mt19937 gen(random_device{}());
    return uniform_real_distribution<double>{-1, 1}(gen);
}
string get_time()
{
    std::time_t time_t_format = std::chrono::system_clock::to_time_t(system_clock::now());
    std::ostringstream ss;
    ss << std::put_time(std::localtime(&time_t_format), "%Y-%m-%d %H:%M:%S");
    return ss.str();
}

class NeuralNetwork
{
    public:

        class Layer
        {
            public:
                int num_neurons;
                int num_next_layer_neurons; 
                string activation;

                vector<double> values;  
                vector<double> biases; 
                vector<double> weights; 

                //vector<double> errors;

                Layer() {}
                Layer(int n, int m) : num_neurons(n), num_next_layer_neurons(m) {
                    values.resize(n, 0.0);
                    biases.resize(n, 0.0);
                    //errors.resize(n, 0.0);
                    weights.resize(n * m);
                    for (int i = 0; i < n * m; ++i) {
                        weights[i] = get_random();
                    }
                }
        };
     
        // Data

        struct TrainingData
        {
            vector<vector<double>> inputs;
            vector<vector<double>> targets;
            void clear(){
                inputs.clear();
                targets.clear();
            }
        };
        struct metadata 
        {
            string model_name;
            string model_version;
            string author;
            string creation_timestamp;
            string timestamp;
            long long param_count;
        };

        metadata meta;
        TrainingData data;

        vector<int> sizes;
        
        int input_nodes;
        vector<int> hidden_layers;
        int output_nodes;

        Layer input;
        vector<Layer> hidden;
        Layer output;

        vector<double> output_deltas;
        vector<vector<double>> hidden_deltas;

        // Constructor

        NeuralNetwork(vector<int> list, vector<string> activations, metadata params = {}) : sizes(list), meta(params) { init(activations); }
        NeuralNetwork(initializer_list<int> list, vector<string> activations , metadata params = {}) : sizes(list), meta(params){ init(activations);}
        NeuralNetwork(tuple<vector<int>, vector<string>, metadata> params) : sizes(get<0>(params)), meta(get<2>(params)) { init(get<1>(params)); }
        ~NeuralNetwork() {}
        void init(vector<string> activations)
        {   

            hidden.clear();
            hidden_layers.clear();  
            if (sizes.size() < 2)
            {
                cerr << "The Neural Network has less then 2 layers.";
                return;
            }
            input_nodes = sizes[0];
            output_nodes = sizes.back();

            int numLayersAfterInput = (sizes.size() > 1) ? sizes[1] : 0;
            input = Layer(input_nodes, numLayersAfterInput);
            input.activation = activations[0];


            for (int i = 1; i < sizes.size() - 1; ++i)
            {
                hidden_layers.push_back(sizes[i]);
                hidden.push_back(Layer(sizes[i], sizes[i + 1]));
            }

            for( int i = 0 ; i < hidden.size(); i++ ){
                hidden[i].activation = activations[i+1];
            }


            output = Layer(output_nodes, 0);
            output.activation = activations.back();     
        }
        

        // Save/Load

        void update_timestamp() {
            meta.timestamp = get_time();
        }
        void load_model(const string &model_name){

            cout << "Loading model from " << model_name << "..." <<endl;

            ifstream file( DIRS::MODELS_DIR + model_name + "/config.json");

            if (!file.is_open()) {
                cerr << "Error: Load file not found" << endl;
                return;
            }

            json config;
            file >> config;
            file.close();


            meta.model_name = config["model_name"];

            meta.model_version = config["metadata"]["model_version"];
            meta.author = config["metadata"]["author"];
            meta.creation_timestamp = config["metadata"]["creation_timestamp"];
            meta.timestamp = config["metadata"]["timestamp"];
            meta.param_count = config["metadata"]["param_count"];



            vector<int> saved_sizes;
            saved_sizes.push_back(config["architecture"]["input_layer"]);
            
            vector<int> hidden_layers = config["architecture"]["hidden_layers"];
            for (int h_size : hidden_layers) {
                saved_sizes.push_back(h_size);
                
            }
            saved_sizes.push_back(config["architecture"]["output_layer"]);

            if (saved_sizes.size() != sizes.size()) {
                cerr << "Error: Network layers count mismatch." << endl;
                return;
            }
            
            
            for (int i = 0; i < sizes.size(); i++) {
                if (saved_sizes[i] != sizes[i]) {
                    cerr << "Error: Network layers mismatch." << endl;
                    cerr << "Expected " << sizes[i] << " nodes, but found " << saved_sizes[i] << " in JSON." << endl;
                    return;
                }
            }

            

            vector<string> activs = config["architecture"]["activation_function"];
            

            input.activation = activs[0];
            vector<vector<double>> input_weights = config["params"]["input_layer"]["weights"];
            for (int i = 0; i < input.num_neurons; i++) {
                for (int j = 0; j < input.num_next_layer_neurons; j++) {
                    input.weights[i * input.num_next_layer_neurons + j] = input_weights[i][j];
                }
            }

            for (int i = 0; i < hidden.size(); i++) {
                string layer_key = "hidden_layer_" + to_string(i);
                
                hidden[i].activation = activs[i+1];
                vector<vector<double>> hidden_weights = config["params"][layer_key]["weights"];
                vector<double> hidden_biases = config["params"][layer_key]["bias"];
                
                for (int n = 0; n < hidden[i].num_neurons; n++) {
                    hidden[i].biases[n] = hidden_biases[n];
                    for (int w = 0; w < hidden[i].num_next_layer_neurons; w++) {
                        hidden[i].weights[n * hidden[i].num_next_layer_neurons + w] = hidden_weights[n][w];
                    }
                }
                
            }

            output.activation = activs.back();
            vector<double> output_biases = config["params"]["output_layer"]["bias"];
            for (int i = 0; i < output.num_neurons; i++) {
                output.biases[i] = output_biases[i];
            }
            








            cout << "Neural model loaded successfully!" << endl;
        }
        void save_model(const string &model_name){

            string path = DIRS::MODELS_DIR + model_name;
            error_code ec;

            filesystem::create_directories(path, ec);

            if (ec) {
                cerr << "Error: Could not create directory " << path << " -> " << ec.message() << endl;
                return; 
            }

            cout << "Saving model to " << path << "..." << endl;
        

            ofstream file(path + "/config.json");

            if (!file.is_open()) {
                cerr << "Error opening file for writing." << endl;
                return;
            }

            json config;

            config["model_name"] = meta.model_name;

            config["metadata"]["model_version"] = meta.model_version;
            config["metadata"]["author"] = meta.author;
            config["metadata"]["creation_timestamp"] = meta.creation_timestamp;
            update_timestamp();
            config["metadata"]["timestamp"] = meta.timestamp;
            
            long long param_count = 0;
            for (int i = 0; i < sizes.size() - 1; i++) {
                param_count += (sizes[i]+1)*sizes[i+1]; 
            }

            meta.param_count = param_count;
            config["metadata"]["param_count"] = param_count; 

            config["architecture"]["input_layer"] = input_nodes;
            config["architecture"]["hidden_layers"] = hidden_layers;
            config["architecture"]["output_layer"] = output_nodes;

            vector<string> activation_func;

            activation_func.push_back(input.activation);
            for (auto &layer : hidden)
                activation_func.push_back(layer.activation);
            activation_func.push_back(output.activation);

            config["architecture"]["activation_function"] = activation_func;

            for (int i = 0; i < input.num_neurons; i++) {
                for (int j = 0; j < input.num_next_layer_neurons; j++) {
                    config["params"]["input_layer"]["weights"][i][j] = input.weights[i * input.num_next_layer_neurons + j];
                }
            }
            

            for (int i = 0; i < hidden.size(); i++) {
                string layer_key = "hidden_layer_" + to_string(i);

                for (int n = 0; n < hidden[i].num_neurons; n++) {
                    config["params"][layer_key]["bias"][n] = hidden[i].biases[n];
                    for (int w = 0; w < hidden[i].num_next_layer_neurons; w++) {
                        config["params"][layer_key]["weights"][n][w] = hidden[i].weights[n * hidden[i].num_next_layer_neurons + w];
                    }
                }
            }
            
            for (int i = 0; i < output.num_neurons; i++) {
                config["params"]["output_layer"]["bias"][i] = output.biases[i];
            }
            

            file << config.dump(4); 
            file.close();
            cout << "Model saved successfully!" << endl;
        }
        tuple<vector<int>, vector<string>,  metadata> get_params(const string &model_name){
            
            ifstream file(DIRS::MODELS_DIR + model_name + "/config.json");

            if (!file.is_open()) {
                cerr << "Error: Load file not found" << endl;
                // return {{}, "", {}};
                return make_tuple(vector<int>{}, vector<string>{}, metadata{});
            }

            json config;
            file >> config;
            file.close();

            vector<string> activations = config["architecture"]["activation_function"];
            vector<int> saved_sizes;
            saved_sizes.push_back(config["architecture"]["input_layer"]);

            vector<int> hidden_layers = config["architecture"]["hidden_layers"];
            for (int h_size : hidden_layers) {
                saved_sizes.push_back(h_size);
            }



            saved_sizes.push_back(config["architecture"]["output_layer"]);
            
            
            metadata meta_data;
            meta_data.model_name = config["model_name"];
            meta_data.model_version = config["metadata"]["model_version"];
            meta_data.author = config["metadata"]["author"];
            meta_data.creation_timestamp = config["metadata"]["creation_timestamp"];
            meta_data.timestamp = config["metadata"]["timestamp"];
            meta_data.param_count = config["metadata"]["param_count"];

            return {saved_sizes, activations, meta_data};
        }
        void loadData(const string &inputFile, const string &targetFile)
        {   

            data.inputs.clear();
            data.targets.clear();

            cout << "Loading data from " << inputFile << " and " << targetFile << "..." << endl;
            ifstream inputs(inputFile);
            ifstream targets(targetFile);

            if (!inputs.is_open() || !targets.is_open())
            {
                cerr << "Error: Could not open " << inputFile << " or " << targetFile << endl;
                return;
            }

            double val;
            while (inputs >> val)
            {
                vector<double> row;
                row.push_back(val);
                for (int i = 1; i < input_nodes; ++i)
                {
                    if (inputs >> val)
                        row.push_back(val);
                }
                if (row.size() == input_nodes)
                {
                    data.inputs.push_back(row);
                }
            }

            while (targets >> val)
            {
                vector<double> row;
                row.push_back(val);
                for (int i = 1; i < output_nodes; ++i)
                {
                    if (targets >> val)
                        row.push_back(val);
                }
                if (row.size() == output_nodes)
                {
                    data.targets.push_back(row);
                }
            }

            if (data.inputs.empty())
            {
                cerr << "Error: No data loaded from files." << endl;
                return;
            }

            if (data.inputs.size() != data.targets.size())
            {
                cerr << "Error: Mismatch! Inputs: " << data.inputs.size()
                    << " | Targets: " << data.targets.size() << endl;
                return;
            }

            cout << "Data loaded successfully!" << endl;

        }

        // Activation function

        double ACTIVATION(double value, const string &activation_func, double alpha = 1.0)
        {
            if (activation_func == "none")
                return value;
            if (activation_func == "sigmoid")
                return 1.0 / (1.0 + exp(-value));
            if (activation_func == "relu")
                return (value > 0) ? value : 0;
            if (activation_func == "leaky_relu")
                return (value > 0) ? value : 0.01 * value;
            if (activation_func == "tanh")
                return tanh(value);
            if (activation_func == "elu")
                return (value > 0) ? value : alpha * (exp(value) - 1);
            if (activation_func == "linear")
                return value;

            cerr << "Warning: Unknown activation function '" << activation_func << "'. Defaulting to sigmoid." << endl;
            return 1.0 / (1.0 + exp(-value));
        }
        double GET_DERIVATIVE(double value, const string &activation_func) {

            if (activation_func == "none")
                return 1.0;
            if (activation_func == "sigmoid")
                return value * (1.0 - value);
            if (activation_func == "relu")
                return (value > 0) ? 1.0 : 0.0;
            if (activation_func == "leaky_relu")
                return (value > 0) ? 1.0 : 0.01;
            if (activation_func == "tanh")
                return 1.0 - (value * value);
            if (activation_func == "elu")
                return (value > 0) ? 1.0 : (value + 1.0);
            if (activation_func == "linear")
                return 1.0;

            cerr << "Warning: Unknown activation function '" << activation_func << "'. Defaulting to sigmoid derivative." << endl;
            return value * (1.0 - value);
        }

        // Predict

        void feedZero()
        {
            for (auto &layer : hidden)
            {
                fill(layer.values.begin(), layer.values.end(), 0.0);
            }

            fill(output.values.begin(), output.values.end(), 0.0);
        }
        void feedForward(const vector<double> &inputValues)
        {

            feedZero();

            if (inputValues.size() != input.values.size())
            {
                cerr << "Error: Input size mismatch!" << endl;
                return;
            }

            for (int i = 0; i < input.values.size(); ++i)
            {
                input.values[i] = inputValues[i];
            }


            // I-O

            if (hidden.empty())
            {

                for (int j = 0; j < output.values.size(); ++j)
                {
                    for (int i = 0; i < input.values.size(); ++i)
                    {
                        output.values[j] += input.values[i] * input.weights[i * input.num_next_layer_neurons + j];
                    }
                    output.values[j] = ACTIVATION(output.values[j] + output.biases[j], output.activation);
                }
            }

            else
            {

                // I - L1

                for (int j = 0; j < hidden[0].values.size(); ++j)
                {
                    for (int i = 0; i < input.values.size(); ++i)
                    {
                        hidden[0].values[j] += input.values[i] * input.weights[i * input.num_next_layer_neurons + j];
                    }
                    hidden[0].values[j] = ACTIVATION(hidden[0].values[j] + hidden[0].biases[j], hidden[0].activation);
                }

                // L1 - Ln

                for (int k = 0; k < hidden.size() - 1; ++k){
                    
                    //#pragma omp parallel for    // Parallelizing the outer loop for hidden layers

                    for (int j = 0; j < hidden[k + 1].values.size(); ++j)
                    {
                        for (int i = 0; i < hidden[k].values.size(); ++i)
                        {
                            hidden[k + 1].values[j] += hidden[k].values[i] * hidden[k].weights[i * hidden[k].num_next_layer_neurons + j];
                        }
                        hidden[k + 1].values[j] = ACTIVATION(hidden[k + 1].values[j] + hidden[k + 1].biases[j], hidden[k+1].activation);
                    }
                }

                // Ln - O

                for (int j = 0; j < output.values.size(); ++j)
                {
                    for (int i = 0; i < hidden[hidden.size() - 1].values.size(); ++i)
                    {
                        output.values[j] += hidden[hidden.size() - 1].values[i] * hidden[hidden.size() - 1].weights[i * hidden[hidden.size() - 1].num_next_layer_neurons + j];
                    }
                    output.values[j] = ACTIVATION(output.values[j] + output.biases[j], output.activation);
                }
            }
        }

        // Learn

        void backpropagate(double learning_rate, int epochs, int checkpointInterval = 0, int displayInterval = 0) {
            for (int e = 0; e < epochs; ++e) {


                // Start measuring time
                auto start_time = std::chrono::high_resolution_clock::now();


                double total_mse = 0;

                for (int d = 0; d < data.inputs.size(); ++d) {
                    feedForward(data.inputs[d]);

                    output_deltas.resize(output.num_neurons);

                    for (int i = 0; i < output.num_neurons; ++i) {
                        double val = output.values[i];
                        double error = data.targets[d][i] - val; 
                        total_mse += error * error;
                        
                        output_deltas[i] = error * GET_DERIVATIVE(val, output.activation);
                    }

                    if(!hidden.empty())
                    {
                        hidden_deltas.resize(hidden.size());
                    
                        for (int k = hidden.size() - 1; k >= 0; --k) {

                            hidden_deltas[k].resize(hidden[k].num_neurons);

                            //#pragma omp parallel for    // Parallelizing the outer loop for hidden layers

                            for (int i = 0; i < hidden[k].num_neurons; ++i) {
                                double error = 0;
                                
                                if (k == hidden.size() - 1) {
                                    for (int j = 0; j < output.num_neurons; ++j) {
                                        error += output_deltas[j] * hidden[k].weights[i * hidden[k].num_next_layer_neurons + j];
                                    }
                                } 
                                else {
                                    for (int j = 0; j < hidden[k+1].num_neurons; ++j) {
                                        error += hidden_deltas[k+1][j] * hidden[k].weights[i * hidden[k].num_next_layer_neurons + j];
                                    }
                                }
                                
                                double val = hidden[k].values[i];
                                hidden_deltas[k][i] = error * GET_DERIVATIVE(val, hidden[k].activation);
                            }
                        }

                        
                        int last_h = hidden.size() - 1 ;
                        for (int j = 0; j < output.num_neurons; ++j) {
                            output.biases[j] += learning_rate * output_deltas[j];
                            for (int i = 0; i < hidden[last_h].num_neurons; ++i) {
                                hidden[last_h].weights[i * hidden[last_h].num_next_layer_neurons + j] += learning_rate * output_deltas[j] * hidden[last_h].values[i];
                            }
                        }

                        for (int k = last_h; k > 0; --k) {
                            for (int i = 0; i < hidden[k].num_neurons; ++i) {
                                hidden[k].biases[i] += learning_rate * hidden_deltas[k][i];
                            }
                            for (int j = 0; j < hidden[k-1].num_neurons; ++j) {
                                for (int i = 0; i < hidden[k].num_neurons; ++i) {
                                    hidden[k-1].weights[j * hidden[k-1].num_next_layer_neurons + i] += learning_rate * hidden_deltas[k][i] * hidden[k-1].values[j];
                                }
                            }
                        }

                        for (int i = 0; i < hidden[0].num_neurons; ++i) {
                            hidden[0].biases[i] += learning_rate * hidden_deltas[0][i];
                            for (int j = 0; j < input.num_neurons; ++j) {
                                input.weights[j * input.num_next_layer_neurons + i] += learning_rate * hidden_deltas[0][i] * input.values[j];
                            }
                        }
                    }
                    else {
                        
                        for (int i = 0; i < output.num_neurons; ++i) {
                            output.biases[i] += learning_rate * output_deltas[i];
                            for (int j = 0; j < input.num_neurons; ++j) {
                                input.weights[j * input.num_next_layer_neurons + i] += learning_rate * output_deltas[i] * input.values[j];
                            }
                        }
                    }
                }


                // End measuring time
                auto end_time = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();


                if ( displayInterval > 0 && e % displayInterval == 0) {
                    cout << "Epoch " << e << " | MSE: " << (total_mse / (data.inputs.size() * output.num_neurons)) << " | finished in " << duration / 1000.0 << " seconds.\n" ;
                }

                if( e != 0 && checkpointInterval > 0 && e % checkpointInterval == 0) {
                    save_model("../models/checkpoint_epoch_" + to_string(e) + "+" + meta.model_name + ".json");
                }

            }
        }
        
        // Predict
        
        vector<double> predictWholeNum(const vector<double> &inputValues)
        {
            this->feedForward(inputValues);
            vector<double> results;

            for (auto &value : output.values)
            {
                if (value >= 0.5)
                {
                    results.push_back(1);
                }
                else
                {
                    results.push_back(0);
                }
            }
            return results;
        }
        vector<double> predict(const vector<double> &inputValues)
        {
            if(inputValues.size() != input.num_neurons) {
                cerr << "Error: Input size mismatch!" << endl;
                return {};
            }
            this->feedForward(inputValues);
            vector<double> results;

            for (auto &value : output.values)
            {
                results.push_back((value > 0.00001) ? value : 0);
            }
            return results;
        }
        vector<double> predictBiggest(const vector<double> &inputValues)
        {   

            if(inputValues.size() != input.num_neurons) {
                cerr << "Error: Input size mismatch!" << endl;
                return {};
            }
            
            this->feedForward(inputValues);

            double result = 0;
            double biggest = -1.0;

            for (int i = 0; i < output.num_neurons; ++i)
            {
                if (output.values[i] > biggest)
                {
                    biggest = output.values[i];
                    result = i;
                }
            }
            return {result};
        }

};