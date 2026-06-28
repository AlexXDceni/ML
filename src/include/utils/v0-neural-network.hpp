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

#include "include/json.hpp"

using namespace std;
using json = nlohmann::json;
using namespace std::chrono;


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

          // Layers

        class Node
        {
        public:
            double value;
            double bias;
            vector<double> weights;

            Node(int numNodesNextLayer = 0)
            {
                value = 0;
                bias = 0;
                for (int i = 0; i < numNodesNextLayer; ++i)
                {
                    weights.push_back(get_random());
                }
            }
        };
        class Layer
        {
        public:
            vector<Node> nodes;

            Layer() {};
            Layer(int n, int numNodesNextLayer)
            {
                for (int i = 0; i < n; ++i)
                {
                    nodes.push_back(Node(numNodesNextLayer));
                }
            }
        };
        
        // Data

        struct TrainingData
        {
            vector<vector<double>> inputs;
            vector<vector<double>> targets;
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

        string activation_func;
        vector<int> sizes;
        
        int input_nodes;
        vector<int> hidden_layers;
        int output_nodes;

        Layer input;
        vector<Layer> hidden;
        Layer output;

        // Constructor

        NeuralNetwork(vector<int> list, string activationFunc = "sigmoid", metadata params = {}) : sizes(list), activation_func(activationFunc), meta(params) { init(); }
        NeuralNetwork(initializer_list<int> list, string activationFunc = "sigmoid", metadata params = {}) : sizes(list), activation_func(activationFunc), meta(params) { init(); }
        NeuralNetwork(tuple<vector<int>, string, metadata> params) : sizes(get<0>(params)), activation_func(get<1>(params)), meta(get<2>(params)) { init(); }
        void init()
        {
            if (sizes.size() < 2)
            {
                cerr << "The Neural Network has less then 2 layers.";
                return;
            }
            input_nodes = sizes[0];
            output_nodes = sizes.back();

            int numLayersAfterInput = (sizes.size() > 1) ? sizes[1] : 0;
            input = Layer(input_nodes, numLayersAfterInput);

            for (int i = 1; i < sizes.size() - 1; ++i)
            {
                hidden_layers.push_back(sizes[i]);
                hidden.push_back(Layer(sizes[i], sizes[i + 1]));
            }
            output = Layer(output_nodes, 0);
        }
        ~NeuralNetwork() {}

        // Save/Load

        void update_timestamp() {
            meta.timestamp = get_time();
        }
        void load_model(const string &filename){

            cout << "Loading model from " << filename << "..." <<endl;

            ifstream file(filename);

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

            vector<vector<double>> input_weights = config["params"]["input_layer"]["weights"];
            for (int i = 0; i < input.nodes.size(); i++) {
                for (int j = 0; j < input.nodes[i].weights.size(); j++) {
                    input.nodes[i].weights[j] = input_weights[i][j];
                }
            }

            for (int i = 0; i < hidden.size(); i++) {
                string layer_key = "hidden_layer_" + to_string(i);
                
                vector<vector<double>> hidden_weights = config["params"][layer_key]["weights"];
                vector<double> hidden_biases = config["params"][layer_key]["bias"];

                for (int n = 0; n < hidden[i].nodes.size(); n++) {
                    hidden[i].nodes[n].bias = hidden_biases[n];
                    for (int w = 0; w < hidden[i].nodes[n].weights.size(); w++) {
                        hidden[i].nodes[n].weights[w] = hidden_weights[n][w];
                    }
                }
            }

            vector<double> output_biases = config["params"]["output_layer"]["bias"];
            for (int i = 0; i < output.nodes.size(); i++) {
                output.nodes[i].bias = output_biases[i];
            }
        
            cout << "Neural model loaded successfully!" << endl;
        }
        void save_model(const string &filename){

            cout << "Saving model to " << filename << "..." << endl;
            
            ofstream file(filename);

            if (!file.is_open()) {
                cerr << "Error opening file for writing." << endl;
                return;
            }

            json config;

            config["model_name"] = meta.model_name;

            config["metadata"]["model_version"] = meta.model_version;
            config["metadata"]["author"] = meta.author;
            config["metadata"]["creation_timestamp"] = meta.creation_timestamp;
            config["metadata"]["timestamp"] = get_time();
            
            long long param_count = 0;
            for (int i = 0; i < sizes.size() - 1; i++) {
                param_count += (sizes[i]+1)*sizes[i+1]; 
            }

            config["metadata"]["param_count"] = param_count; 

            config["architecture"]["input_layer"] = input_nodes;
            config["architecture"]["hidden_layers"] = hidden_layers;
            config["architecture"]["output_layer"] = output_nodes;
            config["architecture"]["activation_function"] = activation_func;

            for (int i = 0; i < input.nodes.size(); i++) {
                for (int j = 0; j < input.nodes[i].weights.size(); j++) {
                    config["params"]["input_layer"]["weights"][i][j] = input.nodes[i].weights[j];
                }
            }

            for (int i = 0; i < hidden.size(); i++) {
                string layer_key = "hidden_layer_" + to_string(i);

                for (int n = 0; n < hidden[i].nodes.size(); n++) {
                    config["params"][layer_key]["bias"][n] = hidden[i].nodes[n].bias;
                    for (int w = 0; w < hidden[i].nodes[n].weights.size(); w++) {
                        config["params"][layer_key]["weights"][n][w] = hidden[i].nodes[n].weights[w];
                    }
                }
            }

            for (int i = 0; i < output.nodes.size(); i++) {
                config["params"]["output_layer"]["bias"][i] = output.nodes[i].bias;
            }
            

            file << config.dump(4); 
            file.close();
            cout << "Model saved successfully!" << endl;
        }
        tuple<vector<int>, string, metadata> get_params(const string &filename){
            
            ifstream file(filename);

            if (!file.is_open()) {
                cerr << "Error: Load file not found" << endl;
                // return {{}, "", {}};
                return make_tuple(vector<int>{}, string{""}, metadata{});
            }

            json config;
            file >> config;
            file.close();

            vector<int> saved_sizes;
            saved_sizes.push_back(config["architecture"]["input_layer"]);
        
            vector<int> hidden_layers = config["architecture"]["hidden_layers"];
            for (int h_size : hidden_layers) {
                saved_sizes.push_back(h_size);
            }

            saved_sizes.push_back(config["architecture"]["output_layer"]);
            
            string activation_func = config["architecture"]["activation_function"];

            metadata meta_data;
            meta_data.model_name = config["model_name"];
            meta_data.model_version = config["metadata"]["model_version"];
            meta_data.author = config["metadata"]["author"];
            meta_data.creation_timestamp = config["metadata"]["creation_timestamp"];
            meta_data.timestamp = config["metadata"]["timestamp"];
            meta_data.param_count = config["metadata"]["param_count"];

            return {saved_sizes, activation_func, meta_data};
        }
        void loadData(const string &inputFile, const string &targetFile)
        {   

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

        double ACTIVATION(double value, string activation_func, double alpha = 1.0)
        {
            
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
        double GET_DERIVATIVE(double value, string activation_func) {

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
                for (auto &node : layer.nodes)
                {
                    node.value = 0;
                }
            }
            for (auto &node : output.nodes)
            {
                node.value = 0;
            }
        }
        void feedForward(const vector<double> &inputValues)
        {

            feedZero();

            if (inputValues.size() != input.nodes.size())
            {
                cerr << "Error: Input size mismatch!" << endl;
                return;
            }

            for (int i = 0; i < input.nodes.size(); ++i)
            {
                input.nodes[i].value = inputValues[i];
            }

            bool areHiddenLayers = (hidden.size() > 0) ? 1 : 0;

            // I-O

            if (!areHiddenLayers)
            {

                for (int j = 0; j < output.nodes.size(); ++j)
                {
                    for (int i = 0; i < input.nodes.size(); ++i)
                    {
                        output.nodes[j].value += input.nodes[i].value * input.nodes[i].weights[j];
                    }
                    output.nodes[j].value = ACTIVATION(output.nodes[j].value + output.nodes[j].bias, activation_func);
                }
            }

            else
            {

                // I - L1

                for (int j = 0; j < hidden[0].nodes.size(); ++j)
                {
                    for (int i = 0; i < input.nodes.size(); ++i)
                    {
                        hidden[0].nodes[j].value += input.nodes[i].value * input.nodes[i].weights[j];
                    }
                    hidden[0].nodes[j].value = ACTIVATION(hidden[0].nodes[j].value + hidden[0].nodes[j].bias, activation_func);
                }

                // L1 - Ln

                for (int k = 0; k < hidden.size() - 1; ++k)
                    
                
                    #pragma omp parallel for    // Parallelizing the outer loop for hidden layers

                    for (int j = 0; j < hidden[k + 1].nodes.size(); ++j)
                    {
                        for (int i = 0; i < hidden[k].nodes.size(); ++i)
                        {
                            hidden[k + 1].nodes[j].value += hidden[k].nodes[i].value * hidden[k].nodes[i].weights[j];
                        }
                        hidden[k + 1].nodes[j].value = ACTIVATION(hidden[k + 1].nodes[j].value + hidden[k + 1].nodes[j].bias, activation_func);
                    }

                // Ln - O

                for (int j = 0; j < output.nodes.size(); ++j)
                {
                    for (int i = 0; i < hidden[hidden.size() - 1].nodes.size(); ++i)
                    {
                        output.nodes[j].value += hidden[hidden.size() - 1].nodes[i].value * hidden[hidden.size() - 1].nodes[i].weights[j];
                    }
                    output.nodes[j].value = ACTIVATION(output.nodes[j].value + output.nodes[j].bias, activation_func);
                }
            }
        }

        // Learn

        void backpropagate(double learning_rate, int epochs, int checkpointInterval = 0, int displayInterval = 1) {
            for (int e = 0; e < epochs; ++e) {


                // Start measuring time
                auto start_time = std::chrono::high_resolution_clock::now();


                double total_mse = 0;

                for (int d = 0; d < data.inputs.size(); ++d) {
                    feedForward(data.inputs[d]);

                    vector<double> output_deltas(output.nodes.size());
                    for (int i = 0; i < output.nodes.size(); ++i) {
                        double val = output.nodes[i].value;
                        double error = data.targets[d][i] - val; 
                        total_mse += error * error;
                        
                        output_deltas[i] = error * GET_DERIVATIVE(val, activation_func);
                    }


                    vector<vector<double>> hidden_deltas(hidden.size());
                   
                    for (int k = hidden.size() - 1; k >= 0; --k) {

                        hidden_deltas[k].resize(hidden[k].nodes.size());

                        #pragma omp parallel for    // Parallelizing the outer loop for hidden layers

                        for (int i = 0; i < hidden[k].nodes.size(); ++i) {
                            double error = 0;
                            
                            if (k == hidden.size() - 1) {
                                for (int j = 0; j < output.nodes.size(); ++j) {
                                    error += output_deltas[j] * hidden[k].nodes[i].weights[j];
                                }
                            } 
                            else {
                                for (int j = 0; j < hidden[k+1].nodes.size(); ++j) {
                                    error += hidden_deltas[k+1][j] * hidden[k].nodes[i].weights[j];
                                }
                            }
                            
                            double val = hidden[k].nodes[i].value;
                            hidden_deltas[k][i] = error * GET_DERIVATIVE(val, activation_func);
                        }
                    }


                    int last_h = hidden.size() - 1;
                    for (int j = 0; j < output.nodes.size(); ++j) {
                        output.nodes[j].bias += learning_rate * output_deltas[j];
                        for (int i = 0; i < hidden[last_h].nodes.size(); ++i) {
                            hidden[last_h].nodes[i].weights[j] += learning_rate * output_deltas[j] * hidden[last_h].nodes[i].value;
                        }
                    }

                    for (int k = last_h; k > 0; --k) {
                        for (int i = 0; i < hidden[k].nodes.size(); ++i) {
                            hidden[k].nodes[i].bias += learning_rate * hidden_deltas[k][i];
                        }
                        for (int j = 0; j < hidden[k-1].nodes.size(); ++j) {
                            for (int i = 0; i < hidden[k].nodes.size(); ++i) {
                                hidden[k-1].nodes[j].weights[i] += learning_rate * hidden_deltas[k][i] * hidden[k-1].nodes[j].value;
                            }
                        }
                    }

                    for (int i = 0; i < hidden[0].nodes.size(); ++i) {
                        hidden[0].nodes[i].bias += learning_rate * hidden_deltas[0][i];
                        for (int j = 0; j < input.nodes.size(); ++j) {
                            input.nodes[j].weights[i] += learning_rate * hidden_deltas[0][i] * input.nodes[j].value;
                        }
                    }
                }


                // End measuring time
                auto end_time = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();


                if ( displayInterval > 0 && e % displayInterval == 0) {
                    cout << "Epoch " << e << " | MSE: " << (total_mse / (data.inputs.size() * output.nodes.size())) << " | finished in " << duration / 1000.0 << " seconds.\n" ;
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

            for (auto &node : output.nodes)
            {
                if (node.value >= 0.5)
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
            if(inputValues.size() != input.nodes.size()) {
                cerr << "Error: Input size mismatch!" << endl;
                return {};
            }
            this->feedForward(inputValues);
            vector<double> results;

            for (auto &node : output.nodes)
            {
                results.push_back((node.value > 0.00001) ? node.value : 0);
            }
            return results;
        }
        vector<double> predictBiggest(const vector<double> &inputValues)
        {   

            if(inputValues.size() != input.nodes.size()) {
                cerr << "Error: Input size mismatch!" << endl;
                return {};
            }
            
            this->feedForward(inputValues);

            double result = 0;
            double biggest = -1.0;

            for (int i = 0; i < output.nodes.size(); ++i)
            {
                if (output.nodes[i].value > biggest)
                {
                    biggest = output.nodes[i].value;
                    result = i;
                }
            }
            return {result};
        }

};