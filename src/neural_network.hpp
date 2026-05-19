#include <iostream>
#include <vector>
#include <initializer_list>
#include <fstream>
#include <random>
#include <cmath>
#include <string>
#include "include/json.hpp"
using namespace std;
using json = nlohmann::json;


double get_random()
{
    static mt19937 gen(random_device{}());
    return uniform_real_distribution<double>{-1, 1}(gen);
}

class Network
{
    public:
    
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
                
        // Layers

        class Input_Node
        {
        public:
            double value;
            vector<double> weights;

            Input_Node(int numNodesNextLayer)
            {
                value = 0;
                for (auto it = 0; it != numNodesNextLayer; ++it)
                {
                    weights.push_back(get_random());
                }
            }
        };
        class Input_Layer
        {
        public:
            vector<Input_Node> nodes;

            Input_Layer() {};
            Input_Layer(int n, int numNodesNextLayer)
            {
                for (int i = 0; i < n; ++i)
                {
                    nodes.push_back(Input_Node(numNodesNextLayer));
                }
            }
        };
        class Hidden_Node
        {
        public:
            double value;
            double bias;
            vector<double> weights;

            Hidden_Node(int numNodesNextLayer)
            {
                value = 0;
                bias = 0;
                for (int i = 0; i < numNodesNextLayer; ++i)
                {
                    weights.push_back(get_random());
                }
            }
        };
        class Hidden_Layer
        {
        public:
            vector<Hidden_Node> nodes;

            Hidden_Layer() {};
            Hidden_Layer(int n, int numNodesNextLayer)
            {

                for (int i = 0; i < n; ++i)
                {
                    nodes.push_back(Hidden_Node(numNodesNextLayer));
                }
            }
        };
        class Output_Node
        {
        public:
            double value;
            double bias;
            Output_Node()
            {
                value = 0;
                bias = 0;
            }
        };
        class Output_Layer
        {
        public:
            vector<Output_Node> nodes;

            Output_Layer() {};
            Output_Layer(int n)
            {
                for (int i = 0; i < n; ++i)
                {
                    nodes.push_back(Output_Node());
                }
            }
        };
        
        // Data

        struct TrainingData
        {
            vector<vector<double>> inputs;
            vector<vector<double>> targets;
        };
        struct metadata {
            string model_version;
            string author;
            string creation_timestamp;
            string timestamp;
        };

        metadata meta;
        string model_name;
        
        TrainingData data;

        vector<int> sizes;
        int input_nodes;
        vector<int> hidden_layers;
        int output_nodes;
        string activation_func;

        Input_Layer input;
        vector<Hidden_Layer> hidden;
        Output_Layer output;

        // Constructor

        Network(vector<int> list, string activationFunc = "sigmoid") : sizes(list)
        {
            activation_func = activationFunc;
            init();
        }
        Network(initializer_list<int> list, string activationFunc = "sigmoid") : sizes(list)
        {
            activation_func = activationFunc;
            init();
        }
        Network(pair<vector<int>, string> sizes) : sizes(sizes.first)
        {
            activation_func = sizes.second;
            init();
        }
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
            input = Input_Layer(input_nodes, numLayersAfterInput);

            for (int i = 1; i < sizes.size() - 1; ++i)
            {
                hidden_layers.push_back(sizes[i]);
                hidden.push_back(Hidden_Layer(sizes[i], sizes[i + 1]));
            }
            output = Output_Layer(output_nodes);
        }

        // Save/Load

        void load_modal(string filename){

            cout << "Loading model from " << filename << "..." <<endl;

            ifstream file(filename);

            if (!file.is_open()) {
                cerr << "Error: Load file not found" << endl;
                return;
            }

            json config;
            file >> config;
            file.close();


            model_name = config["model_name"];

            meta.model_version = config["metadata"]["model_version"];
            meta.author = config["metadata"]["author"];
            meta.creation_timestamp = config["metadata"]["creation_timestamp"];
            meta.timestamp = config["metadata"]["timestamp"];



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
        
            cout << "Neural modal loaded successfully!" << endl;
        }
        void save_modal(string filename){

            cout << "Saving model to " << filename << "..." << endl;
            
            ofstream file(filename);

            if (!file.is_open()) {
                cerr << "Error opening file for reading." << endl;
                return;
            }

            json config;

            config["model_name"] = model_name;

            config["metadata"]["model_version"] = meta.model_version;
            config["metadata"]["author"] = meta.author;
            config["metadata"]["creation_timestamp"] = meta.creation_timestamp;
            config["metadata"]["timestamp"] = meta.timestamp;

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
        pair<vector<int>, string> get_sizes(string filename){
            
            ifstream file(filename);

            if (!file.is_open()) {
                cerr << "Error: Load file not found" << endl;
                return {{}, ""};
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

            return {saved_sizes, activation_func};
        }
        void loadData(int inNodes, int outNodes, string inputFile, string targetFile)
        {   

            cout << "Loading data from " << inputFile << " and " << targetFile << "..." << endl;
            ifstream DIin(inputFile);
            ifstream DTin(targetFile);

            if (!DIin.is_open() || !DTin.is_open())
            {
                cerr << "Error: Could not open " << inputFile << " or " << targetFile << endl;
                return;
            }

            double val;
            while (DIin >> val)
            {
                vector<double> row;
                row.push_back(val);
                for (int i = 1; i < inNodes; ++i)
                {
                    if (DIin >> val)
                        row.push_back(val);
                }
                if (row.size() == inNodes)
                {
                    data.inputs.push_back(row);
                }
            }

            while (DTin >> val)
            {
                vector<double> row;
                row.push_back(val);
                for (int i = 1; i < outNodes; ++i)
                {
                    if (DTin >> val)
                        row.push_back(val);
                }
                if (row.size() == outNodes)
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

        // Photos

        void addPhotoToTraining(string filename, double target){ // Need to remake with on my own
            ifstream file(filename, ios::binary);
            if (!file.is_open())
            {
                cerr << "Error: " << filename << " not found!" << endl;
                return;
            }
            string magic;
            file >> magic;
            file >> ws;
            while (file.peek() == '#')
            {
                string dummy;
                getline(file, dummy);
                file >> ws;
            }

            int w, h, maxVal;
            file >> w >> h >> maxVal;
            file.ignore();

            vector<double> pixels;

            if (magic == "P6")
            {
                for (int i = 0; i < w * h; ++i)
                {
                    unsigned char r, g, b;
                    file.read((char *)&r, 1);
                    file.read((char *)&g, 1);
                    file.read((char *)&b, 1);
                    double gray = (0.299 * r + 0.587 * g + 0.114 * b) / 255.0;
                    pixels.push_back(gray);
                }
            }
            else if (magic == "P5")
            {
                for (int i = 0; i < w * h; ++i)
                {
                    unsigned char pixel;
                    file.read((char *)&pixel, 1);
                    pixels.push_back((double)pixel / 255.0);
                }
            }
            else if (magic == "P2")
            {
                for (int i = 0; i < w * h; ++i)
                {
                    int val;
                    file >> val;
                    pixels.push_back((double)val / 255.0);
                }
            }
            else
            {
                cerr << "Format error: " << filename << " is " << magic << ". Not supported!" << endl;
                return;
            }

            if (pixels.size() == 784)
            {
                vector<double> targetVector(10, 0.0);
                if (target >= 0 && target < 10)
                    targetVector[(int)target] = 1.0;
                data.inputs.push_back(pixels);
                data.targets.push_back(targetVector);
                cout << "Loaded " << filename << " as Grayscale (" << magic << ")" << endl;
            }
            else
            {
                cerr << "Error: " << filename << " size mismatch. Got " << pixels.size() << " values, expected 784." << endl;
            }
        }
        int reverseInt(int i) {
            unsigned char c1, c2, c3, c4;
            c1 = i & 255; c2 = (i >> 8) & 255; c3 = (i >> 16) & 255; c4 = (i >> 24) & 255;
            return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
        }
        void loadMnist(string image_path, string label_path, int max_samples = -1) {

            
            cout << "Loading MNIST dataset..." << endl;

            ifstream img_file(image_path, ios::binary);
            ifstream lbl_file(label_path, ios::binary);

            if (!img_file.is_open() || !lbl_file.is_open()) {
                cerr << "Error: Check your paths! Could not open MNIST raw binary files." << endl;
                return;
            }

            int magic, num_items, rows, cols, magic_lbl, num_lbls;
            
            img_file.read((char*)&magic, 4);
            img_file.read((char*)&num_items, 4);
            img_file.read((char*)&rows, 4);
            img_file.read((char*)&cols, 4);
            
            lbl_file.read((char*)&magic_lbl, 4);
            lbl_file.read((char*)&num_lbls, 4);

            num_items = reverseInt(num_items);
            rows = reverseInt(rows);
            cols = reverseInt(cols);
            num_lbls = reverseInt(num_lbls);

            if( max_samples != -1 && num_items > max_samples ) {
                num_items = max_samples;
            }

            for (int i = 0; i < num_items; ++i) {
                vector<double> pixels;
                for (int p = 0; p < rows * cols; ++p) {
                    unsigned char temp = 0;
                    img_file.read((char*)&temp, 1);
                    pixels.push_back((double)temp / 255.0);
                }
                data.inputs.push_back(pixels);

                unsigned char label = 0;
                lbl_file.read((char*)&label, 1);
                vector<double> target(10, 0.0);
                target[(int)label] = 1.0;
                data.targets.push_back(target);
            }

            cout<< "MNIST dataset loaded." << endl; 
        }
        void displayMnistImage(const vector<double> &pixels, int width = 28, int height = 28) {
            cout << "\nMNIST Image (ID-based):" << endl;
            if (pixels.size() != width * height) {
                cerr << "Error: Invalid pixel count for " << width << "x" << height << " image" << endl;
                return;
            }
            
            for (int y = 0; y < height; ++y) {
                for (int x = 0; x < width; ++x) {
                    double pixel = pixels[y * width + x];
                    if (pixel > 0.5) {
                        cout << "##";
                    } else {
                        cout << "  ";
                    }
                }
                cout << endl;
            }
        }
        void displayMnistImageById(int imageId) {
            if (imageId < 0 || imageId >= data.inputs.size()) {
                cerr << "Error: Image ID " << imageId << " out of range. Available: 0-" << (data.inputs.size() - 1) << endl;
                return;
            }
            cout << "Image ID: " << imageId << " | Label: ";
            
            for (int i = 0; i < data.targets[imageId].size(); ++i) {
                if (data.targets[imageId][i] > 0.5) {
                    cout << i << endl;
                    break;
                }
            }
            
            displayMnistImage(data.inputs[imageId]);
        }
        void MnistTest(string input_file, string target_file, int max_samples, int tests_number) {
            
            
            loadMnist(input_file, target_file, max_samples);
            cout<< "Training samples: " << (data.inputs.size() < max_samples ? data.inputs.size() : max_samples) << endl;
    
    
            cout << "Testing on " << tests_number << " samples..." << endl;
    
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
                vector<double> prediction = predictBiggest(data.inputs[idx]);
    
                int actualLabel = -1;
                for (int j = 0; j < data.targets[idx].size(); j++) {
                if (data.targets[idx][j] > 0.9) { 
                actualLabel = j;
                break;
                }
                }
    
                bool isCorrect = (prediction[0] == actualLabel);
                if (isCorrect) correctPredictions++;
    
                cout << setw(4) << (i + 1) 
                << ". Input " << setw(4) << idx 
                << ": Real: " << setw(2) << actualLabel 
                << " | AI: " << setw(2) << prediction[0] 
                << (isCorrect ? " [CORRECT]" : " [WRONG]") << endl;
    
                if (!isCorrect) {
                    displayMnistImage(data.inputs[idx]);
                }
            }
    
            double accuracy = (double)correctPredictions / tests_number * 100.0;
            cout << "\n-----------------------------" << endl;
            cout << "Total correct: " << correctPredictions << "/" << tests_number << endl;
            cout << "Accuracy: " << accuracy << "%" << endl;
            cout << "-----------------------------" << endl;
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

        void backpropagate(double learning_rate, int epochs, int displayInterval = 1) {
            for (int e = 0; e < epochs; ++e) {
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

                if (e % displayInterval == 0) {
                    cout << "Epoch " << e << " | MSE: " << (total_mse / (data.inputs.size() * output.nodes.size())) << endl;
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