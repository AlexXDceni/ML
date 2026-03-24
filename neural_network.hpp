#include <iostream>
#include <vector>
#include <initializer_list>
#include <fstream>
#include <random>
#include <cmath>
#include <string>
using namespace std;

string INPUT_FILE = "ml.in";
string TARGET_FILE = "targets.in";

double BIAS = 0.01;

double get_random()
{
    static mt19937 gen(random_device{}());
    return uniform_real_distribution<double>{-1, 1}(gen);
}

class Network
{
    public:
        // Activation functions

        double LINEAR(double value)
        {
            return value;
        }
        double RELU(double value)
        {
            return (value > 0) ? value : 0;
        }
        double Leaky_RELU(double value)
        {
            return (value > 0) ? value : 0.01 * value;
        }
        double SIGMOID(double value)
        {
            return 1.0 / (1.0 + exp(-value));
        }
        double TANH(double value)
        {
            return tanh(value);
        }
        double ELU(double value, double alpha = 1.0)
        {
            return (value > 0) ? value : alpha * (exp(value) - 1);
        }
        double ACTIVATION(double value, int activation_func)
        {

            switch (activation_func)
            {
            case 0:
                return SIGMOID(value);
                break;
            case 1:
                return RELU(value);
                break;
            case 2:
                return Leaky_RELU(value);
                break;
            case 3:
                return LINEAR(value);
                break;
            case 4:
                return TANH(value);
                break;
            case 5:
                return ELU(value);
                break;
            default:
                return SIGMOID(value);
                break;
            }
        }

        struct TrainingData
        {
            vector<vector<double>> inputs;
            vector<vector<double>> targets;
        };
        TrainingData loadData(int inNodes, int outNodes)
        {
            TrainingData data;
            ifstream DIin(INPUT_FILE);
            ifstream DTin(TARGET_FILE);

            if (!DIin.is_open() || !DTin.is_open())
            {
                cerr << "Error: Could not open " << INPUT_FILE << " or " << TARGET_FILE << endl;
                return {};
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
                return {};
            }

            if (data.inputs.size() != data.targets.size())
            {
                cerr << "Error: Mismatch! Inputs: " << data.inputs.size()
                    << " | Targets: " << data.targets.size() << endl;
                return {};
            }

            return data;
        }

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
                bias = BIAS;
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
                bias = BIAS;
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

        vector<int> sizes;

        int input_nodes;
        vector<int> hidden_layers;
        int output_nodes;

        int activation_func;

        Input_Layer input;
        vector<Hidden_Layer> hidden;
        Output_Layer output;

        // Initiate

        Network(vector<int> list, int activationFunc = 0) : sizes(list)
        {
            activation_func = activationFunc;
            init();
        }
        Network(initializer_list<int> list, int activationFunc = 0) : sizes(list)
        {
            activation_func = activationFunc;
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

        void saveModal(string filename)
        {
            ofstream fout(filename);
            if (!fout.is_open())
            {
                cerr << "Error: Couldn't create save file." << endl;
                return;
            }

            for (auto e : sizes)
            {
                fout << e << ' ';
            }
            fout << '\n';

            for (auto &node : input.nodes)
            {
                for (double w : node.weights)
                {
                    fout << w << " ";
                }
                fout << "\n";
            }

            for (auto &layer : hidden)
            {
                for (auto &node : layer.nodes)
                {
                    fout << node.bias << " ";
                    for (double w : node.weights)
                    {
                        fout << w << " ";
                    }
                    fout << "\n";
                }
            }

            for (auto &node : output.nodes)
            {
                fout << node.bias << " ";
            }
            fout << "\n";

            fout.close();
            cout << "Neural modal saved in: " << filename << "!" << endl;
        }
        void loadModal(string filename)
        {
            ifstream fin_model(filename);
            if (!fin_model.is_open())
            {
                cerr << "Error: Load file not found" << endl;
                return;
            }

            int temp_size;
            for (auto current_layer_size : sizes)
            {
                if (!(fin_model >> temp_size))
                {
                    cerr << "Error: Model file is corrupted or incomplete." << endl;
                    return;
                }
                if (temp_size != current_layer_size)
                {
                    cerr << "Error: Network layers mismatch." << endl;
                    cerr << "Expected " << current_layer_size << " nodes, but found " << temp_size << " in file." << endl;
                    return;
                }
            }

            for (auto &node : input.nodes)
            {
                for (double &w : node.weights)
                {
                    fin_model >> w;
                }
            }

            for (auto &layer : hidden)
            {
                for (auto &node : layer.nodes)
                {
                    fin_model >> node.bias;
                    for (double &w : node.weights)
                    {
                        fin_model >> w;
                    }
                }
            }

            for (auto &node : output.nodes)
            {
                fin_model >> node.bias;
            }

            fin_model.close();
            cout << "Neural modal loaded from: " << filename << "!" << endl;
        }

        void addPhotoToTraining(string filename, double target, TrainingData &data){ // Need to remake with on my on
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

        // Display

        void printInfo()
        {
            cout << "Input Layer: " << input.nodes.size() << " nodes" << endl;
            for (int i = 0; i < hidden.size(); i++)
            {
                cout << "Hidden Layer " << i + 1 << ": " << hidden[i].nodes.size() << " nodes" << endl;
            }
            cout << "Output Layer: " << output.nodes.size() << " nodes" << endl;
        }
        void printResults()
        {
            cout << "--- Output ---" << endl;
            for (int i = 0; i < output.nodes.size(); ++i)
            {
                double val = (output.nodes[i].value < 0.0001) ? 0 : output.nodes[i].value;
                cout << "Node " << i << ": " << val << endl;
            }
            cout << "------------------------" << endl;
        }
        void displayFullNetwork(const vector<double> &targets)
        {
            cout << "\n================ Neural Network ================" << endl;

            cout << "[INPUT LAYER]" << endl;
            for (int i = 0; i < input.nodes.size(); ++i)
            {
                cout << "  Node " << i << " | Value: " << input.nodes[i].value << endl;
                cout << "    Weights to next layer: ";
                for (double w : input.nodes[i].weights)
                    cout << "[" << w << "] ";
                cout << "\n"
                    << endl;
            }

            for (int k = 0; k < hidden.size(); ++k)
            {
                cout << "[HIDDEN LAYER " << k + 1 << "]" << endl;
                for (int i = 0; i < hidden[k].nodes.size(); ++i)
                {
                    cout << "  Node " << i << " | Value: " << hidden[k].nodes[i].value
                        << " | Bias: " << hidden[k].nodes[i].bias << endl;

                    cout << "    Weights to next: ";
                    for (double w : hidden[k].nodes[i].weights)
                        cout << "[" << w << "] ";
                    cout << "\n"
                        << endl;
                }
            }

            cout << "[OUTPUT LAYER]" << endl;
            for (int i = 0; i < output.nodes.size(); ++i)
            {
                cout << "  Node " << i << " | Value: " << output.nodes[i].value
                    << " | Bias: " << output.nodes[i].bias << endl;
            }

            cout << "====================================================" << endl;
        }

        // Calculate

        double calculateCost(const vector<double> &targets)
        {
            if (targets.size() != output.nodes.size())
            {
                return -1;
            }

            double totalError = 0;
            for (int i = 0; i < output.nodes.size(); ++i)
            {
                double error = targets[i] - output.nodes[i].value;
                totalError += error * error;
            }

            return totalError / output.nodes.size();
        }
        double getAverageLoss(const TrainingData &data)
        {
            double totalLoss = 0;
            for (int i = 0; i < data.inputs.size(); ++i)
            {
                this->feedForward(data.inputs[i]);
                totalLoss += this->calculateCost(data.targets[i]);
            }
            return totalLoss / data.inputs.size();
        }

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

        // Learning

        void mutate(double rate)
        {

            for (auto &node : input.nodes)
                for (auto &w : node.weights)
                    w += get_random() * rate;

            for (auto &layer : hidden)
                for (auto &node : layer.nodes)
                {
                    node.bias += get_random() * rate;
                    for (auto &w : node.weights)
                        w += get_random() * rate;
                }

            for (auto &node : output.nodes)
                node.bias += get_random() * rate;
        }
        void evolve(int generations, int populationSize, double mutationRate, const TrainingData &data)
        {
            double bestLoss = getAverageLoss(data);

            for (int g = 1; g <= generations; ++g)
            {
                Network bestCandidate = *this;
                bool foundBetter = false;

                for (int i = 0; i < populationSize; ++i)
                {
                    Network child = *this;
                    child.mutate(mutationRate);

                    double currentLoss = child.getAverageLoss(data);

                    if (currentLoss < bestLoss)
                    {
                        bestLoss = currentLoss;
                        bestCandidate = child;
                        foundBetter = true;
                    }
                }

                if (foundBetter)
                {
                    *this = bestCandidate;
                }

                if (g % 100 == 0)
                {
                    cout << "Epoch: " << g << " | Loss: " << bestLoss << endl;
                }
            }
        }

        void backpropagate(int learning_rate, const TrainingData &data)
        {

            double output_error = 0;

            // MSE

            for (int i = 0; i < output.nodes.size(); ++i)
            {

                double tmp = data.targets[0][i] - output.nodes[i].value;
                output_error += tmp * tmp;
            }
            output_error /= output.nodes.size();

            // Bin Cross Entropy

            // for( int i = 0 ;i < output.nodes.size(); ++i ){

            //     double actual_log =  output.nodes[i].value * log10( data.targets[0][1] );
            //     double target_log =  (1 - output.nodes[i].value ) * log10( 1 - data.targets[0][1] );
            //     output_error += actual_log + target_log;
            // }
            // output_error /= -output.nodes.size();
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
        double predictBiggest(const vector<double> &inputValues)
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
            return result;
        }
};
