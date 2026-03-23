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

//ifstream fout ("ml.out");

double BIAS = 0.01;
double learning_rate = 0.03;


double get_random(){

    static mt19937 gen(random_device{}());
    return uniform_real_distribution<double>{-1, 1}(gen);
}

// Activation functions

double RELU(double value){
    return (value > 0) ? value : 0;
}

double Leaky_RELU(double value) {
    return (value > 0) ? value : 0.01 * value;
}

double SIGMOID(double value) {
    return 1.0 / (1.0 + exp(-value));
}

double TANH(double value) {
    return tanh(value); 
}

double ELU(double value, double alpha = 1.0) {
    return (value > 0) ? value : alpha * (exp(value) - 1);
}




class Input_Node
{
public:
    double value;
    vector<double> weights;

    Input_Node(int numNodesNextLayer)
    {   
        value = 0 ;
        for (auto it = 0; it != numNodesNextLayer; ++it){
            weights.push_back( get_random() );
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

class Network
{
public:
    vector<int> sizes;

    int input_nodes;
    vector<int> hidden_layers;
    int output_nodes;

    Input_Layer input;
    vector<Hidden_Layer> hidden;
    Output_Layer output;

    // Initiate

    Network(vector<int> list) : sizes(list) {
        init(); 
    }

    Network(initializer_list<int> list) : sizes(list) {
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


    // Read

    vector<double> readInput(){

        ifstream fin (INPUT_FILE);
        vector<double> inputValues;
        double tmp;
        while(fin>>tmp){
            inputValues.push_back(tmp);
        }
        fin.close();

        if(inputValues.empty()) {
            cerr << "Error: Input values are empty." << endl;
            return {-1};
        }

        return inputValues;
    }

    vector<double> readTarget(){

        ifstream fin (TARGET_FILE);
        vector<double> targetValues;
        double tmp;
        while(fin>>tmp){
            targetValues.push_back(tmp);
        }
        fin.close();

        if(targetValues.empty()) {
            cerr << "Error: Target values are empty." << endl;
            return {-1};
        }

        return targetValues;
    }

    // Save/Load
    
    void saveModel(string filename) {
        ofstream fout(filename);
        if (!fout.is_open()) {
            cerr << "Error: Couldn't create save file." << endl;
            return;
        }

        for( auto e : sizes){
            fout<<e<<' ';
        }
        fout<<'\n';


        for (auto &node : input.nodes) {
            for (double w : node.weights) {
                fout << w << " ";
            }
            fout << "\n"; 
        }

        for (auto &layer : hidden) {
            for (auto &node : layer.nodes) {
                fout << node.bias << " "; 
                for (double w : node.weights) {
                    fout << w << " "; 
                }
                fout << "\n";
            }
        }

        for (auto &node : output.nodes) {
            fout << node.bias << " ";
        }
        fout << "\n";

        fout.close();
        cout << "Neural modal saved in: " << filename << "!" << endl;
}

    void loadModel(string filename) {
        ifstream fin_model(filename);
        if (!fin_model.is_open()) {
            cerr << "Error: Load file not found" << endl;
            return;
        }

        int temp_size;
        for(auto current_layer_size : sizes) {
            if (!(fin_model >> temp_size)) {
                cerr << "Error: Model file is corrupted or incomplete." << endl;
                return;
            }
            if (temp_size != current_layer_size) {
                cerr << "Error: Network layers mismatch." << endl;
                cerr << "Expected " << current_layer_size << " nodes, but found " << temp_size << " in file." << endl;
                return;
            }
        }

        for (auto &node : input.nodes) {
            for (double &w : node.weights) {
                fin_model >> w;
            }
        }

        for (auto &layer : hidden) {
            for (auto &node : layer.nodes) {
                fin_model >> node.bias;
                for (double &w : node.weights) {
                    fin_model >> w;
                }
            }
        }

        for (auto &node : output.nodes) {
            fin_model >> node.bias;
        }

        fin_model.close();
        cout << "Neural modal loaded from: " << filename << "!" << endl;
}

    // Display

    void printInfo() {
    cout << "Input Layer: " << input.nodes.size() << " nodes" << endl;
    for(int i = 0; i < hidden.size(); i++) {
        cout << "Hidden Layer " << i+1 << ": " << hidden[i].nodes.size() << " nodes" << endl;
    }
    cout << "Output Layer: " << output.nodes.size() << " nodes" << endl;
    }

    void printResults() {
        cout << "--- Output ---" << endl;
        for (int i = 0; i < output.nodes.size(); ++i) {
            cout << "Node " << i << ": " << output.nodes[i].value << endl;
        }
        cout << "------------------------" << endl;
}

    void displayFullNetwork(const vector<double>& targets) {
    cout << "\n================ Neural Network ================" << endl;


    cout << "[INPUT LAYER]" << endl;
    for (int i = 0; i < input.nodes.size(); ++i) {
        cout << "  Node " << i << " | Value: " << input.nodes[i].value << endl;
        cout << "    Weights to next layer: ";
        for (double w : input.nodes[i].weights) cout << "[" << w << "] ";
        cout << "\n" << endl;
    }

 
    for (int k = 0; k < hidden.size(); ++k) {
        cout << "[HIDDEN LAYER " << k + 1 << "]" << endl;
        for (int i = 0; i < hidden[k].nodes.size(); ++i) {
            cout << "  Node " << i << " | Value: " << hidden[k].nodes[i].value 
                 << " | Bias: " << hidden[k].nodes[i].bias << endl;
            
            cout << "    Weights to next: ";
            for (double w : hidden[k].nodes[i].weights) cout << "[" << w << "] ";
            cout << "\n" << endl;
        }
    }

    cout << "[OUTPUT LAYER]" << endl;
    for (int i = 0; i < output.nodes.size(); ++i) {
        cout << "  Node " << i << " | Value: " << output.nodes[i].value 
             << " | Bias: " << output.nodes[i].bias << endl;
    }

    cout <<"\n[LOSS]\n";
    cout<<"  "<<calculateCost(targets)<<'\n';
    cout << "====================================================" << endl;
}


    // Calculate

    double calculateCost(const vector<double>& targets) { 
        if (targets.size() != output.nodes.size()) {
            return -1; 
        }

        double totalError = 0;
        for (int i = 0; i < output.nodes.size(); ++i) {
            double error = targets[i] - output.nodes[i].value;
            totalError += error * error; 
        }

        return totalError / output.nodes.size();
}

    void feedZero() {

        for( auto &layer : hidden ){
            for( auto &node : layer.nodes ){
                node.value = 0;
            }
        }
        for( auto &node : output.nodes  ){
            node.value = 0;
        }
    }

    void feedForward(const vector<double>& inputValues){

        feedZero();

        if (inputValues.size() != input.nodes.size()) {
            cerr << "Error: Input size mismatch!" << endl;
            return;
        }   

        for (int i = 0; i < input.nodes.size(); ++i) {
            input.nodes[i].value = inputValues[i];
        }

        bool areHiddenLayers = (hidden.size() > 0) ? 1 : 0;


        // I-O 

        if(!areHiddenLayers) {

            for(int j = 0; j < output.nodes.size(); ++j) {
                for(int i = 0; i < input.nodes.size(); ++i) {
                    output.nodes[j].value += input.nodes[i].value * input.nodes[i].weights[j];
                } 
            output.nodes[j].value = SIGMOID(output.nodes[j].value + output.nodes[j].bias);
            }
        }
        
        else {
            

            // I - L1

            for(int j = 0; j < hidden[0].nodes.size(); ++j) {
                for(int i = 0; i < input.nodes.size(); ++i) {
                    hidden[0].nodes[j].value += input.nodes[i].value * input.nodes[i].weights[j];
                } 
            hidden[0].nodes[j].value = SIGMOID(hidden[0].nodes[j].value + hidden[0].nodes[j].bias);
            }
            
            
            // L1 - Ln

            for(int k = 0 ; k < hidden.size()-1; ++k )
                for(int j = 0; j < hidden[k+1].nodes.size(); ++j) {
                    for(int i = 0; i < hidden[k].nodes.size(); ++i) {
                        hidden[k+1].nodes[j].value += hidden[k].nodes[i].value * hidden[k].nodes[i].weights[j];
                    } 
                hidden[k+1].nodes[j].value = SIGMOID(hidden[k+1].nodes[j].value + hidden[k+1].nodes[j].bias);
                }
        
            
            // Ln - O

            for(int j = 0; j < output.nodes.size(); ++j) {
                for(int i = 0; i < hidden[hidden.size()-1].nodes.size(); ++i) {
                    output.nodes[j].value += hidden[hidden.size()-1].nodes[i].value * hidden[hidden.size()-1].nodes[i].weights[j];
                } 
            output.nodes[j].value = SIGMOID(output.nodes[j].value + output.nodes[j].bias);
            }
        
        }
    }


    // Learning
    
    void mutate(double rate){

        for( auto &node : input.nodes)
            for( auto &w : node.weights)
                w += get_random() * rate;

        for( auto &layer : hidden )
            for( auto &node : layer.nodes){
                node.bias += get_random() * rate;
                for( auto &w : node.weights )
                    w += get_random() * rate;
            }
        
        for( auto &node : output.nodes )
            node.bias += get_random() * rate;
    }

    void evolve(int numGen, double rate, const vector<double> targets, const vector<double> inputs){

        this->feedForward(inputs);
        double best_loss = this->calculateCost(targets);

        for( int i = 1; i <= numGen ; ++ i ) {
            
            Network child = *this;
            child.mutate(rate);
            child.feedForward(inputs);
            double new_loss = child.calculateCost(targets);

            if(new_loss < best_loss){
                *this = child;
                best_loss = new_loss;
                
                cout<<"Generation: " << i<< " | Loss: " << best_loss<<'\n';
            }
        }
    }
};

Network create_NN(){

    int input,hidden,x,output;
    vector<int> network;

    cout<<"Input nodes: ";
    cin>>input;
    cout<<'\n';
    network.push_back(input);

    cout<<"Hidden layer count: ";
    cin>>hidden;
    cout<<'\n';

    for(int i = 1; i<=hidden;i++){
        cout<<"Nodes in the layer "<< i <<": ";
        cin>>x;
        cout<<'\n';
        network.push_back(x);
    }

    cout<<"Output nodes: ";
    cin>>output;
    cout<<'\n';
    network.push_back(output);
    
    return Network( network );
}



int main()
{
    Network nn = Network({2,8,7,5,2});
    
    vector<double> inputs = nn.readInput();
    vector<double> targets = nn.readTarget();


    cout << "--- Before ---" << endl;
    nn.feedForward(inputs);
    nn.printResults();

    
    cout << "\nBetween..." << endl;
    nn.evolve(10000, 0.05, targets, inputs);


    cout << "\n--- After ---" << endl;
    nn.feedForward(inputs);
    nn.printResults();
    
    
    //nn.saveModel("trained_brain.txt");

    return 0;
}