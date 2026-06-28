#pragma once
#include <iostream>
#include <csignal>
#include <cstdlib>

#include "../neural_network.hpp"

namespace NeuralNetworkSignalHandler {

        inline NeuralNetwork* current_instance = nullptr;

        inline bool shouldSaveModelOnExit = false;

        inline void signalTrigger(int signum) 
        {
            cout << "[Ctrl+C] Signal received. Autosaving model..." << endl;

            if (shouldSaveModelOnExit && current_instance != nullptr) 
            {
                current_instance->save_model("../models/autosave_" + current_instance->meta.model_name + ".json");
            }

            exit(signum); 
        }
        inline void signalHandler(NeuralNetwork* instance, bool shouldSave = true) 
        {
            current_instance = instance;

            shouldSaveModelOnExit = shouldSave;
            
            signal(SIGINT, signalTrigger);
        }

}