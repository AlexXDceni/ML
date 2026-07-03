#pragma once
#include <iostream>
#include "../neural_network.hpp"
#include "gui_video_util.hpp"


namespace GUI_FunctionsUtils
{

    using GeneratorFunc = std::function<std::vector<double>(double, double)>;

    void old_generation(NeuralNetwork& nn, ExternalScreen& screen, int RESOLUTION)
    {   
        // #pragma omp parallel for schedule(dynamic)
        for(int y = 0; y < RESOLUTION; y+=2){
            for(int x = 0; x < RESOLUTION; x += 2){

                double nx = (double)x / RESOLUTION * 2.0 - 1.0;
                double ny = (double)y / RESOLUTION * 2.0 - 1.0;

                //!  HERE IS THE FUNCTION DEFINITION 

                    double dist = std::sqrt(nx*nx + ny*ny);
                    double target_color = 0.0;
                    if(dist < 0.55){
                        target_color = 1.0;
                    }

                //! CHANGE TARGETS INPUT

                    nn.data.inputs.push_back({nx,ny});
                    nn.data.targets.push_back({target_color});

                //!
            }
        }

        int e = 0;
        double learning_rate = 0.02;
        
        while(true){


            nn.backpropagate(learning_rate,1);
            e++;

            // #pragma omp parallel for schedule(dynamic)
            for(int y = 0; y < RESOLUTION; y++){
                for(int x = 0; x < RESOLUTION; x++){

                    double nx = (double)x / RESOLUTION * 2.0 - 1.0;
                    double ny = (double)y / RESOLUTION * 2.0 - 1.0;

                    vector<double> prediction = nn.predict({nx,ny});
                    double value = prediction[0];

                    unsigned char r = static_cast<unsigned char>((1.0 - value)*255); 
                    unsigned char g = 0; 
                    unsigned char b = static_cast<unsigned char>(value * 255); 

                    screen.putPixel(x,y,r,g,b);
                }

                if(y%8==0){
                    screen.reloadScreen();
                }
            }
            
            screen.reloadScreen();
            cout<<"Epoch: "<< e << '\n';

        }
    }
    void classify_points(NeuralNetwork& nn, ExternalScreen& screen, int RESOLUTION, bool isEmptyDataSet = false)
            {
                cout << "========================================================\n";
                cout << " [LMB]  -> Add blue point (Type 1)\n";
                cout << " [RMB]  -> Add orange point (Type 0)\n";
                cout << " [MMB]  -> Delete hovered point\n";
                cout << "========================================================\n";

                struct Points {
                    double nx, ny; 
                    double type;  
                };
                vector<Points> points_data_set;

                if(!isEmptyDataSet){

                    mt19937 rng_puncte(42);
                    uniform_real_distribution<double> dist_random(-1.0, 1.0);

                    while (points_data_set.size() < 300) {

                        double px = dist_random(rng_puncte);
                        double py = dist_random(rng_puncte);
                        double dist = std::sqrt(px * px + py * py);


                        if (dist < 0.35) {
                            points_data_set.push_back({ px, py, 1.0 });
                        }
                        else if (dist > 0.55 && dist < 0.85) {
                            points_data_set.push_back({ px, py, 0.0 });
                        }   

                    }

                    for (const auto& p : points_data_set) {
                        nn.data.inputs.push_back({ p.nx, p.ny });
                        nn.data.targets.push_back({ p.type });
                    }

                }

                int e = 0;
                double learning_rate = 0.05;

                while (true) {
                    
                    if (!points_data_set.empty()) {
                        nn.backpropagate(learning_rate, 1, 0, 0); 
                    }
                    e++;

                    for (int y = 0; y < RESOLUTION; y += 2) { 
                        for (int x = 0; x < RESOLUTION; x += 2) {
                            double nx = (double)x / RESOLUTION * 2.0 - 1.0;
                            double ny = (double)y / RESOLUTION * 2.0 - 1.0;

                            vector<double> prediction = nn.predict({ nx, ny });
                            double val = prediction[0]; 

                            unsigned char r, g, b;
                            if (val > 0.5) {
                                double t = (val - 0.5) * 2.0; 
                                r = static_cast<unsigned char>(225 - t * 20);
                                g = static_cast<unsigned char>(235 - t * 15);
                                b = static_cast<unsigned char>(245 + t * 10);
                            } else {
                                double t = (0.5 - val) * 2.0; 
                                r = static_cast<unsigned char>(245 + t * 10);
                                g = static_cast<unsigned char>(235 - t * 15);
                                b = static_cast<unsigned char>(225 - t * 20);
                            }

                            screen.putPixel(x, y, r, g, b);
                            screen.putPixel(x + 1, y, r, g, b);
                            screen.putPixel(x, y + 1, r, g, b);
                            screen.putPixel(x + 1, y + 1, r, g, b);
                        }
                    }

                    for (const auto& p : points_data_set) {
                        int screen_x = static_cast<int>((p.nx + 1.0) * 0.5 * RESOLUTION);
                        int screen_y = static_cast<int>((p.ny + 1.0) * 0.5 * RESOLUTION);

                        unsigned char pr = (p.type == 1.0) ? 30  : 245;
                        unsigned char pg = (p.type == 1.0) ? 120 : 130;
                        unsigned char pb = (p.type == 1.0) ? 210 : 35;

                        for (int dy = -3; dy <= 3; dy++) {
                            for (int dx = -3; dx <= 3; dx++) {
                                if (screen_x + dx >= 0 && screen_x + dx < RESOLUTION && screen_y + dy >= 0 && screen_y + dy < RESOLUTION) {
                                    if (dx == -3 || dx == 3 || dy == -3 || dy == 3) {
                                        screen.putPixel(screen_x + dx, screen_y + dy, 255, 255, 255);
                                    } else {
                                        screen.putPixel(screen_x + dx, screen_y + dy, pr, pg, pb);
                                    }
                                }
                            }
                        }
                    }
                    screen.reloadScreen();
                    cout << "Epoch: " << e << "\n";



                    if (screen.new_click) {
                        screen.new_click = false;

                        double click_nx = (double)screen.last_click_x / RESOLUTION * 2.0 - 1.0;
                        double click_ny = (double)screen.last_click_y / RESOLUTION * 2.0 - 1.0;

                        if (screen.type_click == 1) { 
                            points_data_set.push_back({ click_nx, click_ny, 1.0 });
                            nn.data.inputs.push_back({ click_nx, click_ny });
                            nn.data.targets.push_back({ 1.0 });
                        } 
                        else if (screen.type_click == 2) { 
                            points_data_set.push_back({ click_nx, click_ny, 0.0 });
                            nn.data.inputs.push_back({ click_nx, click_ny });
                            nn.data.targets.push_back({ 0.0 });
                        } 
                        else if (screen.type_click == 3) { 
                            if (!points_data_set.empty()) {
                                int closest_index = 0;
                                double smallest_dist = 999.0;

                                for (size_t i = 0; i < points_data_set.size(); i++) {
                                    double dx = points_data_set[i].nx - click_nx;
                                    double dy = points_data_set[i].ny - click_ny;
                                    double d = std::sqrt(dx * dx + dy * dy);
                                    if (d < smallest_dist) {
                                        smallest_dist = d;
                                        closest_index = i;
                                    }
                                }
                                if (smallest_dist < 0.05) {
                                    points_data_set.erase(points_data_set.begin() + closest_index);
                                    nn.data.inputs.erase(nn.data.inputs.begin() + closest_index);
                                    nn.data.targets.erase(nn.data.targets.begin() + closest_index);
                                }
                            }
                        }
                    }
                }
            }
  
    void generate_and_learn_map(NeuralNetwork& nn, ExternalScreen& screen, int RESOLUTION, GeneratorFunc target_generator)
    {
        vector<vector<vector<double>>> target_map(RESOLUTION, vector<vector<double>>(RESOLUTION, vector<double>(3, 0.0)));

        #pragma omp parallel for
        for (int y = 0; y < RESOLUTION; y++) {
            for (int x = 0; x < RESOLUTION; x++) {
                double nx = (double)x / RESOLUTION * 3.0 - 1.5;
                double ny = (double)y / RESOLUTION * 3.0 - 1.5;

                std::vector<double> rgb = target_generator(nx, ny);

                target_map[y][x][0] = rgb[0]; // R
                target_map[y][x][1] = rgb[1]; // G
                target_map[y][x][2] = rgb[2]; // B
            }
        }

        int e = 0;
        double learning_rate = 0.02;
        mt19937 rng(1337);
        uniform_int_distribution<int> dist_pixel(0, RESOLUTION - 1);

        while (true) {

            nn.data.inputs.clear();
            nn.data.targets.clear();
            
            for (int i = 0; i < 8000; i++) {
                int rx = dist_pixel(rng);
                int ry = dist_pixel(rng);
                double nx = (double)rx / RESOLUTION * 3.0 - 1.5;
                double ny = (double)ry / RESOLUTION * 3.0 - 1.5;
                
                nn.data.inputs.push_back({ nx, ny });
                nn.data.targets.push_back({ target_map[ry][rx][0], target_map[ry][rx][1], target_map[ry][rx][2] });
            }

            nn.backpropagate(learning_rate, 1, 0, 0); 
            e++;

            #pragma omp parallel for schedule(dynamic)
            for (int y = 0; y < RESOLUTION; y += 2) { 
                for (int x = 0; x < RESOLUTION; x += 2) {
                    double nx = (double)x / RESOLUTION * 3.0 - 1.5;
                    double ny = (double)y / RESOLUTION * 3.0 - 1.5;
                    
                    vector<double> prediction = nn.predict({ nx, ny });
                    
                    unsigned char r = static_cast<unsigned char>(std::clamp(prediction[0], 0.0, 1.0) * 255);
                    unsigned char g = static_cast<unsigned char>(std::clamp(prediction[1], 0.0, 1.0) * 255);
                    unsigned char b = static_cast<unsigned char>(std::clamp(prediction[2], 0.0, 1.0) * 255);
                    
                    screen.putPixel(x, y, r, g, b);
                    screen.putPixel(x + 1, y, r, g, b);
                    screen.putPixel(x, y + 1, r, g, b);
                    screen.putPixel(x + 1, y + 1, r, g, b);
                }
            }

            screen.reloadScreen();
            cout << "Epoch: " << e << "\n";
        }
    }

    //? Functions from AI

    std::vector<double> wave_generator(double nx, double ny) {
        // Generează un pattern sinusoidal curat bazat pe coordonate
        double r = (std::sin(nx * 5.0) + 1.0) / 2.0;
        double g = (std::cos(ny * 5.0) + 1.0) / 2.0;
        double b = (std::sin((nx + ny) * 3.0) + 1.0) / 2.0;
        return { r, g, b };
    }
    std::vector<double> julia_fractal_generator(double nx, double ny) {
        const double c_real = -0.7;
        const double c_imag = 0.27015;
        const int MAX_ITERATION = 60;

        double z_real = nx;
        double z_imag = ny;
        int iter = 0;
        
        while (z_real * z_real + z_imag * z_imag < 4.0 && iter < MAX_ITERATION) {
            double temp = z_real * z_real - z_imag * z_imag + c_real;
            z_imag = 2.0 * z_real * z_imag + c_imag;
            z_real = temp;
            iter++;
        }

        double mu = (double)iter / MAX_ITERATION;
        return {
            std::sin(mu * 3.14159 / 2.0), // R
            mu * mu,                      // G
            std::sqrt(mu)                 // B
        };
    }

}