//
//  Task2.cpp
//  OpenCVProject
//
//  Created by Muhammad Abduh on 12/05/15.
//  Copyright (c) 2015 Muhammad Abduh. All rights reserved.
//

#include <iostream>
#include <opencv2/opencv.hpp>
#include <dirent.h>
#include <random>
#include <fstream>


using namespace std;
using namespace cv;

vector<int> ninety;
vector<int> ten;
vector<string> name_files;
int num_files;
Mat_<int> Xtrain;
int image_width = 19;
int image_height = 19;
char filename[20];

int readDir(void){ // initialization
    DIR *dir;
    struct dirent *ent;
    if ((dir = opendir ("train")) != NULL) {
        /* print all the files and directories within directory */
        // count file
        int count_file=0;
        while ((ent = readdir (dir)) != NULL) {
            if (ent->d_type == 8) { //pgm file
                name_files.push_back(ent->d_name);
                count_file++;
            }
        }
        num_files = count_file;
        closedir (dir);
        return EXIT_SUCCESS;
    } else {
        /* could not open directory */
        perror ("");
        return EXIT_FAILURE;
    }
}

void classify_images(void){
// classify the index of data, classified indices are stored in ninety and ten vector of int
    random_device rd;       // ranom device
    mt19937 gen(rd());      // random generator
    
    vector<int> temp;
    
    int ninety_percent = (int)(ceil(0.9 * num_files));  // 90% number of files
    int ten_percent = (int)(0.1 * num_files);           // 10% number of files
    
    for (int i = 0; i < num_files; i++) {               // initialize working array
        temp.push_back(i);
    }
    
    // get 90% of files
    uniform_int_distribution<> distribution(0, num_files-1);     // generate uniform distribution
    for (int i = 0; i < ninety_percent; i++) {
        int random = distribution(gen);
        while (temp.at(random)==-1) {
            random = distribution(gen);
        }
        ninety.push_back(temp.at(random));
        temp.at(random) = -1;
    }
    
    // get 10% (the rest)
    for (int i = 0; i < num_files; i++) {
        if (temp.at(i)!=-1) {
            ten.push_back(temp.at(i));
        }
    }

    for (int i = 0; i < num_files; i++) {
        printf("%d\n", temp.at(i));
    }
    
    
}

void read_images(){
// read images and save to the matrix as one array
    Xtrain = Mat_<int>::zeros((int)ninety.size(),(image_width*image_height));
    string line;

    
    for (int i = 0; i < (int)ninety.size(); i++) {
        sprintf(filename, "train/face%05d.pgm",ninety[i]);
        Mat image_temp = imread(filename, CV_LOAD_IMAGE_GRAYSCALE);
//        imshow("",image_temp);
        cout << "ninty " << ninety.size() << endl;
//        cout << "xtrain " << Xtrain.rows << " " << Xtrain.cols << endl;
//        cout << "im " << image_temp.rows << " " << image_temp.cols << endl;
        for (int j = 0; j < image_height; j++) {
            for (int k = 0; k < image_width; k++){
                cout << i << " " << j*image_width+k << endl;
                Xtrain[i][j*image_width+k] = image_temp.at<int>(j,k);
            }
        }
    }
    
}

int main( int argc, char** argv ) {
    
    readDir();
    classify_images();
    read_images();
    // Now read training images
    // Just need to open files with name stored in vector filename and index stored in vector ninety
    
    return 0;
}