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
Mat Xtrain;
Mat Xtest;
vector<Mat_<double>> Xtrain_covar;
int image_width = 19;
int image_height = 19;
char filename[20];
Mat row_mean;

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
    uniform_int_distribution<> distribution(1, num_files-1);     // generate uniform distribution
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
}

void plotxy(double* xData, double* yData, int dataSize) {
    FILE *gnuplotPipe,*tempDataFile;
    char *tempDataFileName;
    double x,y;
    int i;
    tempDataFileName = "plot";
    gnuplotPipe = popen("/opt/local/bin/gnuplot","w");
    if (gnuplotPipe) {
        fprintf(gnuplotPipe,"set title \"Eigen values\"\n",tempDataFileName);
        fprintf(gnuplotPipe,"set yrange [0:20000]\n",tempDataFileName);
        fprintf(gnuplotPipe,"plot \"%s\" with points\n",tempDataFileName);
        fflush(gnuplotPipe);
        tempDataFile = fopen(tempDataFileName,"w");
        for (i=0; i <= dataSize; i++) {
            x = xData[i];
            y = yData[i];
            fprintf(tempDataFile,"%lf %lf\n",x,y);
        }
        fclose(tempDataFile);
        printf("press enter to continue...");
        getchar();
        remove(tempDataFileName);
        fprintf(gnuplotPipe,"exit \n");
    } else {
        printf("gnuplot not found...");
    }
}


void process_training_images(void){
    // read images and save to the matrix as one array
    Xtrain = Mat::zeros((int)ninety.size(),(image_width*image_height),CV_8U);
    string line;
    int column;
    Mat image_temp;
    Mat_<int> linear_mat = Mat_<int>::zeros(1, image_width*image_height);
    
    for (int i = 0; i < (int)ninety.size(); i++) {
        sprintf(filename, "train/face%05d.pgm",ninety[i]);
        image_temp = imread(filename, CV_LOAD_IMAGE_GRAYSCALE);

        for (int k = 0; k < image_height; k++) {
            for (int j = 0; j < image_width; j++){
                column = (int)(j+image_height*k);
                Xtrain.at<uint8_t>(i,column) = image_temp.at<uint8_t>(j,k);
            }
        }
    }
    
    // get mean matrix
    row_mean = Mat::zeros(1, image_width*image_height, CV_8U);
    reduce(Xtrain, row_mean, 0, CV_REDUCE_AVG, CV_8U);

    // obtain zero mean data
    Mat tempMat;
    Mat Xtraincenter;
    
    for (int i = 0; i<Xtrain.rows; i++)
    {
        tempMat = (Xtrain.row(i) - row_mean.row(0));
        Xtraincenter.push_back(tempMat.row(0));
    }
    
    //compute Covariance matrix, eigenvalue and eigen factors
    Mat eValuesMat;
    Mat eVectorsMat;
    Xtrain.convertTo(Xtrain, CV_64F);
    Mat C = Xtrain.t() * Xtrain / 1000;// (int)ninety.size();
    
    eigen(C, eValuesMat, eVectorsMat);
    cout << eValuesMat.t() << endl;

    //sort the eigen values in descending order
    Mat indice;
    Mat sortedeValues;
    cv::sortIdx(eValuesMat, indice, CV_SORT_EVERY_COLUMN + CV_SORT_DESCENDING);//get the sorted value indices
    cv::sort(eValuesMat, eValuesMat, CV_SORT_EVERY_COLUMN + CV_SORT_DESCENDING);//sort the eigenvalue
    cout << "indices " << indice.t() << endl;
    
    double sum = cv::sum(eValuesMat)[0];
    cout << "sum " << sum << endl;
    double temp = 0;
    int index = 0;
    for (int i= 0; i < 361; i++){
        temp = temp + eValuesMat.at<double>(i,0);
        cout << "t/s " << temp << " " << sum << " " << temp/sum << endl;
        if (temp >= 0.99*sum)  break;
        //Here I got a super big eigen value on the first place so it break in the first round
        //so I changed it from 0.9 to 0.99
        else index++;
    }
    
    cout << "index = " << index << endl;
    
    // visualize eigen vectors
    vector<Mat> imeigenvector;
    for (int i = 0; i < index; i++) {
        Mat temp_img = Mat::zeros(image_height,image_width,CV_32S);
        
        for (int yi = 0; yi < image_height; yi++) {
            for (int xi = 0; xi < image_width; xi++) {
                int index_int = (int)indice.at<int>(0,i);
                temp_img.at<int>(xi,yi) = eVectorsMat.at<int>(yi*image_width+xi, index_int);
            }
        }
        imeigenvector.push_back(temp_img);
    }
    
    for (int i = 0; i < imeigenvector.size(); i++) {
         namedWindow( "i", WINDOW_NORMAL );
        imshow("i", imeigenvector.at(i));
        waitKey();
    }
    
    // plot eigen values
    double idx[indice.rows*indice.cols];
    double y[indice.rows*indice.cols];
    cout << indice.rows << " " <<indice.cols <<endl;
    cout << eValuesMat.rows << " " <<eValuesMat.cols <<endl;
    for (int i = 0; i < indice.rows; i++) {
        for (int j = 0; j < indice.cols; j++) {
            idx[i*indice.cols+j] = indice.at<int>(j,i);
            y[i*indice.cols+j] = eValuesMat.at<double>(j,i);
        }
    }
    plotxy(idx, y, indice.rows*indice.cols);
}

void process_test_images(void){
    Xtrain = Mat::zeros((int)ten.size(),(image_width*image_height),CV_8U);
    string line;
    int column;
    Mat image_temp;
    Mat_<int> linear_mat = Mat_<int>::zeros(1, image_width*image_height);
    
    for (int i = 0; i < (int)ten.size(); i++) {
        sprintf(filename, "train/face%05d.pgm",ten[i]);
        image_temp = imread(filename, CV_LOAD_IMAGE_GRAYSCALE);
        
        for (int k = 0; k < image_height; k++) {
            for (int j = 0; j < image_width; j++){
                column = (int)(j+image_height*k);
                Xtest.at<uint8_t>(i,column) = image_temp.at<uint8_t>(j,k);
            }
        }
    }

    // obtain zero mean data
    Mat tempMat;
    Mat Xtestcenter;
    
    for (int i = 0; i<Xtest.rows; i++)
    {
        tempMat = (Xtest.row(i) - row_mean.row(0));
        Xtestcenter.push_back(tempMat.row(0));
    }
    
    //compute Covariance matrix, eigenvalue and eigen factors
    Mat eValuesMat;
    Mat eVectorsMat;
    Xtrain.convertTo(Xtrain, CV_64F);
    Mat C = Xtrain.t() * Xtrain / 1000;// (int)ninety.size();
}

int main( int argc, char** argv ) {
    
    readDir();
    classify_images();
    process_training_images();
    process_test_images();
    return 0;
}