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

int image_width = 19;
int image_height = 19;
char filename[20];
Mat row_mean;
Mat eValuesMat;
Mat eVectorsMat;
int k_num = 0;  // k index
double k_threshold = 0.95;  // threshold of k index

// This function will normalize value into 0 to 255
static Mat norm_0_255(InputArray _src) {
    Mat src = _src.getMat();
    // Create and return normalized image:
    Mat dst;
    switch(src.channels()) {
        case 1:
            cv::normalize(_src, dst, 0, 255, NORM_MINMAX, CV_8UC1);
            break;
        case 3:
            cv::normalize(_src, dst, 0, 255, NORM_MINMAX, CV_8UC3);
            break;
        default:
            src.copyTo(dst);
            break;
    }
    return dst;
}

// This function will count number of files in the directory
int readDir(void){
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

// This function is used to classify randomly 90% and 10% of images in directory
// the index of 90% images will be stored in vector<int> ninety
// the index of 10% images will be stored in vector<int> ten
void classify_images(void){
// classify the index of data, classified indices are stored in ninety and ten vector of int
    random_device rd;       // random device
    mt19937 gen(rd());      // random generator
    
    vector<int> temp;
    
    int ninety_percent = (int)(ceil(0.9 * num_files));  // 90% number of files
    int ten_percent = (int)(0.1 * num_files);           // 10% number of files
    
    for (int i = 0; i < num_files+1; i++) {               // initialize working array
        temp.push_back(i);
    }
    
    // get 90% of files
    uniform_int_distribution<> distribution(1, num_files);     // generate uniform distribution
    
    temp.at(0) = -1;
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

// This function is used to plot eigenvalues using GNUPLOT
void ploteigenvalues(double* xData, double* yData, int dataSize) {
    FILE *gnuplotPipe,*tempDataFile;
    char *tempDataFileName;
    double x,y;
    int i;
    tempDataFileName = "EigenValues";
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

// This function is used to plot distance stored in Matrix data using GNUPLOT
void plotdistance(Mat &data) {
    FILE *gnuplotPipe,*tempDataFile;
    char *tempDataFileName;
    double x;
    int i;
    tempDataFileName = "Distance";
    gnuplotPipe = popen("/opt/local/bin/gnuplot","w");
    if (gnuplotPipe) {
        fprintf(gnuplotPipe,"set title \"Distance\"\n",tempDataFileName);
        fprintf(gnuplotPipe,"set yrange [0:2000]\n",tempDataFileName);
        fprintf(gnuplotPipe,"set style line 1   lc rgb '#FF0000'\n",tempDataFileName);
        fprintf(gnuplotPipe,"set style line 2   lc rgb '#FF8000'\n",tempDataFileName);
        fprintf(gnuplotPipe,"set style line 3   lc rgb '#FFFF00'\n",tempDataFileName);
        fprintf(gnuplotPipe,"set style line 4   lc rgb '#80FF00'\n",tempDataFileName);
        fprintf(gnuplotPipe,"set style line 5   lc rgb '#0000FF'\n",tempDataFileName);
        fprintf(gnuplotPipe,"set style line 6   lc rgb '#7AFFFF'\n",tempDataFileName);
        fprintf(gnuplotPipe,"set style line 7   lc rgb '#FF00FF'\n",tempDataFileName);
        fprintf(gnuplotPipe,"set style line 8   lc rgb '#8000FF'\n",tempDataFileName);
        fprintf(gnuplotPipe,"set style line 9   lc rgb '#8A5C2E'\n",tempDataFileName);
        fprintf(gnuplotPipe,"set style line 10   lc rgb '#00000'\n",tempDataFileName);
        fprintf(gnuplotPipe,"plot \"%s\" using 1:2 with lines t \'image1\' ls 1, \"%s\" using 1:3 with lines t \'image2\' ls 2,\"%s\" using 1:4 with lines t \'image3\' ls 3, \"%s\" using 1:5 with lines t \'image4\' ls 4, \"%s\" using 1:6 with lines t \'image5\' ls 5, \"%s\" using 1:7 with lines t \'image6\' ls 6, \"%s\" using 1:8 with lines t \'image7\' ls 7, \"%s\" using 1:9 with lines t \'image8\' ls 8, \"%s\" using 1:10 with lines t \'image9\' ls 9,\"%s\" using 1:11 with lines t \'image10\' ls 10\n", tempDataFileName,tempDataFileName, tempDataFileName,tempDataFileName, tempDataFileName,tempDataFileName, tempDataFileName,tempDataFileName, tempDataFileName,tempDataFileName);
        fflush(gnuplotPipe);
        tempDataFile = fopen(tempDataFileName,"w");
        double y1;
        double y2;
        double y3;
        double y4;
        double y5;
        double y6;
        double y7;
        double y8;
        double y9;
        double y10;
        for (i=0; i < data.cols; i++) {
            x = i+1;
            y1 = data.at<double>(0,i);
            y2 = data.at<double>(1,i);
            y3 = data.at<double>(2,i);
            y4 = data.at<double>(3,i);
            y5 = data.at<double>(4,i);
            y6 = data.at<double>(5,i);
            y7 = data.at<double>(6,i);
            y8 = data.at<double>(7,i);
            y9 = data.at<double>(8,i);
            y10 = data.at<double>(9,i);
            fprintf(tempDataFile,"%lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf\n",x,y1,y2,y3,y4,y5,y6,y7,y8,y9,y10);
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

// Process the training images
// read images
// store to matrix as one dimensional array
// get mean matrix
// centerize matrix
// compute covariance, eigenvalues, and eigenvectors
// sort eigenvalues in descending order
// plot eigenvalues spectrum
// get k number of index as subspace index
// visualize eigenvectors
void process_training_images(void){
    // read images and save to the matrix as one array
    Xtrain = Mat::zeros((int)ninety.size(),(image_width*image_height),CV_8U);
    string line;
    int column;
    Mat image_temp;
    
    for (int i = 0; i < (int)ninety.size(); i++) {
        sprintf(filename, "train/face%05d.pgm",ninety[i]);
        image_temp = imread(filename, CV_LOAD_IMAGE_GRAYSCALE);

        image_height = image_temp.rows;
        image_width = image_temp.cols;
        for (int k = 0; k < image_height; k++) {
            for (int j = 0; j < image_width; j++){
                column = (int)(j+image_width*k);
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
    Xtraincenter.convertTo(Xtrain, CV_64F);
    Mat C = Xtrain.t() * Xtrain / 1000;// (int)ninety.size();
    
    eigen(C, eValuesMat, eVectorsMat);

    //sort the eigen values in descending order
    Mat indice;
    Mat sortedeValues;
    cv::sortIdx(eValuesMat, indice, CV_SORT_EVERY_COLUMN + CV_SORT_DESCENDING);//get the sorted value indices
    cv::sort(eValuesMat, eValuesMat, CV_SORT_EVERY_COLUMN + CV_SORT_DESCENDING);//sort the eigenvalue
    
    // plot spectrum of eigen values
    double idx[indice.rows*indice.cols];
    double y[indice.rows*indice.cols];
    for (int i = 0; i < indice.rows; i++) {
        for (int j = 0; j < indice.cols; j++) {
            idx[i*indice.cols+j] = indice.at<int>(j,i);
            y[i*indice.cols+j] = eValuesMat.at<double>(j,i);
        }
    }
    
    ploteigenvalues(idx, y, indice.rows*indice.cols);
    
    // determine the smallest eigen values
    double sum = cv::sum(eValuesMat)[0];
    double temp = 0;
    int index = 0;

    for (int i= 0; i < 361; i++){
        temp = temp + eValuesMat.at<double>(i,0);
        if (temp >= k_threshold*sum)  break;
        else index++;
    }
    
    cout << "index = " << index << endl;
    k_num = index; // number of ks
    
    // visualize eigen vectors
    vector<Mat> imeigenvector;
    for (int i = 0; i < k_num; i++) {
        Mat ev = eVectorsMat.row(i).clone();
        Mat temp_img = norm_0_255(ev.reshape(1, image_height));
        temp_img = temp_img.t();
        imeigenvector.push_back(temp_img);
    }
    
    Mat images;
    int rows_im = (int)sqrt(imeigenvector.size());
    int ind_rows_end = (int)(pow(rows_im, 2));
    int rows_rest = (int)(imeigenvector.size()) - ind_rows_end;//(int)(pow(rows_im, 2));
    Mat H = Mat::zeros(image_height, image_width, CV_8U);
    Mat V = Mat::zeros(image_height, image_width*(rows_im+1), CV_8U);
    
    for (int i = 0; i < rows_im; i++) {
        for (int j = 0; j < rows_im; j++) {
            hconcat(imeigenvector.at(i*rows_im+j), H, H);
        }
        vconcat(H, V, V);
        H = Mat::zeros(image_height, image_width, CV_8U);
    }
    if (rows_rest!=0) {
        for (int i = 0; i < rows_im-rows_rest; i++) {
            hconcat(H, H, H);
        }

        for (int i = 0; i < rows_rest; i++) {
            hconcat(imeigenvector.at(ind_rows_end+i), H, H);
        }
        H = H.colRange(0, V.cols);
        
        vconcat(H, V, V);
    }

    namedWindow( "Eigen vectors", WINDOW_NORMAL );
    imshow("Eigen vectors", V);
}

// read test images (10% images data)
// centerize test images with mean from training images
// randomly select 10 images
// calculate euclidean distance between training images and 10 test images
// sort the distances and plot
// get k eigenvectors
// project training data and 10 test images
// calculate euclidean distance between projected training images and projected 10 test images and plot the distances
// compare nearest neighbor between lower dimension and normal dimension of train images and 10 test images
void process_test_images(void){
    random_device rd;       // ranom device
    mt19937 gen(rd());      // random generator
    int num_test_images = 10;
    vector<int> test_images_idx;
    
    int ten_percent = (int)(0.1 * num_files);           // 10% number of files
    
    uniform_int_distribution<> distribution(1,ten_percent);     // generate uniform distribution
    for (int i = 0; i < num_test_images; i++) {
        int random = distribution(gen);
        test_images_idx.push_back(random);
    }
    
    Xtest = Mat::zeros((int)ten.size(),(image_width*image_height),CV_8U);
    string line;
    int column;
    Mat image_temp;

    for (int i = 0; i < (int)ten.size(); i++) {
        sprintf(filename, "train/face%05d.pgm",ten[i]);
        image_temp = imread(filename, CV_LOAD_IMAGE_GRAYSCALE);
        for (int k = 0; k < image_height; k++) {
            for (int j = 0; j < image_width; j++){
                column = (int)(j+image_width*k);
                Xtest.at<uint8_t>(i,column) = image_temp.at<uint8_t>(j,k);
            }
        }
    }

    // center test data
    Mat tempMat;
    Mat Xtestcenter;
    
    for (int i = 0; i<Xtest.rows; i++)
    {
        tempMat = (Xtest.row(i) - row_mean.row(0));
        Xtestcenter.push_back(tempMat.row(0));
    }
    
    // get 10 test images
    Mat test_images;
    for (int i = 0; i < num_test_images; i++) {
        test_images.push_back(Xtestcenter.row(test_images_idx.at(i)));
    }
    Mat Xtrainint;
    Xtrain.convertTo(Xtrainint, CV_8U);
    
    // calculate euclidean distance between training images and 10 test images
    Mat matdist;
    vector<double> temp;
    for (int i = 0; i < num_test_images; i++) {
        
        for (int j = 0; j < Xtrainint.rows; j++) {
            double dist = norm(test_images.row(i) , Xtrainint.row(j));
            temp.push_back(dist);
        }
        Mat tempmat = Mat(temp).reshape(1,1);
        temp.clear();
        matdist.push_back(tempmat.row(0));
    }
    
    // sort distance
    Mat sorted_dist;
    cv::sort(matdist, sorted_dist, CV_SORT_EVERY_ROW + CV_SORT_DESCENDING);//sort the distance

    plotdistance(sorted_dist);
    
    // k eigen vectors
    Mat kEigenVectors = eVectorsMat.rowRange(0, k_num-1);
    
    //projected train
    Mat projectedTrain = kEigenVectors * Xtrain.t();
    //projected ten images
    test_images.convertTo(test_images, CV_64F);
    Mat projectedTenIm = kEigenVectors * test_images.t();
    
    //10 images distance with lower dimensional space
    projectedTenIm.convertTo(projectedTenIm, CV_8U);
    projectedTrain.convertTo(projectedTrain, CV_8U);
    projectedTenIm = projectedTenIm.t();
    projectedTrain = projectedTrain.t();
    Mat matdist_k;
    vector<double> temp_k;
    for (int i = 0; i < num_test_images; i++) {
        
        for (int j = 0; j < projectedTrain.rows; j++) {
            double dist = norm(projectedTenIm.row(i) , projectedTrain.row(j));
            temp.push_back(dist);
        }
        Mat tempmat = Mat(temp).reshape(1,1);
        temp.clear();
        matdist_k.push_back(tempmat.row(0));
    }
    
    // sort distance
    Mat sorted_dist_k;
    cv::sort(matdist_k, sorted_dist_k, CV_SORT_EVERY_ROW + CV_SORT_DESCENDING);//sort the distance
    plotdistance(sorted_dist_k);
    
    //nearest neighbor
    vector<Point> lowDim;
    vector<Point> normalDim;
    for (int i = 0; i < num_test_images; i++) {
        Point minLoc, nul;
        minMaxLoc(matdist.row(i), NULL, NULL, &minLoc, NULL);
        lowDim.push_back(minLoc);
        minMaxLoc(matdist.row(i), NULL, NULL, &minLoc, NULL);
        normalDim.push_back(minLoc);
    }
    
    cout << "LowerDim" << " " << "NormalDim" << endl;
    for (int i = 0; i < num_test_images; i++) {
        cout << lowDim[i].x << " " <<normalDim[i].x << endl;
    }
    
}

int main( int argc, char** argv ) {
    
    readDir();
    classify_images();
    process_training_images();
    process_test_images();
    return 0;
}