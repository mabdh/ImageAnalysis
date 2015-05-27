//
//  Task2.cpp
//  OpenCVProject
//
//  Created by Muhammad Abduh on 12/05/15.
//  Copyright (c) 2015 Muhammad Abduh. All rights reserved.
//
//
//#include <iostream>
//#include <opencv2/opencv.hpp>
//#include <dirent.h>
//#include <random>
//#include <fstream>
//
//
//using namespace std;
//using namespace cv;
//
//vector<int> ninety;
//vector<int> ten;
//vector<string> name_files;
//int num_files = 2429;
//Mat_<float> Xtrain;
//int image_width = 19;
//int image_height = 19;
//char filename[20];
//
//
//
//void classify_images(void){
//	// classify the index of data, classified indices are stored in ninety and ten vector of int
//	random_device rd;       // ranom device
//	mt19937 gen(rd());      // random generator
//
//	vector<int> temp;
//
//	int ninety_percent = (int)(ceil(0.9 * num_files));  // 90% number of files
//	int ten_percent = (int)(0.1 * num_files);           // 10% number of files
//
//	for (int i = 0; i < num_files; i++) {               // initialize working array
//		temp.push_back(i);
//	}
//
//	// get 90% of files
//	uniform_int_distribution<> distribution(0, num_files - 1);     // generate uniform distribution
//	for (int i = 0; i < ninety_percent; i++) {
//		int random = distribution(gen);
//		while (temp.at(random) == -1) {
//			random = distribution(gen);
//		}
//		ninety.push_back(temp.at(random));
//		temp.at(random) = -1;
//	}
//
//	// get 10% (the rest)
//	for (int i = 0; i < num_files; i++) {
//		if (temp.at(i) != -1) {
//			ten.push_back(temp.at(i));
//		}
//	}
//
//	for (int i = 0; i < num_files; i++) {
//		printf("%d\n", temp.at(i));
//	}
//
//
//}
//
//void read_images(){
//	// read images and save to the matrix as one array
//	Xtrain = Mat_<int>::zeros((int)ninety.size(), (image_width*image_height));
//	string line;
//
//
//	for (int i = 0; i <3 ; i++) {//(int)ninety.size(); i++) {
//		sprintf_s(filename, "train/face%05d.pgm", ninety[i]);
//		Mat image_temp = imread(filename, CV_LOAD_IMAGE_GRAYSCALE);
//		//        imshow("",image_temp);
//		cout << "ninty " << ninety.size() << endl;
//		//        cout << "xtrain " << Xtrain.rows << " " << Xtrain.cols << endl;
//		//        cout << "im " << image_temp.rows << " " << image_temp.cols << endl;
//		for (int j = 0; j < image_height; j++) {
//			for (int k = 0; k < image_width; k++){
//				cout << i << " " << j*image_width + k << endl;
//				Xtrain[i][j*image_width + k] =image_temp.at<int>(j, k);
//			}
//		}
//	}
//
//}
//
//int main(int argc, char** argv) {
//
//	//readDir();
//	classify_images();
//	//read_images();
//	// read images and save to the matrix as one array
//	Xtrain = Mat_<uchar>::zeros(1000, (image_width*image_height));
//	string line;
//
//
//	for (int i = 0; i <1000;i++){//(int)ninety.size(); i++) {
//		sprintf_s(filename, "train/face%05d.pgm", ninety[i]);
//		Mat image_temp = imread(filename, CV_LOAD_IMAGE_GRAYSCALE);
//		//        imshow("",image_temp);
//		//cout << "ninty " << ninety.size() << endl;
//		//        cout << "xtrain " << Xtrain.rows << " " << Xtrain.cols << endl;
//		//        cout << "im " << image_temp.rows << " " << image_temp.cols << endl;
//		for (int j = 0; j < image_height; j++) {
//			for (int k = 0; k < image_width; k++){
//				//cout << i << " " << j*image_width + k << endl;
//				Xtrain[i][j*image_width + k] = image_temp.at<uchar>(j, k);
//			}
//		}
//	}
//

	//substract the mean to get Xcenter
	Mat row_mean;
	reduce(Xtrain, row_mean, 0, CV_REDUCE_AVG);
	cout << row_mean << endl;

	Mat tempMat;
	Mat Xcenter;

	for (int i = 0; i<Xtrain.rows; i++)
	{
		tempMat = (Xtrain.row(i) - row_mean.row(0));
		Xcenter.push_back(tempMat.row(0));
	}

	//compute Covariance matrix, eigenvalue and eigen factors
	Mat eValuesMat;
	Mat eVectorsMat;
	Mat C = Xtrain.t() * Xtrain / 1000;// (int)ninety.size();
	eigen(C, eValuesMat, eVectorsMat);
	cout << eValuesMat.t() << endl;

	//sort the eigen values in descending order
	Mat indice;
	Mat sortedeValues;
	cv::sortIdx(eValuesMat, indice, CV_SORT_EVERY_COLUMN + CV_SORT_DESCENDING);//get the sorted value indices
	cv::sort(eValuesMat, eValuesMat, CV_SORT_EVERY_COLUMN + CV_SORT_DESCENDING);//sort the eigenvalue
	cout << indice.t() << endl;

	float sum = cv::sum(eValuesMat)[0];
	float temp = 0;
	int index = 0;
	for (int i= 0; i < 361; i++){
		temp = temp + eValuesMat.at<float>(i,0);
		if (temp >= 0.99*sum)  break;
		//Here I got a super big eigen value on the first place so it break in the first round
		//so I changed it from 0.9 to 0.99
		else index++;
	}

	cout << "index = " << index << endl;

	//
	Mat V;
	for (int i = 0; i < index; i++){
		V.push_back(eVectorsMat.row(i));//the eigen() has already sorted the eigen values
		//Normally should use
		//V.push_back(eVectorsMat.row(indice.at<int>(i,1)));
	}
	






	return 0;
}