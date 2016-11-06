#ifndef __NETWORK_H
#define __NETWORK_H

#include <iostream>
#include <stdio.h>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;
using namespace ml;

int networkInit(void);
int networkInit(const char *fileName);
int detectFaceFormImg(const char *imgFilename,Mat *faceRoi);
int networkTrain(std::vector<Mat> *posFaceVector,std::vector<Mat> *negFaceVector,const char *fileName);
int networkClassify(Mat *image);

#endif /* __NETWORK_H */
