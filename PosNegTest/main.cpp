/*
 * main.cpp
 *
 *  Created on: Oct 16, 2015
 *      Author: vareto
 */

#include <stdio.h>
#include <iostream>
#include <dirent.h>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/ml/ml.hpp>

using namespace std;
using namespace cv;
using namespace cv::ml;

void vector2Mat(vector<float> &v, Mat &mat) {
    mat = Mat(1, (int)v.size(), CV_32F);
    memcpy(mat.data, v.data(), v.size() * sizeof (float));
}

Ptr<SVM> train(vector<Mat> &neg, vector<Mat> &pos) {
    HOGDescriptor hog;
    hog.winSize = Size(256, 256);
    
    Mat features;
    Mat labels((int)(neg.size() + pos.size()), 1, CV_32S);
    
    int l = 0;
    cout << "extracting features" << endl;
    
    //negative samples
    for (int i = 0; i < neg.size(); i++) {
        vector<float> inFeature;
        Mat outFeat;
        
        resize(neg[i], neg[i], Size2i(256, 256));
        hog.compute(neg[i], inFeature, Size(16, 16), Size(0, 0), vector<Point>());
        vector2Mat(inFeature, outFeat);
        
        if (i == 0)
            features = outFeat.clone();
        else
            vconcat(features, outFeat, features);
        
        labels.at<int>(l, 0) = -1.0;
        
        l++;
    }
    
    //positive samples
    for (int i = 0; i < pos.size(); i++) {
        vector<float> inFeature;
        Mat outFeat;
        
        resize(pos[i], pos[i], Size(256, 256));
        hog.compute(pos[i], inFeature, Size(16, 16), Size(0, 0), vector<Point>());
        vector2Mat(inFeature, outFeat);
        
        vconcat(features, outFeat, features);
        labels.at<int>(l, 0) = 1.0;
        
        l++;
    }
    cout << "extracted" << endl;
    
    cout << "training" << endl;
    Ptr<SVM> svm = SVM::create();
    svm->setKernel(SVM::LINEAR);
    svm->setC(0.5);
    svm->train(features, ROW_SAMPLE, labels);
    cout << "trained" << endl;
    
    return svm;
}

void predict(Ptr<SVM> &svm, string path) {
    HOGDescriptor hog;
    hog.winSize = Size(256, 256);
    Mat probe = imread(path);
    
    for (int i = 0; (i + 256) <= probe.rows - 1; i += 100)
        for (int j = 0; (j + 256) <= probe.cols - 1; j += 100) {
            Mat outFeature, output;
            Mat img(probe, Rect(j, i, 256, 256));
            
            vector<float> inFeature;
            resize(img, img, Size(256, 256));
            
            //hog.compute(img, inFeature, Size(16, 16), Size(0, 0), vector<Point>());
            hog.compute(img, inFeature);
            vector2Mat(inFeature, outFeature);
            svm->predict(outFeature, output, 0);
            
            if (output.at<float>(0, 0) == 1.0) {
                rectangle(probe, Rect(j, i, 256, 256), Scalar(0, 0, 255), 3);
            }
        }
    
    resize(probe, probe, Size(800, 600));
    imshow("face detected", probe);
    waitKey(0);
    cout << "done" << endl;
}

bool is_imagem(string nome) {
    if ((nome[nome.size() - 1] == 'g') && (nome[nome.size() - 2] == 'n')
        && (nome[nome.size() - 3] == 'p'))
        return true;
    else if ((nome[nome.size() - 1] == 'g') && (nome[nome.size() - 2] == 'p')
             && (nome[nome.size() - 3] == 'j'))
        return true;
    return false;
}

void lerDiretorio(string caminhoDir, vector<Mat> &arquivos) {
    DIR *dir;
    struct dirent *file;
    
    dir = opendir(caminhoDir.c_str());
    while ((file = readdir(dir)) != NULL) {
        if (is_imagem(file->d_name)) {
            arquivos.push_back(imread(caminhoDir + string(file->d_name)));
            cout << file->d_name << " : " << arquivos.back().rows << " " << arquivos.back().cols << endl;
        }
    }
    closedir(dir);
}

int main() {
    vector<Mat> positivo;
    vector<Mat> negativo;
    
    lerDiretorio("neg_examples/", negativo);
    lerDiretorio("pos_samples/", positivo);
    
    Ptr<SVM> svm = train(negativo, positivo);
    predict(svm, "test/003.jpg");
    
    return 0;
}

