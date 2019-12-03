#include <iostream>

#include <opencv/cv.h>

//#include <opencv/highgui.h>  必须包含hpp？？？？？
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

void method_1(cv::Mat img) //color
{
    //yellow 26~34
    int iLowH = 5;
    int iHighH = 55;

    int iLowS = 0;
    int iHighS = 255;

    int iLowV = 0;
    int iHighV = 255;

    cv::cvtColor(image,imghsv,COLOR_BGR2HSV);
    Mat imgThresholded;
    inRange(imghsv, Scalar(iLowH, iLowS, iLowV), Scalar(iHighH, iHighS, iHighV), imgThresholded); //颜色提取
    //imshow("Thresholded Image", imgThresholded);


    Mat colorim(image.size(), CV_8UC3);  //显示对应颜色提取之后的有颜色图
    for(int i=0;i<imgThresholded.rows;++i) {
        uchar *p = imgThresholded.ptr<uchar>(i); //获取第i行第一个像素的指针
        for (int j = 0; j < imgThresholded.cols; ++j) {
            if(imgThresholded.at<uchar>(i,j)==255)
            {
                colorim.at<Vec3b>(i,j)=ss.at<Vec3b>(i,j);
            }

        }
    }

    //imshow("colorim", colorim);

}


//定义滑动条初始值
int g_nThresholdValue=100;  //阈值初始值
int g_nThresholdType=0;     //阈值模式初始值
#define WINDOW_NAME "阈值处理"
Mat g_grayImage,g_dstImage;

void on_Threshold(int,void*)
{
    //进行阈值分割
    threshold(g_grayImage,g_dstImage,g_nThresholdValue,255,g_nThresholdType);
    //显示结果
    imshow(WINDOW_NAME,g_dstImage);
}




void method_2(cv::Mat g_srcImage)
{


    //转换为灰度图
    cvtColor(g_srcImage,g_grayImage,COLOR_RGB2GRAY);//更多转换方式参见官方文档
    //显示原图
    imshow("原图",g_srcImage);

    //大津阈值
    threshold(g_grayImage,g_dstImage,0,255,CV_THRESH_OTSU);
    //自适应阈值
    //adaptiveThreshold(g_grayImage,g_dstImage,255,0,0,7,9);
    imshow(WINDOW_NAME,g_dstImage);
    waitKey();

    //创建滑动条
    namedWindow(WINDOW_NAME);
    createTrackbar("模式",WINDOW_NAME,&g_nThresholdType,4,on_Threshold);
    createTrackbar("阈值",WINDOW_NAME,&g_nThresholdValue,255,on_Threshold);

    //使用回调函数显示图像
    on_Threshold(0,0);

    waitKey();

}






int main() {


    string pattern = "/home/dd/drone_com/822-train/822data/proimg/*.jpg";
    string writepath="/home/dd/drone_com/822-train/822data/slectimg/";
    vector<String> fn;
    glob(pattern, fn, false);

    //    namedWindow("Thresholded Image",0);
    //    namedWindow("Image",0);
    //    namedWindow("colorim",0);
    for(int i=0;i<fn.size();i++) {
        cv::Mat image;
        cv::Mat ss;
        image=imread(fn[i]);
        cout<<fn[i]<<endl;
        image.copyTo(ss);
        Mat imghsv;
        //yellow 26~34
        int iLowH = 5;
        int iHighH = 55;

        int iLowS = 0;
        int iHighS = 255;

        int iLowV = 0;
        int iHighV = 255;


        cv::cvtColor(image,imghsv,COLOR_BGR2HSV);
        Mat imgThresholded;
        inRange(imghsv, Scalar(iLowH, iLowS, iLowV), Scalar(iHighH, iHighS, iHighV), imgThresholded);

        //imshow("Thresholded Image", imgThresholded);
        //imshow("Image", image);

        Mat colorim(image.size(), CV_8UC3);
        for(int i=0;i<imgThresholded.rows;++i) {
            uchar *p = imgThresholded.ptr<uchar>(i); //获取第i行第一个像素的指针
            for (int j = 0; j < imgThresholded.cols; ++j) {
                if(imgThresholded.at<uchar>(i,j)==255)
                {
                    colorim.at<Vec3b>(i,j)=ss.at<Vec3b>(i,j);
                }

            }
        }

    //imshow("colorim", colorim);
    //cout << fn.at(0)<<endl;
    istringstream in(fn.at(i));
    string t;
    vector<string> v;
    while (getline(in, t, '/')) {
        v.push_back(t);
    }
    string pathcv=writepath+v[v.size()-1];

    imwrite(pathcv,colorim);


    //cvWaitKey(1000);


    for(int k=0;k<colorim.rows;++k) {
        uchar *p = colorim.ptr<uchar>(k); //获取第i行第一个像素的指针
        for (int f = 0; f < colorim.cols; ++f) {
            Vec3b pixel;
            pixel[0]=0;
            pixel[1]=0;
            pixel[2]=0;

            colorim.at<Vec3b>(k,f)=pixel;

        }
    }


    }

    return 0;
}




//
//    cv::Mat image=imread("/home/dd/drone_com/pic1/26.jpg");
//    cv::Mat ss;
//
//    image.copyTo(ss);
//    Mat imghsv;
//    //yellow 26~34
//    int iLowH = 16;
//    int iHighH = 44;
//
//    int iLowS = 0;
//    int iHighS = 255;
//
//    int iLowV = 0;
//    int iHighV = 255;
//
//
//    cv::cvtColor(image,imghsv,COLOR_BGR2HSV);
//    Mat imgThresholded;
//    inRange(imghsv, Scalar(iLowH, iLowS, iLowV), Scalar(iHighH, iHighS, iHighV), imgThresholded);
//
//    imshow("Thresholded Image", imgThresholded);
//    imshow("Image", image);
//
//
//    vector<vector<Point> > contours;
//    vector<Vec4i> hierarchy;
//    findContours( imgThresholded, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE );
//
//
//
//    vector<vector<Point> > contours_poly( contours.size() );
//    vector<Rect> boundRect( contours.size() );
//
//    for( int i = 0; i < contours.size(); i++ )
//    {
//        boundRect[i] = boundingRect(contours[i] );
//    }
//
//
//    Mat drawing = Mat::zeros( imgThresholded.size(), CV_8UC3 );
//    for( size_t i = 0; i< contours.size(); i++ )
//    {
//
//        //Scalar color = Scalar( rng.uniform(0, 256), rng.uniform(0,256), rng.uniform(0,256) );
//        drawContours( drawing, contours, (int)i, Scalar(0,255,255), 2, LINE_8, hierarchy, 0 );
//        rectangle( image, boundRect[i].tl(), boundRect[i].br(), Scalar(0,255,0), 2, 8, 0 );
//    }
//    imshow("binImage", drawing);
//
//    imshow("Thresholded Image", imgThresholded);
//    imshow("Image", image);
//
//
//    Mat colorim(image.size(), CV_8UC3);
//    for(int i=0;i<imgThresholded.rows;++i) {
//        uchar *p = imgThresholded.ptr<uchar>(i); //获取第i行第一个像素的指针
//        for (int j = 0; j < imgThresholded.cols; ++j) {
//            if(imgThresholded.at<uchar>(i,j)==255)
//            {
//                colorim.at<Vec3b>(i,j)=ss.at<Vec3b>(i,j);
//            }
//
//        }
//    }
//
//    imshow("colorim", colorim);
//
//
//
//    cvWaitKey(0);
//
//

