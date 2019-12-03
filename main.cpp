#include <iostream>

#include <opencv/cv.h>

//#include <opencv/highgui.h>  必须包含hpp？？？？？
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

Mat method_HSV(cv::Mat img) //color
{
    //yellow 26~34
    //white 0-180 0-30 221-255
    int iLowH = 0;
    int iHighH = 180;

    int iLowS = 0;
    int iHighS = 30;

    int iLowV = 221;
    int iHighV = 255;

    Mat imgThresholded,imghsv;
    cv::cvtColor(img,imghsv,COLOR_BGR2HSV);

    inRange(imghsv, Scalar(iLowH, iLowS, iLowV), Scalar(iHighH, iHighS, iHighV), imgThresholded); //颜色提取
    imshow("Thresholded Image", imgThresholded);


    Mat colorim(img.size(), CV_8UC3);  //显示对应颜色提取之后的有颜色图
    for(int i=0;i<imgThresholded.rows;++i) {
        uchar *p = imgThresholded.ptr<uchar>(i); //获取第i行第一个像素的指针
        for (int j = 0; j < imgThresholded.cols; ++j) {
            if(imgThresholded.at<uchar>(i,j)==255)
            {
//                colorim.at<Vec3b>(i,j)=img.at<Vec3b>(i,j); //把原始图像中的颜色部分加在提取后的
                Vec3b pixel;
                pixel[0]=0;
                pixel[1]=255;
                pixel[2]=255;
                colorim.at<Vec3b>(i,j)=pixel; //把原始图像中的颜色部分加在提取后的
            }

        }
    }

//    imshow("colorim", colorim);
    return imgThresholded;
}


//定义滑动条初始值
int g_nThresholdValue=100;  //阈值初始值
#define WINDOW_NAME "阈值处理"
Mat g_grayImage,g_dstImage;

void on_Threshold(int,void*)
{
    //进行阈值分割
    threshold(g_grayImage,g_dstImage,g_nThresholdValue,255,0);
    //显示结果
    imshow(WINDOW_NAME,g_dstImage);
}


void method_2_adjust(cv::Mat g_srcImage)//阈值提取
{


    //转换为灰度图
    cvtColor(g_srcImage,g_grayImage,COLOR_BGR2GRAY);
    //显示原图
    imshow("原图",g_srcImage);

    //大津阈值
    threshold(g_grayImage,g_dstImage,0,255,CV_THRESH_OTSU);
    //自适应阈值
    //adaptiveThreshold(g_grayImage,g_dstImage,255,0,0,7,9);
    imshow(WINDOW_NAME,g_dstImage);

    //创建滑动条
    namedWindow(WINDOW_NAME);
    createTrackbar("阈值",WINDOW_NAME,&g_nThresholdValue,255,on_Threshold);

    //使用回调函数显示图像
    on_Threshold(0,0);


}




Mat  method_threshold(cv::Mat img)//阈值提取
{

    Mat dstImage;
//    //转换为灰度图
//    cvtColor(img,grayImage,COLOR_RGB2GRAY);

    //大津阈值
    threshold(img,dstImage,249,255,0);
    //自适应阈值
    //adaptiveThreshold(g_grayImage,g_dstImage,255,0,0,7,9);
    imshow("dstImage",dstImage);
    return dstImage;
}


void img_proces(Mat img)
{
    //中值滤波
    Mat MedianBlurImg;
    medianBlur(img, MedianBlurImg, 5);
    imshow("MedianBlurImg", MedianBlurImg);

    //双边滤波
    Mat BilateralFilterImg;
    bilateralFilter(img, BilateralFilterImg, 5, 2, 2);
    imshow("BilateralFilterImg", BilateralFilterImg);

    //方框滤波
    Mat out;
    boxFilter(img, out, -1, Size(3, 3));

    //均值滤波
    Mat out1;
    blur(img,out1,Size(7,7));

    //高斯模糊
    Mat GaussianBlurImg;
    namedWindow("GaussianBlurImg");
    GaussianBlur(img, GaussianBlurImg, Size(5, 5), 1, 1);
    imshow("GaussianBlurImg", GaussianBlurImg);

}



RNG rng(123456);
void method_circle(Mat ori)  //慢，不准，需要调节阈值
{
    //霍夫变换
    /*
        HoughCircles函数的原型为：
        void HoughCircles(InputArray image,OutputArray circles, int method, double dp, double minDist, double param1=100, double param2=100, int minRadius=0,int maxRadius=0 )
        image为输入图像，要求是灰度图像
        circles为输出圆向量，每个向量包括三个浮点型的元素——圆心横坐标，圆心纵坐标和圆半径
        method为使用霍夫变换圆检测的算法，Opencv2.4.9只实现了2-1霍夫变换，它的参数是CV_HOUGH_GRADIENT
        dp为第一阶段所使用的霍夫空间的分辨率，dp=1时表示霍夫空间与输入图像空间的大小一致，dp=2时霍夫空间是输入图像空间的一半，以此类推
        minDist为圆心之间的最小距离，如果检测到的两个圆心之间距离小于该值，则认为它们是同一个圆心
        param1、param2为阈值
        minRadius和maxRadius为所检测到的圆半径的最小值和最大值
    */
    Mat cc;
    if (ori.channels() ==3)
        cvtColor(ori,cc,COLOR_BGR2GRAY);
    else ori.copyTo(cc);
    vector<Vec3f> cir;
    HoughCircles(cc, cir, CV_HOUGH_GRADIENT, 1, 50, 40, 40, 20, 150);

    Mat img;
    cvtColor(cc, img, COLOR_GRAY2RGB);

    for (size_t i = 0; i < cir.size(); i++) {
        Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
        circle(img, Point(cir[i][0], cir[i][1]), cir[i][2], color, 3, 8);
    }
    imshow("circle", img);
}

Mat method_canny(Mat img)
{
    Mat output;
    blur(img, output, Size(3, 3));
    Canny(output, output, 3, 9, 3);
    //sober

//    //求x方向梯度
//    Sobel(src, grad_x, CV_16S, 1, 0, 3, 1, 1, BORDER_DEFAULT);
//    convertScaleAbs(grad_x, abs_grad_x);
//    imshow("【效果图】X方向Sobel", abs_grad_x);
//
//    //求y方向梯度
//    Sobel(src, grad_y, CV_16S, 0, 1, 3, 1, 1, BORDER_DEFAULT);
//    convertScaleAbs(grad_y, abs_grad_y);
//    imshow("【效果图】Y方向Sobel", abs_grad_y);
//
//    //合并梯度（近似）
//    addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, dst);
//    imshow("【效果图】整体方向Sobel", dst);


    imshow("edge",output);
    return output;
}

Mat method_contours(Mat img,Mat &ori,Point &rect_center)
{
    std::vector<std::vector<Point>> contours;
    std::vector<Vec4i> hierarchy;

    findContours(img, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

    int index = 0;
    for (; index >= 0; index = hierarchy[index][0])
    {
        Scalar color(rand() & 255, rand() & 255, rand()&255);
        drawContours(img, contours, index, Scalar(255,255,255), 1, 8, hierarchy);
    }

//    findContours(img, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

    double maxArea = 0;
    std::vector<cv::Point> maxContour;
    for (size_t i = 0; i < contours.size(); i++)
    {
        double area = cv::contourArea(contours[i]);
        if (area > maxArea)
        {
            maxArea = area;
            maxContour = contours[i]; //最大轮廓
        }
    }

    Rect rect = boundingRect(maxContour);
    rectangle(ori,rect,Scalar(0, 255, 255),2);
    Scalar color_center(0,255,255);
    circle(ori,{rect.x+rect.width/2,rect.y+rect.height/2},5,Scalar(0, 255, 0),2,8);

    imshow("contours",ori);

    rect_center = {rect.x+rect.width/2,rect.y+rect.height/2}; //矩形中心

    //重心   更准哦
    //m00：表示0阶矩 m01：表示1阶水平矩 m10：表示1阶垂直矩
    Moments mo = moments(Mat(maxContour));
    circle(ori, cv::Point(mo.m10 / mo.m00, mo.m01 / mo.m00), 5, cv::Scalar(0,0,255), 2);

    imshow("ori", ori);
    return img;
}

void method_process(Mat &img)
{
    Mat element = getStructuringElement(MORPH_RECT, Size(5, 5));
    erode(img,img,element);
    dilate(img,img,element);
    imshow("proce  ss",img);
}



Point method_find_ceter(Mat img) { //通过像素中心寻找矩形框
//    for (int i = 0; i < img.rows; ++i) { //注意先是行，再是列，即先访问y坐标，再访问x
//        uchar *p = img.ptr<uchar>(i); //获取第i行第一个像素的指针
//        for (int j = 0; j < img.cols; ++j) {
//            int dd = p[j];
////            int dd = img.at<uchar>(i,j);
//            if (dd>0)
//            cout<<i<<" "<<j<<" "<<dd<<endl;
//          }
//
//        }


    for (int j = 0; j < img.cols; ++j) {//每一列
        uchar *p = img.ptr<uchar>(j); //获取第j列第一个像素的指针
        for (int i = 0; i < img.rows; ++i) {
            int pix = p[i];
        }
    }


    //to be continue


    Point center;
    center.x = 0;
    center.y = 0;
    return center;


}


void set_cam(VideoCapture capture)
{
    //设置摄像头参数 不要随意修改
    //capture.set(CV_CAP_PROP_FRAME_WIDTH, 1080);//宽度
    //capture.set(CV_CAP_PROP_FRAME_HEIGHT, 960);//高度
    //capture.set(CV_CAP_PROP_FPS, 30);//帧数
    //capture.set(CV_CAP_PROP_BRIGHTNESS, 1);//亮度 1
    //capture.set(CV_CAP_PROP_CONTRAST,40);//对比度 40
    //capture.set(CV_CAP_PROP_SATURATION, 50);//饱和度 50
    //capture.set(CV_CAP_PROP_HUE, 50);//色调 50
    capture.set(CV_CAP_PROP_EXPOSURE, 50);//曝光 50


//    CV_CAP_PROP_POS_MSEC - 影片目前位置，为毫秒数或者视频获取时间戳
//    CV_CAP_PROP_POS_FRAMES - 将被下一步解压/获取的帧索引，以0为起点
//    CV_CAP_PROP_POS_AVI_RATIO - 视频文件的相对位置（0 - 影片的开始，1 - 影片的结尾)
//    CV_CAP_PROP_FRAME_WIDTH - 视频流中的帧宽度
//    CV_CAP_PROP_FRAME_HEIGHT - 视频流中的帧高度
//    CV_CAP_PROP_FPS - 帧率
//    CV_CAP_PROP_FOURCC - 表示codec的四个字符
//    CV_CAP_PROP_FRAME_COUNT - 视频文件中帧的总数

}



int value = 0;
int max_value = 5;

Mat src1,src2;
void match_template(int, void*)
{
    Mat dst;
    int dst_col = src1.cols - src2.cols + 1;
    int dst_row = src1.rows - src2.rows + 1;
    //dst必须规定是 这个尺寸和类型
    dst.create(Size(dst_col,dst_row),CV_32FC1);

    //模板匹配
    matchTemplate(src1,src2,dst,value);
    //归一化
    normalize(dst,dst,0,1,NORM_MINMAX,-1,Mat());
    double minval, maxval;
    Point minloc, maxloc, loc;
    //找到模板匹配的最大值最小值和位置
    minMaxLoc(dst,&minval,&maxval,&minloc,&maxloc,Mat());
    if (value==CV_TM_SQDIFF||value==CV_TM_SQDIFF_NORMED)
    {
        loc = minloc;
    }
    else
    {
        loc = maxloc;
    }
    rectangle(dst,Rect(loc.x,loc.y,src2.cols,src2.rows),Scalar(0,0,255),2,LINE_AA);
    rectangle(src1,Rect(loc.x, loc.y, src2.cols, src2.rows), Scalar(255, 255, 255), 2, LINE_AA);
    imshow("output",dst);
    imshow("src", src1);


}



int main() {
//    cv::VideoCapture cam(0);
//    if (!cam.isOpened())
//    {
//        cout<<"cam open failed";
//        return 0;
//    }
//    while (cam.isOpened())
//    {
//        Mat frame;
//        cam >> frame;
//        method_1(frame);
    //}

    //**********************************one piece****************************
//    string path = "/home/dd/CLionProjects/drone/LED/WIN_20191111_21_20_31_Pro.jpg";
//    Mat img = imread(path);
//    if (!img.data)
//    {
//        cout<<"no pic read";
//        return 0;
//    }
//
//    imshow("ori",img);
//
//
//    method_process(img);
//
//    Mat imgThresholded = method_HSV(img);
//
//    Mat dstImage = method_threshold(imgThresholded);
//    imshow("dd",dstImage);
//
//    Point rect_center ;
//    Mat output_bin_img = method_contours(dstImage,img,rect_center);
//
//
//    Mat combine;
//    cvtColor(output_bin_img,combine,COLOR_GRAY2BGR);
//    hconcat(img,combine,combine);
//    namedWindow("combine",0);
//    imshow("combine",combine);

//**********************************two piece****************************
//
//    src1 = imread("/home/dd/CLionProjects/drone/LED/WIN_20191111_21_21_05_Pro.jpg");
//    //template
//    src2 = imread("/home/dd/CLionProjects/drone/temp.jpg");
//    namedWindow("src",CV_WINDOW_AUTOSIZE);
//    namedWindow("output",CV_WINDOW_AUTOSIZE);
//    createTrackbar("trackbar","output",&value,max_value,match_template);
//    match_template(0, 0);

//    waitKey();


    string pattern = "/home/dd/CLionProjects/drone/LED/*.jpg";
    string writepath="/home/dd/CLionProjects/drone/find/";
    vector<String> fn;
    glob(pattern, fn, false);

    //    namedWindow("Thresholded Image",0);
    //    namedWindow("Image",0);
    //    namedWindow("colorim",0);
    int jj = 0;
    for(int i=0;i<fn.size();i++) {
        Mat img = imread(fn[i]);

        if (!img.data)
        {
            cout<<"no pic read";
            return 0;
        }

        imshow("ori",img);


        method_process(img);

        Mat imgThresholded = method_HSV(img);

        Mat dstImage = method_threshold(imgThresholded);
        imshow("dd",dstImage);

        Point rect_center ;
        Mat output_bin_img = method_contours(dstImage,img,rect_center);


        Mat combine;
        cvtColor(output_bin_img,combine,COLOR_GRAY2BGR);
        hconcat(img,combine,combine);
        namedWindow("combine",0);
        imshow("combine",combine);
        string pathcv=writepath+to_string(jj)+".jpg";
        jj++;
        imwrite(pathcv,combine);

    }


    return 0;
}