#include <Windows.h>
#include <opencv2\highgui.hpp>
#include <opencv2\aruco.hpp>
#include <opencv2\aruco\dictionary.hpp>
#include <opencv2\aruco\charuco.hpp>
#include <opencv2\core.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <opencv2\opencv.hpp>
#include <vector>
#include <iostream>

using namespace std;
using namespace cv;

int main()
{

	cv::VideoCapture inputVideo;
	/*inputVideo.set(CV_CAP_PROP_FRAME_WIDTH, 1920);
	inputVideo.set(CV_CAP_PROP_FRAME_HEIGHT, 1080);*/

	VideoWriter writer("VideoTest.avi", CV_FOURCC('M', 'J', 'P', 'G'), 15, Size(640, 480));
	inputVideo.open(1);
	

	int WIDTH = inputVideo.get(CV_CAP_PROP_FRAME_WIDTH);
	int HEIGHT = inputVideo.get(CV_CAP_PROP_FRAME_HEIGHT);
	int fps = inputVideo.get(CV_CAP_PROP_FPS);

	//cout << "WIDTH = " << WIDTH << endl << "WIDTH =" << HEIGHT << endl << "fps = " << fps << endl;

	Mat image1;
	int cnt = 1;
	while(1)
	{
		inputVideo >> image1;//抓取视频中的一张照片
		cv::imshow("out", image1);
		//image.copyTo(imageCopy);
		if (image1.empty())
			continue;
		writer.write(image1);
		waitKey(1);

		/*int key = cv::waitKey(1000);*/

		//string path = "E:\\myvcwork\\11\\" + to_string(cnt) + ".jpg";
		//imwrite(path, image1);
		//cnt++;
		
	}
	destroyAllWindows();
	//inputVideo.set(CV_CAP_PROP_AUTO_EXPOSURE, -1);
	VideoCapture(1).release();
}