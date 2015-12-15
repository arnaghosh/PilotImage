#include <iostream>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;

int main ( int argc, char **argv )
{
   Mat im_gray = imread(argv[1],0);
   Mat blur, binary;

   GaussianBlur(im_gray, blur, Size(5,5), 0, 0);
   threshold(blur, binary, 0,255, CV_THRESH_BINARY | CV_THRESH_OTSU);
   namedWindow("output");
   //imshow("output", binary);
   imwrite(argv[2], binary);
   //waitKey(0);
   return 0;
}  
