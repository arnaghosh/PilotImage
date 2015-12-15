#include <iostream>
#include <cmath>
#include <vector>
#include <queue>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>

cv::Mat rgb2gray(cv::Mat img){
	cv::Mat res(img.rows,img.cols,CV_8UC1,cvScalarAll(0));
	for(int i=0;i<img.rows;i++){
		for(int j=0;j<img.cols;j++){
			res.at<uchar>(i,j)=(4*img.at<cv::Vec3b>(i,j)[0]+4*img.at<cv::Vec3b>(i,j)[1]+img.at<cv::Vec3b>(i,j)[2])/9;
		}
	}
	return res;
}

void removesmallblob(cv::Mat &binimg,int **visited,int count){
	int i,j;
	for(i=0;i<binimg.rows;i++)
			for(j=0;j<binimg.cols;j++)
				if(visited[i][j]==count){
					visited[i][j]=0;
					binimg.at<uchar>(i,j) = 0;
				}
				return;
}

void apply_sauvola(cv::Mat &img, cv::Mat &thresh_img, int w,float k){
	cv::Mat sauvola=img.clone();

	cv::Mat integrated, integrated_sq;

	cv::integral(img, integrated, integrated_sq, CV_64F);

	for(int i=1+w/2;i<img.rows-w/2-1;i++){
		for(int j=1+w/2;j<img.cols-w/2-1;j++){
				double mean_s=integrated.at<double>(i+w/2,j+w/2)+integrated.at<double>(i-1-w/2,j-1-w/2)-integrated.at<double>(i-1-w/2,j+w/2)-integrated.at<double>(i+w/2,j-1-w/2);		//sum calculated
				double mean=mean_s/(1.0*w*w);  //mean=sum/w*w
				//if(mean<0){mean+=(256);std::cout<<"\nSomething horrible\n";}        //if overflow
				double var_s=integrated_sq.at<double>(i+w/2,j+w/2)+integrated_sq.at<double>(i-1-w/2,j-1-w/2)-integrated_sq.at<double>(i-1-w/2,j+w/2)-integrated_sq.at<double>(i+w/2,j-1-w/2);	//sum of sq calculated	
				double var=var_s/(1.0*w*w); //var is avg of squares.
				//if(var<0){var+=(65536);std::cout<<"\nSomething disastrous\n";}  //if overflow.
				double dev=(var-mean*mean);     //std dev=var-mean^2
				double sq_var=sqrt(dev); 
				double temp_thresh=sq_var/128; //max std dev for grayscale 8U is 128.
				double thresh=mean*(1+k*(temp_thresh-1)); //sauvola formula.
				//std::cout<<(int)img.at<uchar>(i,j)<<" "<<var_s<<","<<var<<" "<<mean<<std::endl;
				//if(dev<0) std::cout<<mean_s<<","<<mean<<" "<<var_s<<","<<var<<"  "<<dev<<std::endl;
				if(i > thresh_img.rows || j > thresh_img.cols)exit(9);
				thresh_img.at<double>(i,j) = thresh;
				if((int)img.at<uchar>(i,j)>=(int)thresh)img.at<uchar>(i,j)=255;
				else img.at<uchar>(i,j)=0;
		}
	}
	//return sauvola;	
}

cv::Mat sauvola_binarize(cv::Mat img,int w,float k){
	cv::Mat sauvola=img.clone();

	cv::Mat integrated, integrated_sq;

	cv::integral(img, integrated, integrated_sq, CV_64F);

	for(int i=1+w/2;i<img.rows-w/2;i++){
		for(int j=1+w/2;j<img.cols-w/2;j++){
				double mean_s=integrated.at<double>(i+w/2,j+w/2)+integrated.at<double>(i-1-w/2,j-1-w/2)-integrated.at<double>(i-1-w/2,j+w/2)-integrated.at<double>(i+w/2,j-1-w/2);		//sum calculated
				double mean=mean_s/(1.0*w*w);  //mean=sum/w*w
				//if(mean<0){mean+=(256);std::cout<<"\nSomething horrible\n";}        //if overflow
				double var_s=integrated_sq.at<double>(i+w/2,j+w/2)+integrated_sq.at<double>(i-1-w/2,j-1-w/2)-integrated_sq.at<double>(i-1-w/2,j+w/2)-integrated_sq.at<double>(i+w/2,j-1-w/2);	//sum of sq calculated	
				double var=var_s/(1.0*w*w); //var is avg of squares.
				//if(var<0){var+=(65536);std::cout<<"\nSomething disastrous\n";}  //if overflow.
				double dev=(var-mean*mean);     //std dev=var-mean^2
				double sq_var=sqrt(dev); 
				double temp_thresh=sq_var/128; //max std dev for grayscale 8U is 128.
				double thresh=mean*(1+k*(temp_thresh-1)); //sauvola formula.
				//std::cout<<(int)img.at<uchar>(i,j)<<" "<<var_s<<","<<var<<" "<<mean<<std::endl;
				//if(dev<0) std::cout<<mean_s<<","<<mean<<" "<<var_s<<","<<var<<"  "<<dev<<std::endl;
				if((int)img.at<uchar>(i,j)>=(int)thresh)sauvola.at<uchar>(i,j)=255;
				else sauvola.at<uchar>(i,j)=0;
		}
	}
	return sauvola;	
}

cv::Mat slow_sauvola_binarize(cv::Mat img,int w,float k){
	cv::Mat sauvola=img.clone();
		//std::cout<<"hi"<<std::endl;
	for(int i=w/2;i<img.rows-w/2;i++){
		for(int j=w/2;j<img.cols-w/2;j++){
			long long int sum=0,sq_sum=0;
			for(int m=-w/2;m<=w/2;m++){
				for(int l=-w/2;l<=w/2;l++){
					int val=(int)img.at<uchar>(i+m,j+l);
					sum+=val;
					sq_sum+=(val*val);
				}
			}
			float mean=sum/(1.0*w*w);
			float var=sq_sum/(1.0*w*w);
			float dev=var-(mean*mean);
			float sq_var=sqrt(dev);
			float temp_thresh=sq_var/128; //max std dev for grayscale 8U is 128.
			float thresh=mean*(1+k*(temp_thresh-1)); //sauvola formula.
			if((int)img.at<uchar>(i,j)>=thresh)sauvola.at<uchar>(i,j)=255;
			else sauvola.at<uchar>(i,j)=0;
		}
	}
	return sauvola;
}

int main(int argc, char **argv){
	cv::Mat orig1=cv::imread(argv[1],1);
	cv::Mat orig=rgb2gray(orig1);
	cv::namedWindow("image",0);
	//cv::imshow("image",orig);
	//cv::waitKey(0);


	int window=51;
	int k=0.34*200.0;
	cv::createTrackbar( "Track c->adaptive", "image", &k, 200.0);
	cv::createTrackbar( "Track window_size", "image", &window, 200);
	// Also for OCR specific, k=0.4 and windows = 60 was found good // but acc to me, smaller k ~0.2 would be better
	/*do{
		cv::Mat sauvola=sauvola_binarize(orig,window,k/200.0);
		cv::imshow("image",sauvola);
		cv::imwrite(argv[2], sauvola);
		if(cv::waitKey(33)==27)break;
	}while(1);*/

	cv::Mat sauvola=sauvola_binarize(orig,window,k/200.0);
	cv::imwrite(argv[2], sauvola);
}
