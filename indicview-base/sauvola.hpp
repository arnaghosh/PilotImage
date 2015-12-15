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

int findset(int a,int par[],int rank[])
{
	while(par[a]!=a)
	{
		a=par[a];
	};
	return a;
}
void joinset(int a,int b,int par[],int rank[])
{
	int pa=findset(a,par,rank);
	int pb=findset(b,par,rank);
	if(rank[pa]>rank[pb])
		par[pb]=pa;
	else if(rank[pb]>rank[pa])
		par[pa]=pb;
	else
	{
		par[pa]=pb;
		par[a]=pb;   // makes the code faster
		rank[pb]++;
	}
}

cv::Mat thresh(cv::Mat img,int thresh){
	cv::Mat res(img.rows,img.cols,CV_8UC1,cvScalarAll(0));
	for (int i=0;i<img.rows;i++){
		for(int j=0;j<img.cols;j++){
			if(img.at<uchar>(i,j)>=thresh)res.at<uchar>(i,j)=255;
			else res.at<uchar>(i,j)=0;
		}}
	return res;
}

double maxArea(int w,int s,double q){
	if(s==1)return 0.7*w*w;
	if(s==8)return 10000; //see is effect
	return maxArea(w,s-1,q)*q*q;
}

double minArea(int w,int s,double q){
	if(s==1)return 0.0;
	return 0.9*maxArea(w,s-1,q)/(q*q);
}

cv::Mat fast_cc(cv::Mat input,int w,int s,int q)
{	
	//int w,s;
	//double q;
	//std::cin>>w>>s>>q;
	double max_area=maxArea(w,s,q),min_area=minArea(w,s,q);
	input=thresh(input,100);
	int tot = input.rows*input.cols;
	int par[tot],rank[tot];
	for(int i=0;i<tot;i++)
	{
		par[i]=i;
	    rank[i]=0;
	}
	int val[input.rows][input.cols];
	memset(val,0,sizeof(val));
	int cnt=1;
	std::vector<long long int> area;
	area.push_back(0);
	area.push_back(1);
	cv::Mat newimg(input.size(),CV_8U, cv::Scalar(255));
	for(int i=1;i<input.rows;i++)
	{
		for(int j=1;j<input.cols;j++)
		{
			if(input.at<uchar>(i,j)==0)
			{
			if(input.at<uchar>(i,j-1)!=0 && input.at<uchar>(i-1,j)!=0){
				val[i][j]=cnt++;
				area.push_back(1);
			}
			else if(input.at<uchar>(i,j-1)==0 && input.at<uchar>(i-1,j)!=0){
				val[i][j]=val[i][j-1];
				area[val[i][j]]++;
			}
			else if(input.at<uchar>(i,j-1)!=0 && input.at<uchar>(i-1,j)==0){
				val[i][j]=val[i-1][j];
				area[val[i][j]]++;
			}
			else
			{
				if(findset(val[i][j-1],par,rank)==findset(val[i-1][j],par,rank)){
					val[i][j]=findset(val[i][j-1],par,rank);
					area[val[i][j]]++;
				}
				else
				{
					joinset(val[i][j-1],val[i-1][j],par,rank);
					val[i][j]=findset(val[i-1][j],par,rank);
					area[val[i][j]]++;
				}
			}
		    }
		}
	}
	for(int i=0;i<input.rows;i++){
		for(int j=0;j<input.cols;j++){
		    if(val[i][j]!=0 && area[val[i][j]]<=max_area && area[val[i][j]]>=min_area)
		    	newimg.at<uchar>(i,j)=0;
		    else{
				val[i][j]=0;
		    	newimg.at<uchar>(i,j)=255;
			}
		}
	}
	/*cv::namedWindow("input",0);
	cv::imshow("input",input);
	cv::namedWindow("output",0);
	cv::imshow("output",newimg);
	cv::waitKey(0);*/

	return newimg;
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
cv::Mat sauvola_kx_binarize(cv::Mat img, int w, float k){

	cv::Mat image1 = img.clone();
	cv::Mat image2, image3, image4;

	cv::pyrDown(img, image2); // By Default is 2, if need to change, add a 3rd parameter
	cv::pyrDown(image2, image3);
	cv::pyrDown(image3, image4);

	cv::Mat thresh1(image1.size(), CV_32F, cv::Scalar(0));
	cv::Mat thresh2(image2.size(), CV_32F, cv::Scalar(0));
	cv::Mat thresh3(image3.size(), CV_32F, cv::Scalar(0));
	cv::Mat thresh4(image4.size(), CV_32F, cv::Scalar(0));

	apply_sauvola(image1, thresh1, w, k);
	apply_sauvola(image2, thresh2, w/2, k);
	apply_sauvola(image3, thresh3, w/4, k);
	apply_sauvola(image4, thresh4, w/8, k);
	
	cv::Mat s1,s2,s3,s4;
	cv::Mat s1_rescaled, s2_rescaled, s3_rescaled, s4_rescaled; 

	s1=fast_cc(thresh1, w, 1,2);
	s2=fast_cc(thresh2, w/2, 2,2);
	s3=fast_cc(thresh3, w/4, 4,2);
	s4=fast_cc(thresh4, w/8, 8,2);

	/*std::cout << std::endl << image1.size() << "\t" << w;
	std::cout << std::endl << image2.size() << "\t" << w/2;
	std::cout << std::endl << image3.size() << "\t" << w/4;
	std::cout << std::endl << image4.size() << "\t" << w/8;*/

	s1_rescaled = s1.clone();
	cv::resize(s2, s2_rescaled, s2_rescaled.size());
	cv::resize(s3, s3_rescaled, s3_rescaled.size());
	cv::resize(s4, s4_rescaled, s4_rescaled.size());

	cv::imshow("thresh",s1);
	cv::waitKey(0);

	cv::imshow("thresh",s2);
	cv::waitKey(0);

	cv::imshow("thresh",s3);
	cv::waitKey(0);

	cv::imshow("thresh",s4);
	cv::waitKey(0);

	cv::namedWindow("thresh",1);
	cv::imshow("thresh",thresh1);
	cv::waitKey(0);

	cv::imshow("thresh",thresh2);
	cv::waitKey(0);

	cv::imshow("thresh",thresh3);
	cv::waitKey(0);

	cv::imshow("thresh",image1);
	cv::waitKey(0);

	cv::imshow("thresh",image2);
	cv::waitKey(0);

	cv::imshow("thresh",image3);
	cv::waitKey(0);

	cv::imshow("thresh",image4);
	cv::waitKey(0);
	return img;
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

cv::Mat sauvola(cv::Mat input){
	if(input.empty())exit(1);

	int window=51;
	int k=0.34*200.0;
	// Also for OCR specific, k=0.4 and windows = 60 was found good // but acc to me, smaller k ~0.2 would be better
	cv::Mat sauvola=sauvola_binarize(input,window,k/200.0);
	cv::imshow("image",sauvola);	
	return sauvola;
}
