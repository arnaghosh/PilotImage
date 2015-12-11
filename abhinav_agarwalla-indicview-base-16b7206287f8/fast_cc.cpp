#include <iostream>
#include <stdio.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
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
int main()
{
	cv::Mat input = cv::imread("sudoku.jpg",0);
	if(input.empty())exit(1);

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
	cv::Mat newimg(input.size(),CV_8U, cv::Scalar(255));
	for(int i=1;i<input.rows;i++)
	{
		for(int j=1;j<input.cols;j++)
		{
			if(input.at<uchar>(i,j)==0)
			{
			if(input.at<uchar>(i,j-1)!=0 && input.at<uchar>(i-1,j)!=0)
				val[i][j]=cnt++;
			else if(input.at<uchar>(i,j-1)==0 && input.at<uchar>(i-1,j)!=0)
				val[i][j]=val[i][j-1];
			else if(input.at<uchar>(i,j-1)!=0 && input.at<uchar>(i-1,j)==0)
				val[i][j]=val[i-1][j];
			else
			{
				if(findset(val[i][j-1],par,rank)==findset(val[i-1][j],par,rank))
					val[i][j]=findset(val[i][j-1],par,rank);
				else
				{
					joinset(val[i][j-1],val[i-1][j],par,rank);
					val[i][j]=findset(val[i-1][j],par,rank);
				}
			}
		    }
		    if(val[i][j]!=0)
		    	newimg.at<uchar>(i,j)=0;
		    else
		    	newimg.at<uchar>(i,j)=255;
		}
	}
	cv::namedWindow("input",0);
	cv::imshow("input",input);
	cv::namedWindow("output",0);
	cv::imshow("output",newimg);
	cv::waitKey(0);

	return 0;
}
