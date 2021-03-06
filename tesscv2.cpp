// tesscv.cpp:
// Using Tesseract API with OpenCV

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <tesseract/baseapi.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <cmath>
#include <utility>
#include <vector>
#include <algorithm>
#include <queue>
#include <string>
#include <sstream>

cv::Mat erode(cv::Mat im,int kernel){
	cv::Mat res;
	cv::Mat element = cv::getStructuringElement(2,cv::Size(2*kernel+1,2*kernel+1),cv::Point(kernel,kernel));
	cv::erode(im,res,element);
	return res;
}

cv::Mat dilate(cv::Mat im,int kernel){
	cv::Mat res;
	cv::Mat element = cv::getStructuringElement(2,cv::Size(2*kernel+1,2*kernel+1),cv::Point(kernel,kernel));
	cv::dilate(im,res,element);
	return res;
}

void useTess(cv::Mat gray,std::string file_name){

	tesseract::TessBaseAPI tess;
	tess.Init(NULL, "hin", tesseract::OEM_DEFAULT);
	tess.SetPageSegMode(tesseract::PSM_SINGLE_BLOCK);
	tess.SetImage((uchar*)gray.data, gray.cols, gray.rows, 1, gray.cols);

	// Get the text
	char* out = tess.GetUTF8Text();
	std::cout << out << std::endl;

	std::ofstream file(file_name.c_str(), std::ofstream::app);
	file << out <<" ";
	file.close();
}

std::vector<std::pair<int,int> > arrange_words(std::vector<std::pair<int,int> > all_words,int maxBlobHeight){
	int size=all_words.size();
	std::vector<std::pair<int,int> > new_words(size);
	for(int i=0;i<size;i++){
		int count=0;
		for(int j=0;j<size;j++){
			if(abs(all_words[i].first-all_words[j].first)>maxBlobHeight){
				if(all_words[i].first>all_words[j].first){count++;continue;}
			}
			else {
				if(all_words[i].second>all_words[j].second){count++;continue;}
			}
		}
		new_words[count]=all_words[i];
	}
	return new_words;
}

cv::Mat restructure(cv::Mat im){
	cv::Mat res(im.rows+8,im.cols+8,CV_8UC1);
	for(int i=0;i<res.rows;i++){
		for(int j=0;j<res.cols;j++){
			res.at<uchar>(i,j)=255;
		}
	}
	for(int i=0;i<im.rows;i++){
		for(int j=0;j<im.cols;j++){
			res.at<uchar>(i+4,j+4)=im.at<uchar>(i,j);
		}
	}
	return res;
}

cv::Mat clearImage(cv::Mat original, int* valid,int** A){
	for(int i=0;i<original.rows;i++){
		for(int j=0;j<original.cols;j++){
			if(valid[A[i][j]]==0)original.at<uchar>(i,j)=255;
		}
	}
	return original;
} 

cv::Mat blob_joined(cv::Mat bin,cv::Mat original,int name,int* final_count,std::string file_name){
	cv::Point* lt,*rb,*center;
	int** A= new int*[bin.rows];
	for(int i=0;i<bin.rows;i++)A[i]=new int[bin.cols];
	for(int i=0;i<bin.rows;i++)
		for(int j=0;j<bin.cols;j++)
			A[i][j]=-1;
	int count=0;
	std::queue<cv::Point> q;
	for(int i=0;i<bin.rows;i++){
		for(int j=0;j<bin.cols;j++){
			if((int)bin.at<uchar>(i,j)==0 && A[i][j]==-1){
				q.push(cv::Point(i,j));
				while(q.size()!=0){
					cv::Point p=q.front();
					if(A[p.x][p.y]==-1){
						count++;
					}
					for(int k=p.x-1;k<=p.x+1;k++){
						for(int l=p.y-1;l<=p.y+1;l++){
							if((k>=0) && (l>=0) && (k<bin.rows) && (l<bin.cols) && ((int)bin.at<uchar>(k,l)==0) && (A[k][l]==-1)){								
								q.push(cv::Point(k,l));
								A[k][l]=0;
							}
						}
					}
					A[p.x][p.y]=count;
					q.pop();
				}
			}
		}
	}
	//bfs done -> count returns no of blos
	//std::cout<<"count="<<count<<std::endl;

	//lt -> left top, rb -> right bottom
	lt=new cv::Point[count+1];	
	rb=new cv::Point[count+1];
	center=new cv::Point[count+1];
	int* valid=new int[count+1];
	int* hist = new int[count+1];
	for(int a=0;a<=count;a++){
		lt[a]=cv::Point(bin.cols,bin.rows);
		rb[a]=cv::Point(0,0);
		hist[a]=0;
		valid[a]=1;
	}
	for(int i=0;i<bin.rows;i++){
		for(int j=0;j<bin.cols;j++){
			if(A[i][j]!=-1){
				hist[A[i][j]]++;
				if(i<lt[A[i][j]].y)lt[A[i][j]].y=i;
				if(j<lt[A[i][j]].x)lt[A[i][j]].x=j;
				if(i>rb[A[i][j]].y)rb[A[i][j]].y=i;
				if(j>rb[A[i][j]].x)rb[A[i][j]].x=j;		
			}		
		}
	}

	//max blob ht to be found
	int maxBlobHeight=0;
	for(int a=1;a<=count;a++){
		center[a].x=(lt[a].x+rb[a].x)/2;
		center[a].y=(lt[a].y+rb[a].y)/2;
		//if area of blob < im area/5000 , blob rejected
		if(hist[a] < (bin.rows*bin.cols)/5000) valid[a]=0;
		if(lt[a].x==0 || rb[a].x==bin.cols-1 || lt[a].y==0 || rb[a].y==bin.rows-1)valid[a]=0;
		if(valid[a] && ((rb[a].y-lt[a].y)>maxBlobHeight))maxBlobHeight=(rb[a].y-lt[a].y);
	}
	//remove invalid blobs
	original = clearImage(original,valid,A);
	
	//join nearby blobs for completing the word
	for(int a=1;a<=count;a++){
		for(int b=a+1;b<=count;b++){
			if(valid[a]==0 || valid[b]==0) continue;
			if((center[b].y-center[a].y)>maxBlobHeight)continue;
			if((center[b].y-center[a].y)<(rb[a].y-lt[a].y) && (center[b].x)<=(rb[a].x) && (center[b].x)>=(lt[a].x)){
				valid[b]=0;
				if(rb[b].x>rb[a].x)rb[a].x=rb[b].x;
				if(rb[b].y>rb[a].y)rb[a].y=rb[b].y;
				if(lt[b].x<lt[a].x)lt[a].x=lt[b].x;
				if(lt[b].y<lt[a].y)lt[a].y=lt[b].y;
				rb[b]=rb[a];
				lt[b]=lt[a];
				center[a].x=(lt[a].x+rb[a].x)/2;
				center[a].y=(lt[a].y+rb[a].y)/2;
			}
			else if((center[b].y-center[a].y)<(rb[b].y-lt[b].y) && (center[a].x)<=(rb[b].x) && (center[a].x)>=(lt[b].x)){
				valid[a]=0;
				if(rb[b].x<rb[a].x)rb[b].x=rb[a].x;
				if(rb[b].y<rb[a].y)rb[b].y=rb[a].y;
				if(lt[b].x>lt[a].x)lt[b].x=lt[a].x;
				if(lt[b].y>lt[a].y)lt[b].y=lt[a].y;
				lt[a]=lt[b];
				rb[a]=rb[b];
				center[b].x=(lt[b].x+rb[b].x)/2;
				center[b].y=(lt[b].y+rb[b].y)/2;
			}
		}
	}
	
	cv::Mat bound=bin.clone();
	cv::Mat temp;
	int fc=0;
	std::vector<std::pair<int,int> > all_words;
	for(int a=1;a<=count;a++){
		if(valid[a]){
			if((rb[a].y-lt[a].y)*(rb[a].x-lt[a].x) < 100) continue;
			std::pair<int,int> p;
			p.first=center[a].y;
			p.second=center[a].x;
			all_words.push_back(p);
		}
	}

	//arrange blobs acc to x,y
	all_words=arrange_words(all_words,maxBlobHeight);		

	//remove blobs at image border
	for(int i=0;i<all_words.size();i++){
		int a;
		for(int j=1;j<=count;j++){
			if((center[j].y==all_words[i].first) && (center[j].x==all_words[i].second)) a=j;	
		}

		cv::rectangle(bound,lt[a],rb[a],cv::Scalar(250),1);
		if(lt[a].x!=rb[a].x && lt[a].y!=rb[a].y){
			fc++;
			temp = original(cv::Rect(lt[a].x,lt[a].y,rb[a].x - lt[a].x,rb[a].y - lt[a].y));
			temp=restructure(temp);
			useTess(temp,file_name);   // Pass it to Tesseract API
		}
	}			
	*final_count=fc;
	return bound;
}

cv::Mat slow_sauvola_binarize(cv::Mat img,int w,float k){
	
	cv::Mat sauvola=img.clone();
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
	cv::Mat final_res=sauvola(cv::Rect(w/2,w/2,img.cols-w,img.rows-w));
	return final_res;	
}

double compute_skew(cv::Mat src){
   
   // Load in grayscale.
	cv::Size size = src.size();
	cv::bitwise_not(src, src);
	std::vector<cv::Vec4i> lines;
	//cv::HoughLinesP(src, lines, 1, CV_PI/180, 100, size.width / 2.f,20);
	cv::HoughLinesP(src, lines, 1, CV_PI/180, 100,size.width/5,20);
	cv::Mat disp_lines(size, CV_8UC1, cv::Scalar(0, 0, 0));
    double angle = 0.;
    unsigned nb_lines = lines.size();
    for (unsigned i = 0; i < nb_lines; ++i)
    {
        cv::line(disp_lines, cv::Point(lines[i][0], lines[i][1]),
                 cv::Point(lines[i][2], lines[i][3]), cv::Scalar(255, 0 ,0));
        double t_angle = atan2((double)lines[i][3] - lines[i][1],
                       (double)lines[i][2] - lines[i][0]);            
        angle+=t_angle;
    }
    angle /= nb_lines; // mean angle, in radians.
    return (angle*180/CV_PI);
 
}

cv::Mat deskew(cv::Mat img, double angle){

	cv::bitwise_not(img, img);

	std::vector<cv::Point> points;
	cv::Mat_<uchar>::iterator it = img.begin<uchar>();
	cv::Mat_<uchar>::iterator end = img.end<uchar>();
	for (; it != end; ++it)
	if (*it)
	  points.push_back(it.pos());

	cv::RotatedRect box = cv::minAreaRect(cv::Mat(points));
	cv::Mat rot_mat = cv::getRotationMatrix2D(box.center, angle, 1);

	cv::Mat top_left = cv::Mat(3,1,CV_64FC1);
	top_left.at<double>(0,0)=0;top_left.at<double>(1,0)=0;top_left.at<double>(2,0)=1;
	cv::Mat new_top_left = rot_mat*top_left;
	cv::Mat top_right = cv::Mat(3,1,CV_64FC1);
	top_right.at<double>(0,0)=0;top_right.at<double>(1,0)=img.cols;top_left.at<double>(2,0)=1;
	cv::Mat new_top_right = rot_mat*top_right;
	cv::Mat bottom_left = cv::Mat(3,1,CV_64FC1);
	bottom_left.at<double>(0,0)=img.rows;bottom_left.at<double>(1,0)=0;bottom_left.at<double>(2,0)=1;
	cv::Mat new_bottom_left = rot_mat*bottom_left;
	cv::Mat bottom_right = cv::Mat(3,1,CV_64FC1);
	bottom_right.at<double>(0,0)=img.rows;bottom_right.at<double>(1,0)=img.cols;bottom_right.at<double>(2,0)=1;
	cv::Mat new_bottom_right = rot_mat*bottom_right;

	cv::Point tl,br;
	if(new_top_left.at<double>(0,0)<new_top_right.at<double>(0,0))tl.y=new_top_right.at<double>(0,0);
	else tl.y=new_top_left.at<double>(0,0);
	if(new_top_left.at<double>(1,0)<new_bottom_left.at<double>(1,0))tl.x=new_bottom_left.at<double>(1,0);
	else tl.x=new_top_left.at<double>(1,0);
	if(new_bottom_left.at<double>(0,0)<new_bottom_right.at<double>(0,0))br.y=new_bottom_left.at<double>(0,0);
	else br.y=new_bottom_right.at<double>(0,0);
	if(new_bottom_right.at<double>(1,0)<new_top_right.at<double>(1,0))br.x=new_bottom_right.at<double>(1,0);
	else br.x=new_top_right.at<double>(1,0);


	cv::Mat rotated;
	cv::warpAffine(img, rotated, rot_mat, img.size(), cv::INTER_CUBIC);
	cv::Size box_size = box.size;
	if (box.angle < -45.)
	std::swap(box_size.width, box_size.height);
	cv::Mat cropped;
	cv::getRectSubPix(rotated, box_size, box.center, cropped);
	return cropped;
}

cv::Mat simpleThreshold(cv::Mat im){
	for(int i=0;i<im.rows;i++){
		for(int j=0;j<im.cols;j++){
			if(im.at<uchar>(i,j)!=255)im.at<uchar>(i,j)=0;
		}
	}
	return im;
}

int main(int argc, char** argv)
{
    // Usage: tesscv image.png
    std::cout << "Hello, you are inside tesscv";
    if (argc != 3)
    {
        std::cout << "Please specify the input image!" << std::endl;
        return -1;
    }

    // Load image
    cv::Mat im = cv::imread(argv[1],0);
    if (im.empty())
    {
        std::cout << "Cannot open source image!" << std::endl;
        return -1;
    }

    cv::Mat gray=im;
    
    // ...all pre-processing here...
	cv::Mat bin=slow_sauvola_binarize(gray,15,0.1);
	double angle =compute_skew(bin);

	cv::Mat image=deskew(bin,angle);
	image = simpleThreshold(image);
	 //cv::bitwise_not(image, image);
	image=dilate(image,1);
	//bin2=erode(bin2,2);
	cv::Mat bin2=image;
	int final_count=0;
	int i=0;
	cv::Mat bound=blob_joined(bin2,image,i,&final_count,argv[2]);
    return 0;
}
