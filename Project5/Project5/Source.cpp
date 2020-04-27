#include <opencv2/opencv.hpp>
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/features2d.hpp"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv_modules.hpp>
#include <opencv2/stitching.hpp>
#include <opencv2/stitching/detail/warpers.hpp>
#include "opencv2/highgui.hpp" 
#include "opencv2/core/utility.hpp"
#include<opencv2/xfeatures2d/include/opencv2/xfeatures2d/nonfree.hpp>


#include <iostream>
#include <fstream>
#include <string>

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;


Mat imagestitch(Mat leftImg, Mat rightImg)
{


	cv::Mat gray_rightImg;
	cv::Mat gray_leftImg;
	cvtColor(rightImg, gray_rightImg, COLOR_RGB2GRAY);
	cvtColor(leftImg, gray_leftImg, COLOR_RGB2GRAY);
	int points = 100000;
	

	// Variables to store keypoints and descriptors
	std::vector<KeyPoint> keypoints1, keypoints2;
	Mat descriptors1, descriptors2;

	// Detect ORB features and compute descriptors.
	Ptr<ORB> sift = ORB::create(points);
	sift->detectAndCompute(rightImg, Mat(), keypoints1, descriptors1);
	sift->detectAndCompute(leftImg, Mat(), keypoints2, descriptors2);


	std::vector<DMatch> matches;
	/*
	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
	matcher->match(descriptors1, descriptors2, matches, Mat());

	// Sort matches by score
	std::sort(matches.begin(), matches.end());

	// Remove not so good matches
	const int numGoodMatches = matches.size() * good_match_percent;
	matches.erase(matches.begin() + numGoodMatches, matches.end());
	*/

	//-- Matching descriptor vectors using FLANN matcher
	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::BRUTEFORCE_HAMMING);
	std::vector< std::vector<DMatch> > knn_matches;
	matcher->knnMatch(descriptors1, descriptors2, knn_matches, 2);
	//-- Filter matches using the Lowe's ratio test
	const float ratio_thresh = 0.70f;
	std::vector<DMatch> good_matches;
	for (size_t i = 0; i < knn_matches.size(); i++)
	{
		if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance)
		{
			good_matches.push_back(knn_matches[i][0]);
		}
	}


	// Draw top matches
	//Mat imMatches;
	//drawMatches(rightImg, keypoints1, leftImg, keypoints2, matches, imMatches);
	//drawMatches(rightImg, keypoints1, leftImg, keypoints2, matches, imMatches, Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	//namedWindow("Matches", WINDOW_NORMAL);
	//imshow("Matches", imMatches);
	
	//drawMatches(rightImg, keypoints1, leftImg, keypoints2, good_matches, imMatches, Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	//namedWindow("Good Matches", WINDOW_NORMAL);
	//imshow("Good Matches", imMatches);


	// Extract location of good matches
	std::vector<Point2f> points1, points2;

	for (size_t i = 0; i < good_matches.size(); i++)
	{
		points1.push_back(keypoints1[good_matches[i].queryIdx].pt);
		points2.push_back(keypoints2[good_matches[i].trainIdx].pt);
	}


	// Calculating homography used for aligning the images:
	cv::Mat ransac_mat;
	std::cout << "Computing homography..." << std::endl;

	cv::Mat homography = cv::findHomography(points1, points2, ransac_mat, cv::RANSAC, 9.0);


	float inlier = 0, outlier = 0;
	for (int i = 0; i < ransac_mat.rows; i++) {

		// We have an inlier:
		if ((int)ransac_mat.at<uchar>(i, 0) == 1) inlier = inlier + 1;

		// We have an outlier:
		else outlier = outlier + 1;
	}

	std::cout << "Total matches checked: " << ransac_mat.rows << std::endl;
	std::cout << "Inliers: " << inlier << std::endl;
	std::cout << "Outliers: " << outlier << std::endl;
	std::cout << "Procent inliers: " << (inlier / (ransac_mat.rows * 1.0)) * 100 << std::endl;
	std::cout << "Procent outliers: " << (outlier / (ransac_mat.rows * 1.0)) * 100 << std::endl;


	cv::Mat H = findHomography(points1, points2, RANSAC,3.0);
	



	//warp the on image using the estimated homography matrix and copy to image
	cv::Mat result;
	warpPerspective(rightImg, result, H, cv::Size(rightImg.cols + leftImg.cols*0.2, rightImg.rows));
	//namedWindow("Warp", WINDOW_NORMAL);
	//imshow("Warp", result);
	cv::Mat half(result, cv::Rect(0, 0, leftImg.cols, leftImg.rows));

	leftImg.copyTo(half);

	//namedWindow("Result", WINDOW_NORMAL);
	//imshow("Result", result);

	//namedWindow("Result", WINDOW_NORMAL);
	//imshow("Result", result);
	return result;
}


int main(int argc, char** argv)
{
	vector <Mat> imgs;
	vector<String> files;
	Mat result;
	//glob("C:/Users/egils/OneDrive/Pictures/skyline", files);
	glob("C:/Users/egils/OneDrive/Pictures/horizontal", files);
	//glob("C:/Users/egils/OneDrive/Pictures/keyboard", files);
	//glob("C:/Users/egils/OneDrive/Pictures/test", files);
	


	//result = imagestitch(left, right);

	
	for (int i = 0; i < files.size(); i++)
	{
		Mat img = imread(files[i]); // load the image
		if (img.empty())            // invalid image, skip it.
		{
			std::cout << files[i] << " is invalid!" << endl;
			continue;
		}

		imgs.push_back(img);

	}

	
	for (int j = imgs.size(); j >= 0; j--) {
		if (j == imgs.size() - 2)
		{
			cout << "Stitching images" << "\n";
			result = imagestitch(imgs[j], imgs[j+1]);
		}
		else if (j<imgs.size()-2)
		{
			cout << "Stitching image "<<j<<"......" << "\n";
			result = imagestitch(imgs[j], result);
		}
	}
	
	namedWindow("Result", WINDOW_NORMAL);
	imshow("Result", result);
	cout << "Done";


	waitKey(0);

}