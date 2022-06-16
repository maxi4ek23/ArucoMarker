
#include<opencv2/core.hpp>
#include<opencv2/imgcodecs.hpp>
#include<opencv2/imgproc.hpp>
#include<opencv2/highgui.hpp>
#include<opencv2/aruco.hpp>
#include<opencv2/calib3d.hpp>
#include<iomanip>

#include<sstream>
#include <iostream>
#include <fstream>

#include<ctime>
//#include "windows.h"
//#include <thread> 

using namespace std;
using namespace cv;

const float calibrationSquareDimension = 0.0265f;
const float arucoSquareDimension = 0.0365f; //side length of the ArUco marker (in meters)
const Size chessboardDimension = Size(9, 6); //Dimension of the calibration board

//Function for creating markers
void createArucoMarkers()
{
	Mat outputMarker;

	Ptr<aruco::Dictionary> markerDictionery = aruco::getPredefinedDictionary(aruco::PREDEFINED_DICTIONARY_NAME::DICT_6X6_250);

	for (int i = 0; i < 250; i++)
	{
		aruco::drawMarker(markerDictionery, i, 500, outputMarker, 1);
		ostringstream convert;
		string imageName = "6x6Marker_";
		convert << imageName << i << ".jpg";
		imwrite(convert.str(), outputMarker);
	}
}

void createKnownBoardPosition(Size boardSize, float squareEdgelength, vector<Point3f>& corners)
{
	for (int i = 0; i < boardSize.height; i++)
	{
		for (int j = 0; j < boardSize.width; j++)
		{
			corners.push_back(Point3f(j * squareEdgelength, i * squareEdgelength, 0.0f));
		}
	}
}

void getChessBoardCorners(vector<Mat> images, vector<vector<Point2f>>& allFoundCorners, bool showResults = false)
{
	for (vector<Mat>::iterator iter = images.begin(); iter != images.end(); iter++)
	{
		vector<Point2f> pointBuf;
		bool found = findChessboardCorners(*iter, Size(9, 6), pointBuf, CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE);

		if (found)
		{
			allFoundCorners.push_back(pointBuf);
		}

		if (showResults)
		{
			drawChessboardCorners(*iter, Size(9, 6), pointBuf, found);
			imshow("Looking for corners", *iter);
			waitKey(0);
		}

	}

}

void cameraCalibration(vector<Mat> calibrationImages, Size boardSize, float squareEdgeLength, Mat& cameraMatrix, Mat& distanceCoefficients)
{
	vector<vector<Point2f>> checkerboardImageSpacePoints;
	getChessBoardCorners(calibrationImages, checkerboardImageSpacePoints, false);

	vector<vector<Point3f>> worldSpaseCornerPoints(1);

	createKnownBoardPosition(boardSize, squareEdgeLength, worldSpaseCornerPoints[0]);
	worldSpaseCornerPoints.resize(checkerboardImageSpacePoints.size(), worldSpaseCornerPoints[0]);

	vector<Mat> rVectors, tVectors;
	distanceCoefficients = Mat::zeros(8, 1, CV_64F);

	calibrateCamera(worldSpaseCornerPoints, checkerboardImageSpacePoints, boardSize, cameraMatrix, distanceCoefficients, rVectors, tVectors);

}

bool saveCameraCalibration(string name, Mat cameraMatrix, Mat distanceCoefficients)
{
	ofstream outStream(name);
	if (outStream)
	{
		uint16_t rows = cameraMatrix.rows;
		uint16_t columns = cameraMatrix.cols;

		outStream << rows << endl;
		outStream << columns << endl;

		for (int r = 0; r < rows; r++)
		{
			for (int c = 0; c < columns; c++)
			{
				double value = cameraMatrix.at<double>(r, c);
				outStream << value << endl;
			}
		}

		rows = distanceCoefficients.rows;
		columns = distanceCoefficients.cols;

		outStream << rows << endl;
		outStream << columns << endl;

		for (int r = 0; r < rows; r++)
		{
			for (int c = 0; c < columns; c++)
			{
				double value = distanceCoefficients.at<double>(r, c);
				outStream << value << endl;
			}
		}
		outStream.close();
		return true;
	}
	return false;
}

bool loadCameraCalibration(string name, Mat& cameraMatrix, Mat& distanceCoefficients)
{
	ifstream instream(name);
	if (instream)
	{
		uint16_t rows;
		uint16_t columns;

		instream >> rows;
		instream >> columns;

		cameraMatrix = Mat(Size(columns, rows), CV_64F);

		for (int r = 0; r < rows; r++)
		{
			for (int c = 0; c < columns; c++)
			{
				double read = 0.0f;
				instream >> read;
				cameraMatrix.at<double>(r, c) = read;
				std::cout << cameraMatrix.at<double>(r, c) << "\n";
			}
		}

		//Distance Coefficients
		instream >> rows;
		instream >> columns;

		distanceCoefficients = Mat::zeros(rows, columns, CV_64F);


		for (int r = 0; r < rows; r++)
		{
			for (int c = 0; c < columns; c++)
			{
				double read = 0.0f;
				instream >> read;
				distanceCoefficients.at<double>(r, c) = read;
				std::cout << distanceCoefficients.at<double>(r, c) << "\n";
			}
		}
		instream.close();
		return true;
	}
	return 0;
}

//Function for recognizing markers using the camera
int startWebcamMonitoring(const Mat& cameraMatrix, Mat& distanceCoefficients, float arucoSquareDimension)
{
	
	Mat frame;
	vector<int> markerIds;
	vector<vector<Point2f>> markerCorners, rejectedCandidates;
	aruco::DetectorParameters parameters;
	Ptr<aruco::Dictionary> markerDictionary = aruco::getPredefinedDictionary(aruco::PREDEFINED_DICTIONARY_NAME::DICT_6X6_250);
	VideoCapture vid(0);
	if (!vid.isOpened())
	{
		return -1;
	}
	namedWindow("Camera", WINDOW_AUTOSIZE);
	vector<Vec3d> rotationVectors, translationVectors;
	while (true)
	{
		if (!vid.read(frame))break;
		aruco::detectMarkers(frame, markerDictionary, markerCorners, markerIds); //marker recognition function
		aruco::estimatePoseSingleMarkers(markerCorners, arucoSquareDimension, cameraMatrix, distanceCoefficients, rotationVectors, translationVectors); //function for determining the positionand orientation of recognized markers relative to the camera
		for (int i = 0; i < markerIds.size(); i++)
		{
			aruco::drawDetectedMarkers(frame, markerCorners, markerIds); //contour of markers
			//aruco::drawAxis(frame, cameraMatrix, distanceCoefficients, rotationVectors[i], translationVectors[i], 0.05f);
		}
		imshow("Camera", frame);
		if (waitKey(30) >= 0) break;
	}
	return 1;
}



//Function for camera calibration
void cameraCalibrationProcess(Mat& cameraMatrix, Mat& distanceCoefficients)
{
	Mat frame;
	Mat drawToFrame;

	vector<Mat> savedImages;

	vector<vector<Point2f>> markerCorners, rejectedCanditates;

	VideoCapture vid(0);

	if (!vid.isOpened())
	{
		return;
	}

	int framePerSecond = 20;

	namedWindow("Калібровка камери", WINDOW_AUTOSIZE);

	while (true)
	{
		if (!vid.read(frame))
			break;

		vector<Vec2f> foundPoints;
		bool found = false;

		found = findChessboardCorners(frame, chessboardDimension, foundPoints, CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE);
		frame.copyTo(drawToFrame);
		drawChessboardCorners(drawToFrame, chessboardDimension, foundPoints, found);
		if (found)
		{
			imshow("Калібровка камери", drawToFrame);
		}
		else
		{
			imshow("Калібровка камери", frame);
		}
		char character = waitKey(1000 / framePerSecond);

		/*if (found)
		{
			Mat temp;
			frame.copyTo(temp);
			savedImages.push_back(temp);
			std::cout << savedImages.size() << endl;
			Sleep(500);
		}*/

		switch (character)
		{
		case' ':
			//saving image
			if (found)
			{
				Mat temp;
				frame.copyTo(temp);
				savedImages.push_back(temp);
				std::cout << savedImages.size() << endl;
			}
			break;

		case 13:
			//start calibration
			if (savedImages.size() > 10)
			{
				cameraCalibration(savedImages, chessboardDimension, calibrationSquareDimension, cameraMatrix, distanceCoefficients);
				saveCameraCalibration("IloveCameraCalibration", cameraMatrix, distanceCoefficients);
			}
			break;
		case 27:
			//exit
			return;
			break;
		}
	}


}


//Search function for a specific marker (Called in startWebcamMonitoringMod)
void markerNumberSearch(vector<int>& markerIds, const int& markerNumber, bool& markerDetected)
{
	for (int i = 0; i < markerIds.size(); i++)
	{
		if (markerIds[i] == markerNumber)
		{
			markerDetected = true;
			break;
		}
		else
			markerDetected = false;
	}
	if (markerIds.size() == 0)markerDetected = false;
}

//Function for displaying in the frame: marker position coordinates, marker number and if marker detected or not
void showInFrame(const Mat& frame, const Vec3d xyz, const int& markerNumber, const bool& markerDetected)
{
	ostringstream vector_to_marker;

	vector_to_marker.str(std::string());
	vector_to_marker << std::setprecision(4) << "X: " << std::setw(8) << xyz(0);
	cv::putText(frame, vector_to_marker.str(), Point(10, 25), cv::FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 0, 255), 2);

	vector_to_marker.str(std::string());
	vector_to_marker << std::setprecision(4) << "Y: " << std::setw(8) << xyz(1);
	cv::putText(frame, vector_to_marker.str(), Point(10, 50), cv::FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 252, 124), 2);

	vector_to_marker.str(std::string());
	vector_to_marker << std::setprecision(4) << "Z: " << std::setw(8) << xyz(2);
	cv::putText(frame, vector_to_marker.str(), Point(150, 25), cv::FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 0, 0), 2);

	vector_to_marker.str(std::string());
	vector_to_marker << std::setprecision(4) << "Marker" << std::setw(3) << markerNumber; //markerNumber
	cv::putText(frame, vector_to_marker.str(), Point(150, 50), cv::FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 255, 255), 2);

	string DetOrNot;
	markerDetected ? DetOrNot = "Detected" : DetOrNot = " ";

	vector_to_marker.str(std::string());
	vector_to_marker << std::setprecision(4) << DetOrNot;
	cv::putText(frame, vector_to_marker.str(), Point(260, 50), cv::FONT_HERSHEY_SIMPLEX, 0.6, Scalar(50, 200, 0), 2);
}

//Function for displaying a rotation matrix in a frame
void showInFrameRMatrix(const Mat& frame, const Mat& rmatrix)
{
	ostringstream vector_to_marker;

	vector_to_marker.str(std::string());
	vector_to_marker << std::setprecision(4)
		<< "00: " << std::setw(8) << rmatrix.at<double>(0, 0);
	cv::putText(frame, vector_to_marker.str(),
		Point(10, 360), cv::FONT_HERSHEY_SIMPLEX, 0.6,
		Scalar(0, 0, 255), 2);

	vector_to_marker.str(std::string());
	vector_to_marker << std::setprecision(4)
		<< "01: " << std::setw(8) << rmatrix.at<double>(0, 1);
	cv::putText(frame, vector_to_marker.str(),
		Point(260, 360), cv::FONT_HERSHEY_SIMPLEX, 0.6,
		Scalar(0, 255, 124), 2);

	vector_to_marker.str(std::string());
	vector_to_marker << std::setprecision(4)
		<< "02: " << std::setw(8) << rmatrix.at<double>(0, 2);
	cv::putText(frame, vector_to_marker.str(),
		Point(510, 360), cv::FONT_HERSHEY_SIMPLEX, 0.6,
		Scalar(255, 0, 0), 2);

	vector_to_marker.str(std::string());
	vector_to_marker << std::setprecision(4)
		<< "10: " << std::setw(8) << rmatrix.at<double>(1, 0);
	cv::putText(frame, vector_to_marker.str(),
		Point(10, 410), cv::FONT_HERSHEY_SIMPLEX, 0.6,
		Scalar(0, 0, 255), 2);

	vector_to_marker.str(std::string());
	vector_to_marker << std::setprecision(4)
		<< "11: " << std::setw(8) << rmatrix.at<double>(1, 1);
	cv::putText(frame, vector_to_marker.str(),
		Point(260, 410), cv::FONT_HERSHEY_SIMPLEX, 0.6,
		Scalar(0, 255, 124), 2);

	vector_to_marker.str(std::string());
	vector_to_marker << std::setprecision(4)
		<< "12: " << std::setw(8) << rmatrix.at<double>(1, 2);
	cv::putText(frame, vector_to_marker.str(),
		Point(510, 410), cv::FONT_HERSHEY_SIMPLEX, 0.6,
		Scalar(255, 0, 0), 2);

	vector_to_marker.str(std::string());
	vector_to_marker << std::setprecision(4)
		<< "20: " << std::setw(8) << rmatrix.at<double>(2, 0);
	cv::putText(frame, vector_to_marker.str(),
		Point(10, 460), cv::FONT_HERSHEY_SIMPLEX, 0.6,
		Scalar(0, 0, 255), 2);

	vector_to_marker.str(std::string());
	vector_to_marker << std::setprecision(4)
		<< "21: " << std::setw(8) << rmatrix.at<double>(2, 1);
	cv::putText(frame, vector_to_marker.str(),
		Point(260, 460), cv::FONT_HERSHEY_SIMPLEX, 0.6,
		Scalar(0, 255, 124), 2);

	vector_to_marker.str(std::string());
	vector_to_marker << std::setprecision(4)
		<< "22: " << std::setw(8) << rmatrix.at<double>(2, 2);
	cv::putText(frame, vector_to_marker.str(),
		Point(510, 460), cv::FONT_HERSHEY_SIMPLEX, 0.6,
		Scalar(255, 0, 0), 2);
}

//Function for highlighting marker
int startWebcamMonitoringMod(const Mat& cameraMatrix, Mat& distanceCoefficients, float arucoSquareDimension)
{
	srand(time(NULL));
	int iter = 0;
	int markerNumber = 2;
	Vec3d xyz;
	Mat rmatrix = (Mat_<double>(3, 3) <<
		0, 0, 0,
		0, 0, 0,
		0, 0, 0);
	bool markerDetected = false;

	Mat frame;
	vector<int> markerIds;
	vector<vector<Point2f>> markerCorners, rejectedCandidates;
	aruco::DetectorParameters parameters;
	Ptr<aruco::Dictionary> markerDictionary = aruco::getPredefinedDictionary(aruco::PREDEFINED_DICTIONARY_NAME::DICT_6X6_250);
	VideoCapture vid(0);
	if (!vid.isOpened())
	{
		return -1;
	}
	namedWindow("Окошко в будущее", WINDOW_AUTOSIZE);
	vector<Vec3d> translationVectors, rotationVectors;
	while (true)
	{
		if (!vid.read(frame))break;
		aruco::detectMarkers(frame, markerDictionary, markerCorners, markerIds);
		aruco::estimatePoseSingleMarkers(markerCorners, arucoSquareDimension, cameraMatrix, distanceCoefficients, rotationVectors, translationVectors);
		markerNumberSearch(markerIds, markerNumber, markerDetected);
		for (int i = 0; i < markerIds.size(); i++)
		{
			aruco::drawDetectedMarkers(frame, markerCorners, markerIds);
			if (markerIds[i] == markerNumber)
			{
				xyz = translationVectors[i] * 1000;
				aruco::drawDetectedMarkers(frame, markerCorners, markerIds);
				//aruco::drawAxis(frame, cameraMatrix, distanceCoefficients, rotationVectors[i], translationVectors[i], 0.05f);
				Rodrigues(rotationVectors[i], rmatrix);
			}
		}
		showInFrame(frame, xyz, markerNumber, markerDetected);
		showInFrameRMatrix(frame, rmatrix);
		imshow("Окошко в будущее", frame);
		if (waitKey(30) >= 0) break;
		iter++;
		if (iter == 25)
		{
			markerNumber = rand() % 34;
			iter = 0;
		}
	}
}


int main()
{
	Mat cameraMatrix = Mat::eye(3, 3, CV_64F);
	Mat distanceCoefficients;
	loadCameraCalibration("IloveCameraCalibration", cameraMatrix, distanceCoefficients);
	//cameraCalibrationProcess(cameraMatrix, distanceCoefficients);
	//startWebcamMonitoring(cameraMatrix, distanceCoefficients, arucoSquareDimension);
	startWebcamMonitoringMod(cameraMatrix, distanceCoefficients, arucoSquareDimension);


	return 0;
}