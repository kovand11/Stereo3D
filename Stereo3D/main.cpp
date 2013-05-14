
#include <iostream>
#include <string>
#include <pcl/common/common_headers.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <boost/thread/thread.hpp>
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/contrib/contrib.hpp"

#include <vector>
#include <string>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>

using namespace cv;
using namespace std;



static bool readStringList( const string& filename, vector<string>& list)
{
    list.resize(0);
    FileStorage fs(filename, FileStorage::READ);
    if( !fs.isOpened() )
        return false;

    FileNode node = fs.getFirstTopLevelNode();
    if( node.type() != FileNode::SEQ )
        return false;
    for(FileNodeIterator it = node.begin() ; it != node.end(); ++it )
        list.push_back(static_cast<string>(*it));
    return true;
}


static void saveXYZ(const char* filename, const Mat& mat)
{
	const double max_z = 1.0e4;
	ofstream outputFile(filename);
	for(int y = 0; y < mat.rows; y++)
    {
		for(int x = 0; x < mat.cols; x++)
        {
            Vec3f point = mat.at<Vec3f>(y, x);
            if(fabs(point[2] - max_z) < FLT_EPSILON || fabs(point[2]) > max_z) continue;
			outputFile<<point[0]<<" "<<point[1]<<" "<<point[2]<<endl;
        }
	}
	outputFile.close();
}

#define CUSTOM_REPROJECT //bypass cv::reprojectImageTo3D


static void StereoCalib(const vector<string>& imagelist, Size boardSize,Mat Q, bool useCalibrated=true, bool showRectified=true)
{
    if( imagelist.size() % 2 != 0 )
    {
        cout << "Error: the image list contains odd (non-even) number of elements\n";
        return;
    }

    bool displayCorners = true;
    const int maxScale = 2;	
    const float squareSize = 1600.f;  // Set this to your actual square size
    // ARRAY AND VECTOR STORAGE:

    vector<vector<Point2f> > imagePoints[2];
    vector<vector<Point3f> > objectPoints;
    Size imageSize;

    int i, j, k, nimages = (int)imagelist.size()/2;

    imagePoints[0].resize(nimages);
    imagePoints[1].resize(nimages);
    vector<string> goodImageList;

    for( i = j = 0; i < nimages; i++ )
    {
        for( k = 0; k < 2; k++ )
        {
            const string& filename = imagelist[i*2+k];
            Mat img = imread(filename, 0);
            if(img.empty())
                break;
            if( imageSize == Size() )
                imageSize = img.size();
            else if( img.size() != imageSize )
            {
                cout << "The image " << filename << " has the size different from the first image size. Skipping the pair\n";
                break;
            }
            bool found = false;
            vector<Point2f>& corners = imagePoints[k][j];
            for( int scale = 1; scale <= maxScale; scale++ )
            {
                Mat timg;
                if( scale == 1 )
                    timg = img;
                else
                    resize(img, timg, Size(), scale, scale);
                found = findChessboardCorners(timg, boardSize, corners,CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_NORMALIZE_IMAGE);
                if( found )
                {
                    if( scale > 1 )
                    {
                        Mat cornersMat(corners);
                        cornersMat *= 1./scale;
                    }
                    break;
                }
            }
            if( displayCorners )
            {
                cout << filename << endl;
                Mat cimg, cimg1;
                cvtColor(img, cimg, CV_GRAY2BGR);
                drawChessboardCorners(cimg, boardSize, corners, found);
                double sf = 640./MAX(img.rows, img.cols);
                resize(cimg, cimg1, Size(), sf, sf);
				if (k==0)
					imshow("corners1", cimg1);
				else
					imshow("corners2", cimg1);
                char c = (char)waitKey(500);
                if( c == 27 || c == 'q' || c == 'Q' ) //Allow ESC to quit
                    exit(-1);
            }
            else
                putchar('.');
            if( !found )
                break;
            cornerSubPix(img, corners, Size(11,11), Size(-1,-1),
                         TermCriteria(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS,
                                      30, 0.01));
        }
        if( k == 2 )
        {
            goodImageList.push_back(imagelist[i*2]);
            goodImageList.push_back(imagelist[i*2+1]);
            j++;
        }
    }
    cout << j << " pairs have been successfully detected.\n";
    nimages = j;
    if( nimages < 1 )
    {
        cout << "Error: too little pairs to run the calibration\n";
        return;
    }

    imagePoints[0].resize(nimages);
    imagePoints[1].resize(nimages);
    objectPoints.resize(nimages);

    for( i = 0; i < nimages; i++ )
    {
        for( j = 0; j < boardSize.height; j++ )
            for( k = 0; k < boardSize.width; k++ )
                objectPoints[i].push_back(Point3f(j*squareSize, k*squareSize, 0));
    }

    cout << "Running stereo calibration ...\n";

    Mat cameraMatrix[2], distCoeffs[2];
    cameraMatrix[0] = Mat::eye(3, 3, CV_64F);
    cameraMatrix[1] = Mat::eye(3, 3, CV_64F);
    Mat R, T, E, F;

    double rms = stereoCalibrate(objectPoints, imagePoints[0], imagePoints[1],
                    cameraMatrix[0], distCoeffs[0],
                    cameraMatrix[1], distCoeffs[1],
                    imageSize, R, T, E, F,
                    TermCriteria(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, 100, 1e-5),
                    CV_CALIB_FIX_ASPECT_RATIO +
                    CV_CALIB_ZERO_TANGENT_DIST +
                    CV_CALIB_SAME_FOCAL_LENGTH +
                    CV_CALIB_RATIONAL_MODEL +

					CV_CALIB_FIX_K3 + CV_CALIB_FIX_K4 + CV_CALIB_FIX_K5);
    cout << "done with RMS error=" << rms << endl;

// CALIBRATION QUALITY CHECK
// because the output fundamental matrix implicitly
// includes all the output information,
// we can check the quality of calibration using the
// epipolar geometry constraint: m2^t*F*m1=0
    double err = 0;
    int npoints = 0;
    vector<Vec3f> lines[2];
    for( i = 0; i < nimages; i++ )
    {
        int npt = (int)imagePoints[0][i].size();
        Mat imgpt[2];
        for( k = 0; k < 2; k++ )
        {
            imgpt[k] = Mat(imagePoints[k][i]);
            undistortPoints(imgpt[k], imgpt[k], cameraMatrix[k], distCoeffs[k], Mat(), cameraMatrix[k]);
            computeCorrespondEpilines(imgpt[k], k+1, F, lines[k]);
        }
        for( j = 0; j < npt; j++ )
        {
            double errij = fabs(imagePoints[0][i][j].x*lines[1][j][0] +
                                imagePoints[0][i][j].y*lines[1][j][1] + lines[1][j][2]) +
                           fabs(imagePoints[1][i][j].x*lines[0][j][0] +
                                imagePoints[1][i][j].y*lines[0][j][1] + lines[0][j][2]);
            err += errij;
        }
        npoints += npt;
    }
    cout << "average reprojection err = " <<  err/npoints << endl;

    // save intrinsic parameters
    FileStorage fs("intrinsics.yml", CV_STORAGE_WRITE);
    if( fs.isOpened() )
    {
        fs << "M1" << cameraMatrix[0] << "D1" << distCoeffs[0] <<
            "M2" << cameraMatrix[1] << "D2" << distCoeffs[1];
        fs.release();
    }
    else
        cout << "Error: can not save the intrinsic parameters\n";

    Mat R1, R2, P1, P2;//, Q;
    Rect validRoi[2];

    stereoRectify(cameraMatrix[0], distCoeffs[0],
                  cameraMatrix[1], distCoeffs[1],
                  imageSize, R, T, R1, R2, P1, P2, Q,
                  CALIB_ZERO_DISPARITY, 1, imageSize, &validRoi[0], &validRoi[1]);

    fs.open("extrinsics.yml", CV_STORAGE_WRITE);
    if( fs.isOpened() )
    {
        fs << "R" << R << "T" << T << "R1" << R1 << "R2" << R2 << "P1" << P1 << "P2" << P2 << "Q" << Q;
        fs.release();
    }
    else
        cout << "Error: can not save the intrinsic parameters\n";

    // OpenCV can handle left-right
    // or up-down camera arrangements
    bool isVerticalStereo = fabs(P2.at<double>(1, 3)) > fabs(P2.at<double>(0, 3));

// COMPUTE AND DISPLAY RECTIFICATION
    if( !showRectified )
        return;

    Mat rmap[2][2];
// IF BY CALIBRATED (BOUGUET'S METHOD)
    if( useCalibrated )
    {
        // we already computed everything
    }
// OR ELSE HARTLEY'S METHOD
    else
 // use intrinsic parameters of each camera, but
 // compute the rectification transformation directly
 // from the fundamental matrix
    {
        vector<Point2f> allimgpt[2];
        for( k = 0; k < 2; k++ )
        {
            for( i = 0; i < nimages; i++ )
                std::copy(imagePoints[k][i].begin(), imagePoints[k][i].end(), back_inserter(allimgpt[k]));
        }
        F = findFundamentalMat(Mat(allimgpt[0]), Mat(allimgpt[1]), FM_8POINT, 0, 0);
        Mat H1, H2;
        stereoRectifyUncalibrated(Mat(allimgpt[0]), Mat(allimgpt[1]), F, imageSize, H1, H2, 3);

        R1 = cameraMatrix[0].inv()*H1*cameraMatrix[0];
        R2 = cameraMatrix[1].inv()*H2*cameraMatrix[1];
        P1 = cameraMatrix[0];
        P2 = cameraMatrix[1];
    }

    //Precompute maps for cv::remap()
    initUndistortRectifyMap(cameraMatrix[0], distCoeffs[0], R1, P1, imageSize, CV_16SC2, rmap[0][0], rmap[0][1]);
    initUndistortRectifyMap(cameraMatrix[1], distCoeffs[1], R2, P2, imageSize, CV_16SC2, rmap[1][0], rmap[1][1]);

    Mat canvas;
    double sf;
    int w, h;
    if( !isVerticalStereo )
    {
        sf = 600./MAX(imageSize.width, imageSize.height);
        w = cvRound(imageSize.width*sf);
        h = cvRound(imageSize.height*sf);
        canvas.create(h, w*2, CV_8UC3);
    }
    else
    {
        sf = 300./MAX(imageSize.width, imageSize.height);
        w = cvRound(imageSize.width*sf);
        h = cvRound(imageSize.height*sf);
        canvas.create(h*2, w, CV_8UC3);
    }

    for( i = 0; i < nimages; i++ )
    {
        for( k = 0; k < 2; k++ )
        {
            Mat img = imread(goodImageList[i*2+k], 0), rimg, cimg;
            remap(img, rimg, rmap[k][0], rmap[k][1], CV_INTER_LINEAR);
            cvtColor(rimg, cimg, CV_GRAY2BGR);
            Mat canvasPart = !isVerticalStereo ? canvas(Rect(w*k, 0, w, h)) : canvas(Rect(0, h*k, w, h));
            resize(cimg, canvasPart, canvasPart.size(), 0, 0, CV_INTER_AREA);
            if( useCalibrated )
            {
                Rect vroi(cvRound(validRoi[k].x*sf), cvRound(validRoi[k].y*sf),
                          cvRound(validRoi[k].width*sf), cvRound(validRoi[k].height*sf));
                rectangle(canvasPart, vroi, Scalar(0,0,255), 3, 8);
            }
        }

        if( !isVerticalStereo )
            for( j = 0; j < canvas.rows; j += 16 )
                line(canvas, Point(0, j), Point(canvas.cols, j), Scalar(0, 255, 0), 1, 8);
        else
            for( j = 0; j < canvas.cols; j += 16 )
                line(canvas, Point(j, 0), Point(j, canvas.rows), Scalar(0, 255, 0), 1, 8);
        imshow("rectified", canvas);
        char c = (char)waitKey();
        if( c == 27 || c == 'q' || c == 'Q' )
            break;
    }
}


//This function creates a PCL visualizer, sets the point cloud to view and returns a pointer
boost::shared_ptr<pcl::visualization::PCLVisualizer> createVisualizer (pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr cloud)
{
  boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
  viewer->setBackgroundColor (0, 0, 0);
  pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(cloud);
  viewer->addPointCloud<pcl::PointXYZRGB> (cloud, rgb, "reconstruction");
  //viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "reconstruction");
  viewer->addCoordinateSystem ( 1.0 );
  viewer->initCameraParameters ();
  return (viewer);
}


int main( int argc, char** argv )
{
	map <string,string> CLP; //command line parameters

	string currentFlag;
	for( int i = 1; i < argc; i++ )
	{
		string Param=string(argv[i]);
		if (Param[0]=='-')
		{
			currentFlag=Param;
		}
		else
		{
			CLP[currentFlag]=Param;
		}
	}

	bool missingParam=false;
	

	if (CLP.count("-task")==0){cout<<"No task defined."<<endl;missingParam=true;}

	if (CLP["-task"]=="calib")
	{
		if (CLP.count("-chessw")==0){cout<<"No chessboard width defined."<<endl;missingParam=true;}
		if (CLP.count("-chessh")==0){cout<<"No chessboard height defined."<<endl;missingParam=true;}
		if (CLP.count("-calimgs")==0){cout<<"No input calib file defined."<<endl;missingParam=true;}
	}

	if (CLP["-task"]=="recon")
	{
		if (CLP.count("-intr")==0){cout<<"No intrinsic param. defined."<<endl;missingParam=true;}
		if (CLP.count("-extr")==0){cout<<"No extrinsic param. defined."<<endl;missingParam=true;}
		if (CLP.count("-limg")==0){cout<<"No left image defined."<<endl;missingParam=true;}
		if (CLP.count("-rimg")==0){cout<<"No right image defined"<<endl;missingParam=true;}						
	}

	if (missingParam)
		return 0;


	Size boardSize;
	boardSize.width=atoi(CLP["-chessw"].c_str());
	boardSize.height=atoi(CLP["-chessh"].c_str());
	string calibImagesXML=CLP["-calimgs"];
	vector<string> calibImagesList;
	bool success = readStringList(calibImagesXML, calibImagesList);
	bool showRectified=true;
	Mat Q;

	if(!success || calibImagesList.empty())
	{
		cout << "Can not open " << calibImagesXML << " or the string list is empty." << endl;
		return 0;
	}

	StereoCalib(calibImagesList, boardSize, Q , true, showRectified);

	string intrinsicFile=CLP["-intr"];
	string extrinsicFile=CLP["-extr"];


	//TMP

	//const char* algorithm_opt = "--algorithm=";
    //const char* maxdisp_opt = "--max-disparity=";
    //const char* blocksize_opt = "--blocksize=";
    //const char* nodisplay_opt = "--no-display=";
    //const char* scale_opt = "--scale=";


	string leftImgFile=CLP["-limg"];
	string rightImgFile=CLP["-rimg"];
	string disparityFile=CLP["-disp"];
	string pointcloudFile=CLP["-pointcl"];
	string matchAlg=CLP["-matchalg"];




    enum { STEREO_BM=0, STEREO_SGBM=1, STEREO_HH=2, STEREO_VAR=3 };
    int alg = STEREO_VAR; //SELECT
    int SADWindowSize = 0, numberOfDisparities = 0;
    bool no_display = false;
    float scale = 1.f;

    StereoBM bm;
    StereoSGBM sgbm;
    StereoVar var;


    no_display = true;


	


    int color_mode = alg == STEREO_BM ? 0 : -1;
    Mat img1 = imread(leftImgFile, color_mode);
	Mat img2 = imread(rightImgFile, color_mode);

    if( scale != 1.f )
    {
        Mat temp1, temp2;
        int method = scale < 1 ? INTER_AREA : INTER_CUBIC;
        resize(img1, temp1, Size(), scale, scale, method);
        img1 = temp1;
        resize(img2, temp2, Size(), scale, scale, method);
        img2 = temp2;
    }

    Size img_size = img1.size();

    Rect roi1, roi2;

  // reading intrinsic parameters
    FileStorage fs(intrinsicFile, CV_STORAGE_READ);
    if(!fs.isOpened())
    {
        printf("Failed to open file %s\n", intrinsicFile);
        return -1;
    }

    Mat M1, D1, M2, D2;
    fs["M1"] >> M1;
    fs["D1"] >> D1;
    fs["M2"] >> M2;
    fs["D2"] >> D2;

    M1 *= scale;
    M2 *= scale;

	fs.open(extrinsicFile, CV_STORAGE_READ);
    if(!fs.isOpened())
    {
        printf("Failed to open file %s\n", extrinsicFile);
        return -1;
    }

    Mat R, T, R1, P1, R2, P2;
    fs["R"] >> R;
    fs["T"] >> T;

    stereoRectify( M1, D1, M2, D2, img_size, R, T, R1, R2, P1, P2, Q, CALIB_ZERO_DISPARITY, -1, img_size, &roi1, &roi2 );

    Mat map11, map12, map21, map22;
    initUndistortRectifyMap(M1, D1, R1, P1, img_size, CV_16SC2, map11, map12);
    initUndistortRectifyMap(M2, D2, R2, P2, img_size, CV_16SC2, map21, map22);

    Mat img1r, img2r;
    remap(img1, img1r, map11, map12, INTER_LINEAR);
    remap(img2, img2r, map21, map22, INTER_LINEAR);

    img1 = img1r;
    img2 = img2r;


    numberOfDisparities = numberOfDisparities > 0 ? numberOfDisparities : ((img_size.width/8) + 15) & -16;

    bm.state->roi1 = roi1;
    bm.state->roi2 = roi2;
    bm.state->preFilterCap = 31;
    bm.state->SADWindowSize = SADWindowSize > 0 ? SADWindowSize : 9;
    bm.state->minDisparity = 0;
    bm.state->numberOfDisparities = numberOfDisparities;
    bm.state->textureThreshold = 10;
    bm.state->uniquenessRatio = 15;
    bm.state->speckleWindowSize = 100;
    bm.state->speckleRange = 32;
    bm.state->disp12MaxDiff = 1;

    sgbm.preFilterCap = 63;
    sgbm.SADWindowSize = SADWindowSize > 0 ? SADWindowSize : 3;

    int cn = img1.channels();

    sgbm.P1 = 8*cn*sgbm.SADWindowSize*sgbm.SADWindowSize;
    sgbm.P2 = 32*cn*sgbm.SADWindowSize*sgbm.SADWindowSize;
    sgbm.minDisparity = 0;
    sgbm.numberOfDisparities = numberOfDisparities;
    sgbm.uniquenessRatio = 10;
    sgbm.speckleWindowSize = bm.state->speckleWindowSize;
    sgbm.speckleRange = bm.state->speckleRange;
    sgbm.disp12MaxDiff = 1;
    sgbm.fullDP = alg == STEREO_HH;

    var.levels = 3;                                 // ignored with USE_AUTO_PARAMS
    var.pyrScale = 0.5;                             // ignored with USE_AUTO_PARAMS
    var.nIt = 25;
    var.minDisp = -numberOfDisparities;
    var.maxDisp = 0;
    var.poly_n = 3;
    var.poly_sigma = 0.0;
    var.fi = 15.0f;
    var.lambda = 0.03f;
    var.penalization = var.PENALIZATION_TICHONOV;   // ignored with USE_AUTO_PARAMS
    var.cycle = var.CYCLE_V;                        // ignored with USE_AUTO_PARAMS
    var.flags = var.USE_SMART_ID | var.USE_AUTO_PARAMS | var.USE_INITIAL_DISPARITY | var.USE_MEDIAN_FILTERING ;

    Mat disp, disp8;
    //Mat img1p, img2p, dispp;
    //copyMakeBorder(img1, img1p, 0, 0, numberOfDisparities, 0, IPL_BORDER_REPLICATE);
    //copyMakeBorder(img2, img2p, 0, 0, numberOfDisparities, 0, IPL_BORDER_REPLICATE);

    int64 t = getTickCount();
    if( alg == STEREO_BM )
        bm(img1, img2, disp);
    else if( alg == STEREO_VAR ) {
        var(img1, img2, disp);
    }
    else if( alg == STEREO_SGBM || alg == STEREO_HH )
        sgbm(img1, img2, disp);
    t = getTickCount() - t;
    printf("Time elapsed: %fms\n", t*1000/getTickFrequency());

    //disp = dispp.colRange(numberOfDisparities, img1p.cols);
    if( alg != STEREO_VAR )
        disp.convertTo(disp8, CV_8U, 255/(numberOfDisparities*16.));
    else
        disp.convertTo(disp8, CV_8U);
    if( !no_display )
    {
        namedWindow("left", 1);
        imshow("left", img1);
        namedWindow("right", 1);
        imshow("right", img2);
        namedWindow("disparity", 0);
        imshow("disparity", disp8);
        printf("press any key to continue...");
        fflush(stdout);
        waitKey();
        printf("\n");
    }

	//if(disparityFile)
	imwrite(disparityFile, disp8);

    //if(point_cloud_filename)

    printf("storing the point cloud...");
    fflush(stdout);
    Mat xyz;
    reprojectImageTo3D(disp, xyz, Q, true);
	saveXYZ(pointcloudFile.c_str(), xyz);
    printf("\n");


	 ////END_OF_DISPARITY////////////




	////////START_OF_3DIMAGE/////////
	cv::Mat img_rgb;
	cv::Mat img_disparity;
	img_rgb = cv::imread(leftImgFile, CV_LOAD_IMAGE_COLOR);
	//img_disparity = cv::imread(disparityFile, CV_LOAD_IMAGE_GRAYSCALE);
	img_disparity=disp8;

	//If size of Q is not 4x4 exit
	if (Q.cols != 4 || Q.rows != 4)
	{
		std::cerr << "ERROR: Could not read matrix Q (doesn't exist or size is not 4x4)" << std::endl;
		return 1;
	}

	#ifdef CUSTOM_REPROJECT
	//Get the interesting parameters from Q
	double Q03, Q13, Q23, Q32, Q33;
	Q03 = Q.at<double>(0,3);
	Q13 = Q.at<double>(1,3);
	Q23 = Q.at<double>(2,3);
	Q32 = Q.at<double>(3,2);
	Q33 = Q.at<double>(3,3);
  
	std::cout << "Q(0,3) = "<< Q03 <<"; Q(1,3) = "<< Q13 <<"; Q(2,3) = "<< Q23 <<"; Q(3,2) = "<< Q32 <<"; Q(3,3) = "<< Q33 <<";" << std::endl;
  
	#endif  
  
  
	
	//Show the values inside Q (for debug purposes)

//  for (int y = 0; y < Q.rows; y++)
  //{
   // const double* Qy = Q.ptr<double>(y);
   // for (int x = 0; x < Q.cols; x++)
   // {
     // std::cout << "Q(" << x << "," << y << ") = " << Qy[x] << std::endl;
   // }
 // }
  
	//Both images must be same size
	if (img_rgb.size() != img_disparity.size())
	{
		std::cerr << "ERROR: rgb-image and disparity-image have different sizes " << std::endl;
		return 1;
	}
  
	//Show both images (for debug purposes)
	cv::namedWindow("rgb-image");
	cv::namedWindow("disparity-image");
	cv::imshow("rbg-image", img_rgb);
	cv::imshow("disparity-image", img_disparity);
	std::cout << "Press a key to continue..." << std::endl;
	cv::waitKey(0);
	cv::destroyWindow("rgb-image");
	cv::destroyWindow("disparity-image");
  
	#ifndef CUSTOM_REPROJECT
	//Create matrix that will contain 3D corrdinates of each pixel

	cv::Mat recons3D(img_disparity.size(), CV_32FC3);
  
	//Reproject image to 3D
	std::cout << "Reprojecting image to 3D..." << std::endl;
	cv::reprojectImageTo3D( img_disparity, recons3D, Q, false, CV_32F );
	#endif  
	//Create point cloud and fill it
	std::cout << "Creating Point Cloud..." <<std::endl;
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr point_cloud_ptr (new pcl::PointCloud<pcl::PointXYZRGB>);
  
	double px, py, pz;
	uchar pr, pg, pb;
  
	for (int i = 0; i < img_rgb.rows; i++)
	{
		uchar* rgb_ptr = img_rgb.ptr<uchar>(i);
#ifdef CUSTOM_REPROJECT
		uchar* disp_ptr = img_disparity.ptr<uchar>(i);
#else
		double* recons_ptr = recons3D.ptr<double>(i);
#endif
		for (int j = 0; j < img_rgb.cols; j++)
		{
      //Get 3D coordinates
#ifdef CUSTOM_REPROJECT
			uchar d = disp_ptr[j];
			if ( d == 0 ) continue; //Discard bad pixels
			double pw = -1.0 * static_cast<double>(d) * Q32 + Q33; 
			px = static_cast<double>(j) + Q03;
			py = static_cast<double>(i) + Q13;
			pz = Q23;

			px = px/pw;
			py = py/pw;
			pz = pz/pw;

#else
			px = recons_ptr[3*j];
			py = recons_ptr[3*j+1];
			pz = recons_ptr[3*j+2];
#endif
      
		//Get RGB info
			pb = rgb_ptr[3*j];
			pg = rgb_ptr[3*j+1];
			pr = rgb_ptr[3*j+2];
      
		//Insert info into point cloud structure
			pcl::PointXYZRGB point;
			point.x = px;
			point.y = py;
			point.z = pz;
			uint32_t rgb = (static_cast<uint32_t>(pr) << 16 |
			      static_cast<uint32_t>(pg) << 8 | static_cast<uint32_t>(pb));
			point.rgb = *reinterpret_cast<float*>(&rgb);
			point_cloud_ptr->points.push_back (point);
		}
	}
	point_cloud_ptr->width = (int) point_cloud_ptr->points.size();
	point_cloud_ptr->height = 1;
  
	//Create visualizer
	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer;
	viewer = createVisualizer( point_cloud_ptr );
  
	//Main loop
	while ( !viewer->wasStopped())
	{
		viewer->spinOnce(100);
		boost::this_thread::sleep (boost::posix_time::microseconds (100000));
	}
  
  ////////END_OF_3DIMAGE/////////	

	return 0;
}






