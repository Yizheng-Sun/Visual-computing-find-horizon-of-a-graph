#include <stdio.h>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <opencv2/imgproc/imgproc.hpp>
#include <cmath>
using namespace std;
using namespace cv;

Mat image_grey;
Mat img_canny;
Mat src, dst, cdst, cdstP;
const int max_Value=255;

float Square(float value){
    // Multiply value two times
    return value*value;
}
//Binary threshold variable
int threshold = 70;

//Polynomial regression function
vector<double> fitPoly(vector<Point> points, int n)
{
  //Number of points
  int nPoints = points.size();

  //Vectors for all the points' xs and ys
  vector<float> xValues = vector<float>();
  vector<float> yValues = vector<float>();

  //Split the points into two vectors for x and y values
  for(int i = 0; i < nPoints; i++)
  {
    xValues.push_back(points[i].x);
    yValues.push_back(points[i].y);
  }

  //Augmented matrix
  double matrixSystem[n+1][n+2];
  for(int row = 0; row < n+1; row++)
  {
    for(int col = 0; col < n+1; col++)
    {
      matrixSystem[row][col] = 0;
      for(int i = 0; i < nPoints; i++)
        matrixSystem[row][col] += pow(xValues[i], row + col);
    }

    matrixSystem[row][n+1] = 0;
    for(int i = 0; i < nPoints; i++)
      matrixSystem[row][n+1] += pow(xValues[i], row) * yValues[i];

  }

  //Array that holds all the coefficients
  double coeffVec[n+2] = {};  // the "= {}" is needed in visual studio, but not in Linux

  //Gauss reduction
  for(int i = 0; i <= n-1; i++)
    for (int k=i+1; k <= n; k++)
    {
      double t=matrixSystem[k][i]/matrixSystem[i][i];

      for (int j=0;j<=n+1;j++)
        matrixSystem[k][j]=matrixSystem[k][j]-t*matrixSystem[i][j];

    }

  //Back-substitution
  for (int i=n;i>=0;i--)
  {
    coeffVec[i]=matrixSystem[i][n+1];
    for (int j=0;j<=n+1;j++)
      if (j!=i)
        coeffVec[i]=coeffVec[i]-matrixSystem[i][j]*coeffVec[j];

    coeffVec[i]=coeffVec[i]/matrixSystem[i][i];
  }

  //Construct the cv vector and return it
  vector<double> result = vector<double>();
  for(int i = 0; i < n+1; i++)
    result.push_back(coeffVec[i]);
  return result;
}

//Returns the point for the equation determined
//by a vector of coefficents, at a certain x location
Point pointAtX(vector<double> coeff, double x)
{
  double y = 0;
  for(int i = 0; i < coeff.size(); i++)
  y += pow(x, i) * coeff[i];
  return Point(x, y);
}

Mat blurredImage, origin;

int main (int argc, char *argv[])
{
    string image_path;
    //print the openCV version
    printf("OpenCV version: %d.%d\n", CV_MAJOR_VERSION, CV_MINOR_VERSION);
    cout << "Type image path:"<<endl;
    cin >> image_path;
    src = imread(image_path, IMREAD_GRAYSCALE);
    origin = imread(image_path, IMREAD_COLOR);
    if(src.empty()){
        printf(" Error opening image\n");
        return -1;
    }
    // Edge detection
    GaussianBlur( src, blurredImage, Size( 15, 15 ), 1.0);
    Canny(blurredImage, dst, 100, 150);
    cvtColor(dst, cdst, COLOR_GRAY2BGR);
    imwrite("/home/csimage/Desktop/Visual_Comp/lab4/canny_edge_"+image_path, cdst);
    imshow("canny edges", cdst);

    // Probabilistic Line Transform
    cdstP = cdst.clone();
    vector<Vec4i> linesP; // will hold the results of the detection
    HoughLinesP(dst, linesP, 1, CV_PI/180, 50, 0, 8 ); // runs the actual detection
    // Draw the lines
    for( size_t i = 0; i < linesP.size(); i++ )
    {
    Vec4i l = linesP[i];
    line( cdstP, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0,0,255), 2, LINE_AA);
    }
    imwrite("/home/csimage/Desktop/Visual_Comp/lab4/pHough_lines_"+image_path, cdstP);
    imshow("all hough lines", cdstP);



    //Filter short lines
    Mat cdstP_short_removed = cdst.clone();
    vector<Vec4i> long_lines;
    int max=0;
    int min=0;
    int total_distance = 0;

  // First loop to find statistic about the lines
    for (size_t i=0; i<linesP.size();i++){
        Vec4i l = linesP[i];
        int distance = sqrt(Square(l[0]-l[2])+Square(l[1]-l[3]));
        if (distance>max){
          max = distance;
        }else if (distance<min){
          min = distance;
        }
        total_distance+=distance;
    }
    int avg_dis = total_distance/linesP.size();
    int max_difference = max-min; // Record the difference between longest and shortest lines

    // cout<<avg_dis<<"\n";
    // cout<<max_difference;

// Loop again to remove short lines
    for (size_t i=0; i<linesP.size();i++){
        Vec4i l = linesP[i];
        int distance = sqrt(Square(l[0]-l[2])+Square(l[1]-l[3]));
        if (max_difference > 120){ // if difference is larger than 120, means there are very short lines
          if (distance>=avg_dis+130){ // so we set distance threshold higher
              long_lines.push_back(l);
          }
        }else{
          if (distance>=avg_dis+10){ // if difference is less than 120, means lines' length are similar
              long_lines.push_back(l); // so we set distance threshold lower
          }
        }
    }

    // Draw the long lines
    for( size_t i = 0; i < long_lines.size(); i++ )
    {
    Vec4i l = long_lines[i];
    line( cdstP_short_removed, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0,0,255), 2, LINE_AA);
    }
    imwrite("/home/csimage/Desktop/Visual_Comp/lab4/short_lines_removed_"+image_path, cdstP_short_removed);
    imshow("remove short lines", cdstP_short_removed);

  
    //Filter vertical lines 
    vector<Vec4i> vertical_lines;
    vector<Point> vertical_points;
    for (size_t i=0; i<long_lines.size();i++){
        Vec4i l = long_lines[i];
        int horizon_dis = abs(l[0]-l[2]);
        int vertical_dis = abs(l[1]-l[3]);
        float tan;
        if (horizon_dis != 0 && vertical_dis != 0){
          tan = horizon_dis/vertical_dis;
        } else if (horizon_dis == 0) {
          tan = 0;
        } else if (vertical_dis ==0){
          tan = 10;
        }
        if (tan > 4){
            vertical_lines.push_back(l);
            vertical_points.push_back(Point(l[0],l[1]));
            vertical_points.push_back(Point(l[2],l[3]));
        }
    }
    // Draw the horizontal lines
    Mat horizontal_lines = cdst.clone();
    for( size_t i = 0; i < vertical_lines.size(); i++ )
    {
    Vec4i l = vertical_lines[i];
    line( horizontal_lines, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0,0,255), 3, LINE_AA);
    }
    imwrite("/home/csimage/Desktop/Visual_Comp/lab4/horizontal_lines_"+image_path, horizontal_lines);
    imshow("remove vertical lines", horizontal_lines);

    vector<double> coeff = fitPoly(vertical_points, 2);
    vector<Point> polynomial_points;
    for (size_t i=0; i<2000;i++){
        // Point p = vertical_points[i];
        Point polynomial_point = pointAtX(coeff, i);
        polynomial_points.push_back(polynomial_point);
    }

    // draw curve
    cv::polylines(origin, polynomial_points, false, Scalar(0, 0, 255), 3);
    imwrite("/home/csimage/Desktop/Visual_Comp/lab4/detected_horizon"+image_path, origin);

    imshow("Detected horizon", origin);
    waitKey(0);
    destroyAllWindows();

    return 0;
}