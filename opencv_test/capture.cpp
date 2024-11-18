#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/core/base.hpp>
#include <opencv2/core/types.hpp>

#include <opencv2/imgproc.hpp>

#include <opencv2/highgui/highgui.hpp>

using namespace cv;
using namespace std;

#define ESCAPE_KEY (27)
#define SYSTEM_ERROR (-1)

#define WIDTH  (320)
#define HEIGHT (240) 

int main()
{
//    VideoCapture cam0(0);
   VideoCapture cap("IMG_2841_MOV_AdobeExpress.mp4");
   namedWindow("video_display");
   char winInput;


   cap.set(CAP_PROP_FRAME_WIDTH, WIDTH);
   cap.set(CAP_PROP_FRAME_HEIGHT, HEIGHT);

//    if(cap.get(CAP_PROP_FRAME_WIDTH) != WIDTH ||
//       cap.get(CAP_PROP_FRAME_HEIGHT) != HEIGHT){

//         cout << "dimension mismatch" << endl;
//     }
   
   
   while (1)
   {
      Mat frame, frame_final;
      
      // read in a frame
      cap.read(frame); 

      if(!frame){
        cout << "no frame detected, ending program" << endl;
        break;
      }
      
      
      imshow("video_display", frame);

      if ((winInput = waitKey(10)) == ESCAPE_KEY)
      {
          break;
      }
      
   }

   destroyWindow("video_display"); 
};
