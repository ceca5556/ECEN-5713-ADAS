#include <iostream>
#include <opencv2/highgui/highgui.hpp>

#include <opencv2/imgproc.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/core/base.hpp>
#include <opencv2/core/types.hpp>

#include "opencv2/objdetect.hpp"
#include <opencv2/videoio.hpp> // Video write

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#include <pthread.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sched.h>
#include <unistd.h>
#include <syslog.h>
#include <omp.h>

// #include "lanelib.hpp"

using namespace cv; using namespace std;
// double alpha=1.0;  int beta=10;  /* contrast and brightness control */


#define ESCAPE_KEY (27)
#define SYSTEM_ERROR (-1)

#define NUM_FEAT (2)
#define NUM_CPUS (4)
#define SCHED_POLICY SCHED_FIFO

#define LANE_DETECT (0)
#define STOP_DETECT (1)


#define SECOND   (1)
#define MINUTE   (60)


typedef struct frameParams_s
{

  pthread_t thread_ID;
  int frameIdx;
  Mat frame;
  Mat frame_gray;

  Rect lane_roi;

  Rect stop_roi;
  int focal;
  bool hog;
  
  Size in_size;

  bool disp;

} frameParams_t;

typedef struct
{

  // pthread_t thread_ID;
  // int frameIdx;
  // Mat orig_frame;

  //VideoCapture input;
  Mat lane_frame;
  Vec4i left_lane;
  Vec4i right_lane;
  Rect lane_roi;
  
  
  Size in_size;

  bool disp;
  
//    float time;
  //Mat image;
  //Mat new_image;
  //int channel;
} laneParams_t;

typedef struct
{

  //CascadeClassifier stop_cascade;
  Mat stop_frame;
  vector<Rect> stop_signs;
  Rect stop_roi;
  int focal;
  bool hog;
  
  Size in_size;

  bool disp;
  
//    float time;
  //Mat image;
  //Mat new_image;
  //int channel;
} stopParams_t;

int lowThreshold = 17;
const int max_lowThreshold = 200;
const int ratio_canny = 4;
const int kernel_size = 3;


int numberOfProcessors = NUM_CPUS;


// pthread_t threads[NUM_FEAT];
//threadParams_lane_t laneParams;
// threadParams_t threadParams[NUM_FEAT];
// frameParams_t threadParams;
pthread_attr_t sched_attr;
struct sched_param sched_param;

// Mat orig_frame;
char winInput;

bool DISP_LINES = true;
bool DISP_STOP = true;


const string final_out = "output";

Size in_size;

int lane_roi_top_offset = 125;
int lane_roi_bottom_offset = 375;
int lane_roi_width_offset = 0;
int lane_line_thresh = 50;


HOGDescriptor stop_hog;
CascadeClassifier stop_cascade;
int stop_roi_top_offset = 150;
int stop_roi_bottom_offset = 50;
int stop_roi_left_offset = 50;
int stop_roi_right_offset = 0;

static int fps_sum = 0;
static int call_count = 0;
static float sensor_width = 4.55; // milimeters


/**************************************************************
*
*  @brief dt_sec: calculates the difference between two timespec
*         structs in seconds
*
*  @param[in] start
*        - pointer to starting timespec struct (used for 60 sec calculation)
*
*  @param[in] start_frame
*      - pointer to starting timespec struct of the frame
*
*
*  @param[in] stop
*        - pointer to stopping timespec struct of the frame
*
*
*  @param[in] frame_num
*        - pointer to number of frames
*
*  @returns none
*
*
***************************************************************/
static void frame_time_calc(struct timespec *start,struct timespec *start_frame, struct timespec *stop,int *frame_num){
  
  int full_time = stop->tv_sec - start->tv_sec;
  int frme_time = stop->tv_sec - start_frame->tv_sec;
  
  
  
  if(frme_time >= SECOND){
    syslog(LOG_NOTICE, "FPS: %d frames in %d second", *frame_num, frme_time);
    fps_sum += *frame_num;
    call_count++;
    *frame_num = 0;
    clock_gettime(CLOCK_REALTIME, start_frame);
  }
  
  // show when 60 seconds have passed
  if(full_time >= MINUTE){
    int fps_avg = fps_sum/call_count;
    syslog(LOG_NOTICE, "FPS: **************************** %d seconds have passed / %d avg FPS **********************", full_time,fps_avg);
    fps_sum = 0;
    call_count = 0;
    clock_gettime(CLOCK_REALTIME, start);
  }

}


/**************************************************************
*
*  @brief edge detection using Canny method         
*
*
***************************************************************/
static void canny_edge(Mat *frame, 
                       Mat *canny_frame,
                       bool disp){//,
                       
  
  Canny(*frame, *canny_frame, lowThreshold, lowThreshold*ratio_canny, kernel_size);
  //Canny(*canny_frame, *canny_frame, lowThreshold_in, lowThreshold_in*ratio_canny_in, kernel_size_in);
  
  if(disp){
    namedWindow("canny",WINDOW_NORMAL);
    imshow("canny", *canny_frame );
  }
  
  if (winInput == '.')// || winInput == 'T')
  {
    lowThreshold += 5;
    if(lowThreshold > max_lowThreshold){
      lowThreshold = 0;
    }
    //hough_thresh += 10;
    //if(hough_thresh > max_hough_thresh){
    //  hough_thresh = 0;
    //}
  }
  else if(winInput == ','){
    lowThreshold -= 5;
    if(lowThreshold < 0){
      lowThreshold = max_lowThreshold;
    }
    //hough_thresh -= 10;
    //if(hough_thresh < 0){
    //  hough_thresh = max_hough_thresh;
    //}
  }
  //cout << "lowThreshold: " << lowThreshold << endl;

}

/*
*    stop sign distance estimation
*/
int stop_dist(Rect rectangle, float f){

  int sign_width = 750; // in mm
  int pixels = rectangle.width; // pixel width of detected 
  
  f = f*in_size.width/sensor_width; // mm * pixel/mm = pixels
  
  float dist = (sign_width*f)/pixels; // mm * pixel / pixel
  
  return dist;
  //syslog(LOG_NOTICE, "stop: stop sign #%d distance: %d mm", num, dist);
  //cout << "stop sign distance: " << dist << "mm" << endl;

}

/*
*    stop sign detection function performed by the threads
*/
void *stop_detect(void *threadp){

    stopParams_t *threadParams = (stopParams_t *)threadp;
    vector<Rect> stop_signs;
    float dist;
    
    if(threadParams->disp){
    //   namedWindow("original",WINDOW_NORMAL);
       namedWindow("stop_roi",WINDOW_NORMAL);
       
       imshow("stop_roi",threadParams->stop_frame);
    }
     
    if( threadParams->hog)
      stop_hog.detectMultiScale(threadParams->stop_frame, threadParams->stop_signs);
    else
      stop_cascade.detectMultiScale(threadParams->stop_frame, threadParams->stop_signs);
    
    int detected_num = threadParams->stop_signs.size();
    //int thrds = 12;
    
    #pragma omp parallel private(dist)
    {
      #pragma omp for
      for(int k = 0; k<detected_num;k++){
    
          threadParams->stop_signs[k].x += threadParams->stop_roi.x;
          threadParams->stop_signs[k].y += threadParams->stop_roi.y;
          
          dist = stop_dist(threadParams->stop_signs[k],threadParams->focal);
          syslog(LOG_NOTICE, "STOP_SIGN: stop sign #%d/%d distance: %02f mm", k+1,detected_num, dist);
        
      }
    }

    return NULL;

}


void lane_check(Vec4i left_lane, Vec4i right_lane){

    int center = in_size.width/2;
    
    bool drift_left = (left_lane[0] < center) && ( left_lane[0] > (center - center/8));
    bool drift_right = (right_lane[0] > center) && ( right_lane[0] < (center + center/8));
    
    if(drift_left && drift_right){
      syslog(LOG_NOTICE, "LANE: detected shallow lane");
      //cout << "LANE: detected shallow lane" << endl;
    }
    else if(drift_left){
      syslog(LOG_NOTICE, "LANE: warning, drifting out of lane from the left");
      //cout << "LANE: warning, drifting out of lane from the left" << endl;
    }
    else if(drift_right){
      syslog(LOG_NOTICE, "LANE: warning, drifting out of lane from the right");
      //cout << "LANE: warning, drifting out of lane from the right" << endl;
    }

}

/*
*    lane detection function performed by the threads
*/
void *lane_detect(void *threadp){

    laneParams_t *threadParams = (laneParams_t *)threadp;
    
    if(threadParams->disp){
    //   namedWindow("original",WINDOW_NORMAL);
       namedWindow("lane_roi",WINDOW_NORMAL);
       //namedWindow("canny",WINDOW_NORMAL);
       
       imshow("lane_roi",threadParams->lane_frame);
     }
    
    Mat canny_frame;//, roi_frame;
    vector<Vec4i> lines;
    
    //roi_frame = threadParams->frame(
    canny_edge(&threadParams->lane_frame,&canny_frame,threadParams->disp);//,50,200,3);
    HoughLinesP(canny_frame, lines, 1, CV_PI/180, 50, 50, 10 );
    
    
    //Vec4i left_lane, right_lane;
    int Lx = 0;
    int Rx = in_size.width;
    Vec4i l;
    
    
    int detected_num = lines.size();
    //int thrd = 12;
    //int left[detected_num];
    //int right[detected_num];
    #pragma omp parallel shared(Lx,Rx,threadParams) private(l)
    {
      
      #pragma omp for
      for( int k = 0; k < detected_num; k++ )
      {
          l = lines[k];
          if((abs(l[1] - l[3]) > lane_line_thresh)){// && DISP_LINES){
        
            if((l[0] > Lx) && (l[0] < (in_size.width/2))){
              threadParams->left_lane = l;
              Lx = l[0];
            }
            else if((l[0] < Rx) && (l[0] > (in_size.width/2))){

              threadParams->right_lane = l;
              Rx = l[0];
            }        
       
          }
      }
    }
       
    threadParams->left_lane[0] += threadParams->lane_roi.x;
    threadParams->left_lane[1] += threadParams->lane_roi.y;
    threadParams->left_lane[2] += threadParams->lane_roi.x;
    threadParams->left_lane[3] += threadParams->lane_roi.y;
    
    threadParams->right_lane[0] += threadParams->lane_roi.x;
    threadParams->right_lane[1] += threadParams->lane_roi.y;
    threadParams->right_lane[2] += threadParams->lane_roi.x;
    threadParams->right_lane[3] += threadParams->lane_roi.y;
    
    
    
    lane_check(threadParams->left_lane,threadParams->right_lane);

    return NULL;

}

void *frame_proc_thread(void *frame_thread_params){



  frameParams_t *frameParams = (frameParams_t *)frame_thread_params;


  stopParams_t stopParams;
  laneParams_t laneParams;


  
  
  laneParams.lane_frame = frameParams->frame_gray(frameParams->lane_roi);    
  laneParams.lane_roi = frameParams->lane_roi;  
  laneParams.in_size = frameParams->in_size;
  laneParams.disp = frameParams->disp;


  stopParams.stop_frame = frameParams->frame_gray(frameParams->stop_roi);
  stopParams.stop_roi = frameParams->stop_roi;  
  stopParams.in_size = frameParams->in_size;
  stopParams.focal = frameParams->focal;
  stopParams.hog = frameParams->hog;
  stopParams.disp = frameParams->disp; 

  pthread_t threads[NUM_FEAT];

  pthread_create(&threads[LANE_DETECT],   // pointer to thread descriptor
                 NULL,     // use default attributes
                 lane_detect, // thread function entry point
                 (void *)&(laneParams) // parameters to pass in
                );

  pthread_create(&threads[STOP_DETECT],   // pointer to thread descriptor
                 NULL,     // use default attributes
                 stop_detect, // thread function entry point
                 (void *)&(stopParams) // parameters to pass in
                );

  for(int k=0;k<NUM_FEAT;k++)
         pthread_join(threads[k], NULL);

        //  cout << "detection complete" << endl;

  
  // lanes
  Vec4i left_lane, right_lane;
  
  //if(threadParams[LANE_DETECT].left_lane == 0);
  //  cout << "left empty" << endl;
  left_lane = laneParams.left_lane;
  
  //if(threadParams[LANE_DETECT].right_lane == 0);
  //  cout << "right empty" << endl;
  right_lane = laneParams.right_lane;
  
  if(DISP_LINES){
    
  
    line( frameParams->frame, Point(left_lane[0], left_lane[1]), Point(left_lane[2], left_lane[3]), Scalar(0,0,255), 3, LINE_AA);
    line( frameParams->frame, Point(right_lane[0], right_lane[1]), Point(right_lane[2], right_lane[3]), Scalar(0,0,255), 3, LINE_AA);
  }
  
  // stop signs
  // int dist;
  //cout << threadParams[STOP_DETECT].stop_signs.size() << endl;
  if(DISP_STOP){
    for(uint16_t k = 0; k<stopParams.stop_signs.size();k++){
        rectangle(frameParams->frame, stopParams.stop_signs[k], Scalar(49,49,255), 5);
    }
  }

  return NULL;

}


static const string keys =  "{ help h | | print help message }"
                            "{ camera c | 0 | capture video from camera (device index starting from 0) }"
                            "{ video v | | use video as input }"
                            "{ display d | false | display intermediate steps }"
                            "{ xml x | stop_sign_classifier_2.xml | model xml file, default is a pre-trianed model }"
                            "{ hog g | false | use hog xml }"
                            "{ focal f| 4 | focal length of camera, assumes logitech C270 otherwise}"
                            "{ store s | | save file for output video }";
                            //"{ store s |my_vid.mp4| save file for output video }";

/*
*    starting point of application
*/
int main( int argc, char** argv )
{
    
    VideoCapture input;
    VideoWriter  output_vid;
    //VideoCapture cap;
    //int frame_tot = 0; 
    int frame_cnt = 0;
    bool OUT_WRITE = 0;
    //CascadeClassifier stop_cascade;
    //Size in_size;
    //mode = false;
    // check for cam or video input
    CommandLineParser parser(argc, argv, keys);
    if (parser.has("help"))
    {
      parser.printMessage();
      return 0;
    }
    
    int camera = parser.get<int>("camera");
    string file = parser.get<string>("video");
    bool disp = parser.get<bool>("display");
    string xml = parser.get<string>("xml");
    bool hog = parser.get<bool>("hog");
    float focal = parser.get<float>("focal");
    string out_file = parser.get<string>("store");
    
    
    if (!parser.check())
    {
      parser.printErrors();
      return 1;
    }
    
    // check video file or cam
    if (file.empty())
    {
      input.open(camera);
      //cap.set(CAP_PROP_FRAME_WIDTH, WIDTH);
      //cap.set(CAP_PROP_FRAME_HEIGHT, HEIGHT);
    }
    else
    {
      file = samples::findFileOrKeep(file);
      input.open(file);
    
    }
    
    // confirm input can be opened
    if (!input.isOpened())
    {
      cout << "Can not open video stream: '" << (file.empty() ? "<camera>" : file) << "'" << endl;
      return 2;
    }
    
    // load xml
    if(hog){
      if(!stop_hog.load(xml)){
        cout << "\nERROR: failed to load xml file\n" << endl;
        exit(SYSTEM_ERROR);
      }
    }
    else{
      if(!stop_cascade.load(xml)){
        cout << "\nERROR: failed to load xml file\n" << endl;
        exit(SYSTEM_ERROR);
      }
    }
    
    // find height and width of video 
    in_size = Size( (int)input.get(CAP_PROP_FRAME_WIDTH), (int)input.get(CAP_PROP_FRAME_HEIGHT));
    
    // create output file if input given
    if(!out_file.empty()){
      int in_fps = input.get(CAP_PROP_FPS);
      int fourcc_code_out = VideoWriter::fourcc('m','p','4','v');
      output_vid.open(out_file, fourcc_code_out, in_fps, in_size,true);
      OUT_WRITE = 1;
    }
    
    // create windows
    if(disp){
      namedWindow("original",WINDOW_NORMAL);
      namedWindow("grayscale",WINDOW_NORMAL);
    }
    namedWindow(final_out,WINDOW_NORMAL);

    Mat frame, frame_gray;
    
    // DETECTION ROI
    Rect lane_roi(0 + lane_roi_width_offset,                  // x starting point
                 in_size.height/2 + lane_roi_top_offset,      // y start
                 in_size.width - lane_roi_width_offset,       // width
                 in_size.height/2 - lane_roi_bottom_offset);  // height
                 
    Rect stop_roi(in_size.width/2 + stop_roi_left_offset,         // x starting point
                 0 + stop_roi_top_offset,                      // y starting point
                 in_size.width/2 - stop_roi_left_offset,        // width
                 in_size.height/2 - stop_roi_bottom_offset);   // height
                 
    
    // DISPLAY TOGGLES
    // bool DISP_LINES = false;
    // bool DISP_STOP = false;
    
    //int TPR = 0;
    //int FPR = 0;

    
///////////////////////////////// begin frame processing /////////////////////////////////    
    
    // cpu_set_t cpuset;
    // int core_id;
    // int max_prio, rc;
    
    struct timespec start,start_frame, end_frame;
    
    syslog(LOG_NOTICE, "general: ********************************* APPLICATION START ****************************");
    
    // start time for 1 minute comparison
    clock_gettime(CLOCK_REALTIME, &start);
    
    // start time for frame
    clock_gettime(CLOCK_REALTIME, &start_frame);

  while(1){
  //for(int i=0; i<NUM_FEAT; i++)
  //{
  
  
      // read in frame
    if(!input.read(frame)){
      cout << "no more frames, video over press any key to continue" << endl;
      //while(1){
      //  if((winInput = waitKey(0)) == ESCAPE_KEY){ 
      //    break;
      //  }
      //}
      waitKey(0);
      break;
    } 
    
    if(disp)
      imshow("original", frame);
    
    // grayscale and blur
    cvtColor( frame, frame_gray, COLOR_BGR2GRAY );
    GaussianBlur(frame_gray, frame_gray, Size(9,9), 2, 2);

    if(disp)
      imshow("grayscale", frame_gray );
    
    
    //////////////////////////// thread stuff for lane detection /////////////////////////////////    
    // cout << sizeof(frameParams_t) << endl;
    // frameParams_t *frameParams = (frameParams_t*)malloc(sizeof(frameParams_t));
    frameParams_t *frameParams = new frameParams_t;
    // threadParams->frameIdx = LANE_DETECT;

    if(frameParams == NULL){
      cout << "unable to malloc memory for frame params" << endl;
      exit(SYSTEM_ERROR);
    }

    // cout << "setting frame params" << endl;

    frameParams->frame = frame.clone();
    frameParams->frame_gray = frame_gray.clone();
    
    // threadParams.lane_frame = frame_gray(lane_roi);
    
    frameParams->lane_roi = lane_roi;
    
    frameParams->in_size = in_size;


    /////////////////////////// thread stuff for stop detection /////////////////////////////////  
    
    // threadParams.stop_frame = frame_gray(stop_roi);
    
    frameParams->stop_roi = stop_roi;
    
    //threadParams[STOP_DETECT].in_size = in_size;
    
    frameParams->focal = focal;
    
    frameParams->disp = disp;
    frameParams->hog = hog;
    
    // cout << "thread params set" << endl;

    pthread_create(&frameParams->thread_ID,   // pointer to thread descriptor
                   NULL,     // use default attributes
                   frame_proc_thread, // thread function entry point
                   (void *)(frameParams) // parameters to pass in
                  );


    ///////////////////////////////////////////////////////////////////////////////////////////////    

    // join and wait for all threads to finish
    // for(int k=0;k<NUM_FEAT;k++)
    //     pthread_join(threads[k], NULL);
    pthread_join(frameParams->thread_ID, NULL);

    ///////////////////////////// display results from detection ///////////////////////////////// 

    // // lanes
    // Vec4i left_lane, right_lane;
    
    // //if(threadParams[LANE_DETECT].left_lane == 0);
    // //  cout << "left empty" << endl;
    // left_lane = threadParams[LANE_DETECT].left_lane;
    
    // //if(threadParams[LANE_DETECT].right_lane == 0);
    // //  cout << "right empty" << endl;
    // right_lane = threadParams[LANE_DETECT].right_lane;
    
    // if(DISP_LINES){
      
    
    //   line( frame, Point(left_lane[0], left_lane[1]), Point(left_lane[2], left_lane[3]), Scalar(0,0,255), 3, LINE_AA);
    //   line( frame, Point(right_lane[0], right_lane[1]), Point(right_lane[2], right_lane[3]), Scalar(0,0,255), 3, LINE_AA);
    // }
    
    // // stop signs
    // // int dist;
    // //cout << threadParams[STOP_DETECT].stop_signs.size() << endl;
    // if(DISP_STOP){
    //   for(uint16_t k = 0; k<threadParams[STOP_DETECT].stop_signs.size();k++){
    //       rectangle(frame, threadParams[STOP_DETECT].stop_signs[k], Scalar(49,49,255), 5);
    //   }
    // }
    // frame display
    imshow(final_out,frameParams->frame);

    delete frameParams;
    
    frame_cnt++;
    
    // end time of current frame
    clock_gettime(CLOCK_REALTIME, &end_frame);
    
    frame_time_calc(&start,&start_frame,&end_frame,&frame_cnt);
    
    //if(threadParams[STOP_DETECT].stop_signs.size()){
      
    //  winInput = waitKey(0);
      
    //  if (winInput == 'y' ){
    //    TPR++;
    //  }
    //  else if(winInput == 'n' ){
    //    FPR++;
    //  }
    //}
    
    // write to output file
    if(OUT_WRITE){
      output_vid.write(frame);
    }
    
    
    //////////////////////////////////////////////////////////////////////////////////////////////  

    if ((winInput = waitKey(1)) == ESCAPE_KEY)
    //if ((winInput = waitKey(0)) == ESCAPE_KEY)
    {
        break;
    }
    else if(winInput == 'L' || winInput == 'l'){
      DISP_LINES ^= 1;
      cout << "lines toggled: " << (DISP_LINES ? "on" : "off") << endl;
    }
    else if(winInput == 'S' || winInput == 's'){
      DISP_STOP ^= 1;
      cout << "stop signs toggled: " << (DISP_STOP ? "on" : "off") << endl;
    }
    else if(winInput == 32){
      cout << "video stopped, press any key to resume" << endl;
      waitKey();
    }
  }
    

  
  //cout << "true positive: " << TPR << endl;
  //cout << "false positive: " << FPR << endl;
  //cout << "total detects: " << (TPR + FPR) << endl;
    return 0;
}
