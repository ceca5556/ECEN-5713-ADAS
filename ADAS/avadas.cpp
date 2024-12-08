#include <iostream>
#include <opencv2/opencv.hpp>
// #include <opencv2/highgui/highgui.hpp>

// #include <opencv2/imgproc.hpp>

// #include <opencv2/core/core.hpp>
// #include <opencv2/core/base.hpp>
// #include <opencv2/core/types.hpp>

// #include "opencv2/objdetect.hpp"
// #include <opencv2/videoio.hpp> // Video write

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
#include <signal.h>
#include <errno.h>

// #include "lanelib.hpp"
#include "queue.h"

using namespace cv; using namespace std;


#define ESCAPE_KEY (27)
#define SYSTEM_ERROR (-1)

#define NUM_FEAT (2)
#define NUM_CPUS (4)
#define SCHED_POLICY SCHED_FIFO

#define LANE_DETECT (0)
#define STOP_DETECT (1)

#define SIZE_240P (Size(320,240))

#define SECOND   (1)
#define MINUTE   (60)

#define THREAD_LIMIT  (500)
// #define MEMORY_FIX

typedef struct frameParams_s
{

  pthread_t thread_ID;
  int frameIdx;
  Mat frame;
  Mat frame_gray;
  int *proc_frame_cnt;

  Rect lane_roi;

  Rect stop_roi;
  int focal;
  bool hog;
  
  Size in_size;
  Size des_size;

  bool disp;
  bool complete;

  pthread_mutex_t *proc_frame_mutex;

  TAILQ_ENTRY(frameParams_s) next_frame;

} frameParams_t;

typedef struct
{

  Mat lane_frame;
  Vec4i left_lane;
  Vec4i right_lane;
  Rect lane_roi;
  
  
  Size in_size;

  bool disp;
  
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
  
} stopParams_t;

typedef struct
{
  // uint16_t total_frames;
  bool save_to_video;
  bool *done;

  // VideoWriter output_vid;
  int *disp_frame_cnt;
  int in_fps;
  int fourcc_code_out;
  Size des_size;
  string out_file;

  pthread_mutex_t *disp_frame_mutex;

}displayParams_t;

typedef struct{

  pthread_mutex_t timer_mutex;
  int call_count;

  pthread_mutex_t *proc_frame_mutex;
  int tot_proc_frame_cnt;
  int *proc_frame_cnt;

  pthread_mutex_t *disp_frame_mutex;
  int tot_disp_frame_cnt;
  int *disp_frame_cnt;

}timerParams_t;


TAILQ_HEAD(frame_head, frameParams_s) frame_list;


int lowThreshold = 17;
const int max_lowThreshold = 200;
const int ratio_canny = 4;
const int kernel_size = 3;


// int numberOfProcessors = NUM_CPUS;

pthread_attr_t sched_attr;
struct sched_param sched_param;

char winInput;

bool DISP_LINES = true;
bool DISP_STOP = true;


const string final_out = "output";

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


static float sensor_width = 4.55; // milimeters

#ifdef MEMORY_FIX
int frames_in_list = 0;
pthread_mutex_t list_num_mutex;
#endif


/* 
   Timer thread signal function, resets 
*/
static void frame_time_calc_thread(union sigval sigval){

  int rc= 0;
  int temp_proc_frames = 0;
  int temp_disp_frames = 0;


  timerParams_t *timerParams = (timerParams_t*) sigval.sival_ptr;

  /////////////////////// lock /////////////////////
    // lock proc frame
    rc = pthread_mutex_lock(timerParams->proc_frame_mutex);
    if(rc != 0){
        syslog(LOG_ERR,"ERRROR fps calc thread: proc mutex lock function failed: %s",strerror(rc));
        exit(SYSTEM_ERROR);
    }
    // lock disp frame
    rc = pthread_mutex_lock(timerParams->disp_frame_mutex);
    if(rc != 0){
        syslog(LOG_ERR,"ERRROR fps calc thread: display mutex lock function failed: %s",strerror(rc));
        exit(SYSTEM_ERROR);
    }

  /////////////////////// perform needed operations /////////////////////
  // timerParams->call_count++;
  temp_proc_frames = *timerParams->proc_frame_cnt;
  temp_disp_frames = *timerParams->disp_frame_cnt;

  *timerParams->proc_frame_cnt = 0;
  *timerParams->disp_frame_cnt = 0;
  // cout << "timer thread called: " << timerParams->call_count << "times" << endl;

  /////////////////////// unlock /////////////////////
    //unlock disp
    rc = pthread_mutex_unlock(timerParams->disp_frame_mutex);
    if(rc != 0){
        syslog(LOG_ERR,"ERRROR fps calc thread: display mutex lock function failed: %s",strerror(rc));
        exit(SYSTEM_ERROR);
    }

    // unlock proc
    rc = pthread_mutex_unlock(timerParams->proc_frame_mutex);
    if(rc != 0){
        syslog(LOG_ERR,"ERRROR fps calc thread: proc mutex lock function failed: %s",strerror(rc));
        exit(SYSTEM_ERROR);
    }

  /////////////////////// other calculations /////////////////////
  syslog(LOG_NOTICE,"P-FPS (processing): %d frames in 1 second",temp_proc_frames);
  syslog(LOG_NOTICE,"D-FPS (displaying): %d frames in 1 second",temp_disp_frames);


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
  
  // if(disp){
  //   namedWindow("canny",WINDOW_NORMAL);
  //   imshow("canny", *canny_frame );
  // }
  
  if (winInput == '.')// || winInput == 'T')
  {
    lowThreshold += 5;
    if(lowThreshold > max_lowThreshold){
      lowThreshold = 0;
    }
  }
  else if(winInput == ','){
    lowThreshold -= 5;
    if(lowThreshold < 0){
      lowThreshold = max_lowThreshold;
    }
  }

}

/*
*    stop sign distance estimation
*/
int stop_dist(Rect rectangle, float f,Size in_size){

  int sign_width = 750; // in mm
  int pixels = rectangle.width; // pixel width of detected 
  
  f = f*in_size.width/sensor_width; // mm * pixel/mm = pixels
  
  float dist = (sign_width*f)/pixels; // mm * pixel / pixel
  
  return dist;

}

/*
*    stop sign detection function performed by the threads
*/
void *stop_detect(void *threadp){

    stopParams_t *threadParams = (stopParams_t *)threadp;
    vector<Rect> stop_signs;
    float dist;
    
    // if(threadParams->disp){
    // //   namedWindow("original",WINDOW_NORMAL);
    //    namedWindow("stop_roi",WINDOW_NORMAL);
       
    //    imshow("stop_roi",threadParams->stop_frame);
    // }
     
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
          
          dist = stop_dist(threadParams->stop_signs[k],threadParams->focal,threadParams->in_size);
          syslog(LOG_NOTICE, "STOP_SIGN: stop sign #%d/%d distance: %02f mm", k+1,detected_num, dist);
        
      }
    }

    return NULL;

}

/*
*   lane location check
*/
void lane_check(Vec4i left_lane, Vec4i right_lane,Size in_size){

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
    
    // if(threadParams->disp){
    //    namedWindow("lane_roi",WINDOW_NORMAL);
       
    //    imshow("lane_roi",threadParams->lane_frame);
    //  }
    
    Mat canny_frame;//, roi_frame;
    vector<Vec4i> lines;
    
    //roi_frame = threadParams->frame(
    canny_edge(&threadParams->lane_frame,&canny_frame,threadParams->disp);//,50,200,3);
    HoughLinesP(canny_frame, lines, 1, CV_PI/180, 50, 50, 10 );
    
    
    //Vec4i left_lane, right_lane;
    int Lx = 0;
    int Rx = threadParams->in_size.width;
    Vec4i l;
    
    
    int detected_num = lines.size();
    #pragma omp parallel shared(Lx,Rx,threadParams) private(l)
    {
      
      #pragma omp for
      for( int k = 0; k < detected_num; k++ )
      {
          l = lines[k];
          if((abs(l[1] - l[3]) > lane_line_thresh)){// && DISP_LINES){
        
            if((l[0] > Lx) && (l[0] < (threadParams->in_size.width/2))){
              threadParams->left_lane = l;
              Lx = l[0];
            }
            else if((l[0] < Rx) && (l[0] > (threadParams->in_size.width/2))){

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
    
    
    
    lane_check(threadParams->left_lane,threadParams->right_lane,threadParams->in_size);

    return NULL;

}

void *frame_proc_thread(void *frame_thread_params){




  frameParams_t *frameParams = (frameParams_t *)frame_thread_params;


  stopParams_t stopParams;
  laneParams_t laneParams;

  int rc= 0;

  
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
  left_lane = laneParams.left_lane;
  right_lane = laneParams.right_lane;
  
  if(DISP_LINES){
    
  
    line( frameParams->frame, Point(left_lane[0], left_lane[1]), Point(left_lane[2], left_lane[3]), Scalar(0,0,255), 3, LINE_AA);
    line( frameParams->frame, Point(right_lane[0], right_lane[1]), Point(right_lane[2], right_lane[3]), Scalar(0,0,255), 3, LINE_AA);
  }
  
  // stop signs
  if(DISP_STOP){
    for(uint16_t k = 0; k<stopParams.stop_signs.size();k++){
        rectangle(frameParams->frame, stopParams.stop_signs[k], Scalar(49,49,255), 5);
    }
  }

  resize(frameParams->frame,frameParams->frame,frameParams->des_size);


  // call mutex lock -> lock frame time calculation for each frame
  rc = pthread_mutex_lock(frameParams->proc_frame_mutex);
  if(rc != 0){
      syslog(LOG_ERR,"ERRROR fps calculation: mutex lock function failed: %s",strerror(rc));
      exit(SYSTEM_ERROR);
  }

  *frameParams->proc_frame_cnt = *frameParams->proc_frame_cnt + 1;

  rc = pthread_mutex_unlock(frameParams->proc_frame_mutex);
  if(rc != 0){
      syslog(LOG_ERR,"ERRROR fps calculation: mutex unlock function failed: %s",strerror(rc));
      exit(SYSTEM_ERROR);
  }

  frameParams->complete = true;

  pthread_exit(NULL);

  return NULL;

}

void *frame_disp_thread(void *display_Params){


  int rc = 0;

  displayParams_t *dispParams = (displayParams_t*)display_Params;
  // struct timespec end_frame;
  uint16_t current_frames = 0;

  VideoWriter  output_vid;

  // check save to video parameter
  if(dispParams->save_to_video)
    output_vid.open(dispParams->out_file, dispParams->fourcc_code_out, dispParams->in_fps, dispParams->des_size,true); // open file

  while((!TAILQ_EMPTY(&frame_list)) || (!*dispParams->done)){

    // grab frame
    frameParams_t *frameParams = TAILQ_FIRST(&frame_list);
    // cout << "frame pointer: " << frameParams << endl;

    // ensure not null
    if(frameParams != NULL){
      
      // check if processing complete
      if(frameParams->complete){
        // join thread
        pthread_join(frameParams->thread_ID, NULL);

        ///////////////////////////// display results from detection ///////////////////////////////// 
        // output 
        if(dispParams->save_to_video){
          output_vid.write(frameParams->frame);
        }
        else{
          imshow(final_out,frameParams->frame);
        }

        // remove from list
        TAILQ_REMOVE(&frame_list, frameParams, next_frame);

       #ifdef MEMORY_FIX
        pthread_mutex_lock(&list_num_mutex);
        frames_in_list--;
        pthread_mutex_unlock(&list_num_mutex);
       #endif

        // unallocate
        delete frameParams;

        rc = pthread_mutex_lock(dispParams->disp_frame_mutex);
        if(rc != 0){
            syslog(LOG_ERR,"ERRROR display thread: display mutex lock function failed: %s",strerror(rc));
            exit(SYSTEM_ERROR);
        }

        // frame complete tracking methods
        *dispParams->disp_frame_cnt = *dispParams->disp_frame_cnt +1;

        rc = pthread_mutex_unlock(dispParams->disp_frame_mutex);
        if(rc != 0){
            syslog(LOG_ERR,"ERRROR display thread: display mutex unlock function failed: %s",strerror(rc));
            exit(SYSTEM_ERROR);
        }


        current_frames++;

        // syslog(LOG_NOTICE, "Display: frame #%d completed", current_frames);
      }
    }
    
    // visual indicator for user
    if(!(current_frames % 50) && dispParams->save_to_video)
      cout << current_frames << "displayed" << endl;

    //////////////////////////////////////////////////////////////////////////////////////////////  

      if ((winInput = waitKey(1)) == ESCAPE_KEY)
      //if ((winInput = waitKey(0)) == ESCAPE_KEY)
      {
        *dispParams->done = true;
        cout << "ESC pressed, terminating program" << endl;
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

  pthread_exit(NULL);
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

  int rc = 0; 
  //////////////////// declare timer variables /////////////
    timerParams_t timerParams;
    struct sigevent sev;
    timer_t timerid;
    int clock_id;
    struct itimerspec timer;

    memset(&timerParams,0,sizeof(timerParams_t));
    memset(&sev,0,sizeof(struct sigevent));
    memset(&timer, 0, sizeof(struct itimerspec));

  //////////////////// declare opencv variables ////////////  
    VideoCapture input;

    displayParams_t dispParams;
    pthread_t disp_thread_ID;

    int proc_frame_cnt = 0;
    int disp_frame_cnt = 0;

    pthread_mutex_t proc_frame_mutex;
    pthread_mutex_t disp_frame_mutex;

    bool OUT_WRITE = false;
    int in_fps = 0;
    int fourcc_code_out = 0;
    Size desired_size = SIZE_240P;
    Size in_size;

    uint16_t total_frames = 0;
    bool done = false;

  //////////////////// initialize timer variables ////////////  

    /* the following code was based off of the example at:
        https://github.com/cu-ecen-aeld/aesd-lectures/blob/master/lecture9/timer_thread.c
    */
    clock_id = CLOCK_MONOTONIC;
    sev.sigev_notify = SIGEV_THREAD;
    sev.sigev_value.sival_ptr = &timerParams;
    sev.sigev_notify_function = frame_time_calc_thread;

    timer.it_interval.tv_sec = 1;
    timer.it_interval.tv_nsec = 0;

    if ( timer_create(clock_id,&sev,&timerid) != 0 ) {
      // printf("Error creating timer\n",errno,strerror(errno));
      cout << "ERROR: could not create timer: " << strerror(errno) << endl;
      exit(SYSTEM_ERROR);
    } 


  //////////////////// check input arguments ////////////  
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
  
  //////////////////// get input propeties ////////////  
    // find properties of input
    in_fps = input.get(CAP_PROP_FPS);
    in_size = Size( (int)input.get(CAP_PROP_FRAME_WIDTH), (int)input.get(CAP_PROP_FRAME_HEIGHT)); //height and width of video 
    total_frames = input.get(CAP_PROP_FRAME_COUNT); // frame count

  //////////////////// create output file if necessary ////////////  
    // create output file if input given
    if(!out_file.empty()){
      fourcc_code_out = VideoWriter::fourcc('m','p','4','v');
      OUT_WRITE = true;
      desired_size = in_size;

    }

  //////////////////// print important propeties //////////// 
    cout << endl << "************** properties **************" << endl; 
    cout << "total frames: " << total_frames << endl;
    cout << "input fps: " << in_fps << endl;
    cout << "input size: " << in_size << endl;
    cout << "output size: "<< desired_size << endl << endl;



  
  //////////////////// create display windows ////////////  
    // if(disp){
    //   namedWindow("original",WINDOW_NORMAL);
    //   namedWindow("grayscale",WINDOW_NORMAL);
    // }
    namedWindow(final_out,WINDOW_NORMAL);

  
  //////////////////// ROI calculation ////////////  

    // DETECTION ROI
    Rect lane_roi(0 + lane_roi_width_offset,                  // x starting point
                  in_size.height/2 + lane_roi_top_offset,      // y start
                  in_size.width - lane_roi_width_offset,       // width
                  in_size.height/2 - lane_roi_bottom_offset);  // height
                  
    Rect stop_roi(in_size.width/2 + stop_roi_left_offset,         // x starting point
                  0 + stop_roi_top_offset,                      // y starting point
                  in_size.width/2 - stop_roi_left_offset,        // width
                  in_size.height/2 - stop_roi_bottom_offset);   // height
                  

    
    //int TPR = 0;
    //int FPR = 0;

  
  ///////////////////////////////// initialize mutex /////////////////////////////////    
  
    
    // rc = pthread_mutex_init(&time_mutex,NULL);
    // if(rc != 0){

    //     syslog(LOG_ERR, "ERROR: could not succesfully initialize mutex, error number: %d", rc);
    //     return SYSTEM_ERROR;

    // }
   #ifdef MEMORY_FIX
    rc = pthread_mutex_init(&list_num_mutex,NULL);
    if(rc != 0){

        syslog(LOG_ERR, "ERROR: could not succesfully initialize mutex, error number: %d", rc);
        return SYSTEM_ERROR;

    }
   #endif

    rc = pthread_mutex_init(&timerParams.timer_mutex,NULL);
    if(rc != 0){

        syslog(LOG_ERR, "ERROR: could not succesfully initialize timer mutex, error number: %d", rc);
        return SYSTEM_ERROR;

    }

    rc = pthread_mutex_init(&proc_frame_mutex,NULL);
    if(rc != 0){

        syslog(LOG_ERR, "ERROR: could not succesfully initialize processing frame mutex, error number: %d", rc);
        return SYSTEM_ERROR;

    }

    rc = pthread_mutex_init(&disp_frame_mutex,NULL);
    if(rc != 0){

        syslog(LOG_ERR, "ERROR: could not succesfully initialize display frame mutex, error number: %d", rc);
        return SYSTEM_ERROR;

    }

  ///////////////////////////////// initialize and create other threads ///////////////////////////////// 
    // initialize linked list
    TAILQ_INIT(&frame_list);

    // initialise display parameters
    dispParams.done = &done;
    dispParams.save_to_video = OUT_WRITE;
    dispParams.in_fps = in_fps;
    dispParams.fourcc_code_out = fourcc_code_out;
    dispParams.des_size = desired_size;
    dispParams.out_file = out_file;
    dispParams.disp_frame_cnt = &disp_frame_cnt;
    dispParams.disp_frame_mutex = &disp_frame_mutex;
    pthread_create(&disp_thread_ID,   // pointer to thread descriptor
                    NULL,     // use default attributes
                    frame_disp_thread, // thread function entry point
                    (void *)(&dispParams) // parameters to pass in
                    );

    // finish intializing timer parameters 
    timerParams.disp_frame_cnt = &disp_frame_cnt;
    timerParams.proc_frame_cnt = &proc_frame_cnt;

    timerParams.disp_frame_mutex = &disp_frame_mutex;
    timerParams.proc_frame_mutex = &proc_frame_mutex;                
  
  ///////////////////////////////// begin frame processing and timer ///////////////////////////////// 
    syslog(LOG_NOTICE, "general: ********************************* APPLICATION START ****************************");

    Mat frame, frame_gray;

    clock_gettime(clock_id, &timer.it_value);
    
    timer.it_value.tv_sec++;

    if(timer_settime(timerid, TIMER_ABSTIME, &timer, NULL) != 0){
      cout << "ERROR: Could not set timer: " << strerror(errno);
      exit(SYSTEM_ERROR);
    }

    while(!done){
    
     #ifdef MEMORY_FIX
      pthread_mutex_lock(&list_num_mutex);

      if(frames_in_list > THREAD_LIMIT){
        pthread_mutex_unlock(&list_num_mutex);
        continue;
      }
      
      pthread_mutex_unlock(&list_num_mutex);
     #endif
    
        // read in frame
      if(!input.read(frame)){
        cout << "all frames have finished processing, displaying remaining frames" << endl;
        done = true;
        break;
      } 
      
      // if(disp)
      //   imshow("original", frame);
      
      // grayscale and blur
      cvtColor( frame, frame_gray, COLOR_BGR2GRAY );
      GaussianBlur(frame_gray, frame_gray, Size(9,9), 2, 2);

      // if(disp)
      //   imshow("grayscale", frame_gray );
      
      //////////////////////////// initialize frame processing thread parameters per frame /////////////////////////////////    
      frameParams_t *frameParams = new frameParams_t;
      // threadParams->frameIdx = LANE_DETECT;

      if(frameParams == NULL){
        cout << "unable to malloc memory for frame params" << endl;
        exit(SYSTEM_ERROR);
      }

      // cout << "setting frame params" << endl;

      frameParams->frame = frame.clone();
      frameParams->frame_gray = frame_gray.clone();

      frameParams->in_size = in_size;
      frameParams->des_size = desired_size;

      frameParams->proc_frame_cnt = &proc_frame_cnt;
      frameParams->proc_frame_mutex = &proc_frame_mutex;
      
      // threadParams.lane_frame = frame_gray(lane_roi);
      
      frameParams->lane_roi = lane_roi;

      /////////////////////////// thread stuff for stop detection /////////////////////////////////  
      
      frameParams->stop_roi = stop_roi;      
      frameParams->focal = focal;
      
      frameParams->disp = disp;
      frameParams->hog = hog;
      frameParams->complete = false;
      
      // create thread
      pthread_create(&frameParams->thread_ID,   // pointer to thread descriptor
                    NULL,     // use default attributes
                    frame_proc_thread, // thread function entry point
                    (void *)(frameParams) // parameters to pass in
                    );

      // cout << "created frame thread with ID: " << frameParams->thread_ID << endl;

      // insert thread to list
      TAILQ_INSERT_TAIL(&frame_list, frameParams,next_frame);


     #ifdef MEMORY_FIX
      pthread_mutex_lock(&list_num_mutex);
      frames_in_list++;
      pthread_mutex_unlock(&list_num_mutex);
     #endif
    }
    
    // wait for display thread
    pthread_join(disp_thread_ID, NULL);

    //begin clean up
    cout << "cleaning up...." << endl;

    // delete timer
    if(timer_delete(timerid) != 0) {
      cout << "ERROR: could not delete timer: " << strerror(errno) << endl;
    }    

    // check if list empty
    if(!TAILQ_EMPTY(&frame_list))
      cout << "finishing remaining frame processing threads..." << endl;

    // stop, remove, and cleanup remaining processing threads
    while(!TAILQ_EMPTY(&frame_list)){
      frameParams_t *frameParams = TAILQ_FIRST(&frame_list);

      if(frameParams != NULL){
        pthread_join(frameParams->thread_ID, NULL);

        TAILQ_REMOVE(&frame_list, frameParams, next_frame);

        delete frameParams;
      }

    }

    // input.release();
    destroyAllWindows();

    cout << "end of application" << endl;
    //cout << "true positive: " << TPR << endl;
    //cout << "false positive: " << FPR << endl;
    //cout << "total detects: " << (TPR + FPR) << endl;
      return 0;
}
