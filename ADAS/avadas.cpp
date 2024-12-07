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
#include <signal.h>
#include <errno.h>

// #include "lanelib.hpp"
#include "queue.h"

using namespace cv; using namespace std;
// double alpha=1.0;  int beta=10;  /* contrast and brightness control */


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

// Size in_size;

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

// static int fps_sum = 0;
// static int call_count = 0;
// static int frame_num = 0;
static float sensor_width = 4.55; // milimeters

// Mat global_frame;
// bool global_check = false;

struct timespec start_60S,start_1S;
pthread_mutex_t time_mutex;

int proc_frame_num;
int disp_frame_num;

int type_proc = 1;
int type_disp = 2;


/**************************************************************
*
*  @brief dt_sec: calculates the difference between two timespec
*         structs in seconds
*
*  @param[in] start_60S
*        - pointer to starting timespec struct (used for 60 sec calculation)
*
*  @param[in] start_1S
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
// static void frame_time_calc(struct timespec *start,struct timespec *start_frame, struct timespec *stop,int *frame_num){
static void frame_time_calc(struct timespec *start_60S,struct timespec *start_1S, struct timespec *stop, int frame_type){
  
  // static int fps_sum;
  static int proc_fps_sum;
  static int disp_fps_sum;
  static int call_count;
  // static int frame_num;
  static int proc_frame;
  static int disp_frame;

  // int rc= 0;
  int full_time = stop->tv_sec - start_60S->tv_sec;
  int frme_time = stop->tv_sec - start_1S->tv_sec;

  // frame_num++;
  if(frame_type == type_proc){
    proc_frame++;
  }
  else if(frame_type == type_disp){
    disp_frame++;
  }

  if(frme_time >= SECOND){ // after 1 second
    // syslog(LOG_NOTICE, "FPS: %d frames in %d second", frame_num, frme_time);
    syslog(LOG_NOTICE, "P-FPS: %d frames in %d second", proc_frame, frme_time);
    syslog(LOG_NOTICE, "D-FPS: %d frames in %d second", disp_frame, frme_time);
    // fps_sum += frame_num;
    proc_fps_sum += proc_frame;
    disp_fps_sum += disp_frame;
    call_count++;
    // frame_num = 0;
    proc_frame = 0;
    disp_frame = 0;
    clock_gettime(CLOCK_MONOTONIC, start_1S);
  }
  
  // show when 60 seconds have passed
  if(full_time >= MINUTE){
    // int fps_avg = fps_sum/call_count;
    int proc_fps_avg = proc_fps_sum/call_count;
    int disp_fps_avg = disp_fps_sum/call_count;
    // syslog(LOG_NOTICE, "FPS: **************************** %d seconds have passed / %d avg FPS **********************", full_time,fps_avg);
    syslog(LOG_NOTICE, "FPS: **************************** %d seconds have passed / %d avg P-FPS (processing) / %d avg D-FPS (displaying) **********************", full_time,proc_fps_avg,disp_fps_avg);
    // fps_sum = 0;
    proc_fps_sum = 0;
    disp_fps_sum = 0;
    call_count = 0;
    clock_gettime(CLOCK_MONOTONIC, start_60S);
  }

}

// static void frame_time_calc_thread(union sigval sigval){
  
//   // static int fps_sum;
//   // static int call_count;
//   // static int frame_num;

//   int rc= 0;
//   // int temp_call_count = 0;
//   int temp_proc_frames = 0;
//   int temp_disp_frames = 0;
//   // int temp_tot_proc = 0;
//   // int temp_tot_disp = 0;

//   // static int call_count;
//   // static int tot_proc;
//   // static int tot_disp;
//   // int full_time = stop->tv_sec - start_60S->tv_sec;
//   // int frme_time = stop->tv_sec - start_1S->tv_sec;

//   timerParams_t *timerParams = (timerParams_t*) sigval.sival_ptr;

//   /////////////////////// lock /////////////////////
//     // lock timer
//     // rc = pthread_mutex_lock(&timerParams->timer_mutex);
//     // if(rc != 0){
//     //     syslog(LOG_ERR,"ERRROR fps calc thread: timer mutex lock function failed: %s",strerror(rc));
//     //     // cleanup(false,0,0,local_w_file_fd);
//     //     // raise(SIGINT);
//     //     exit(SYSTEM_ERROR);
//     // }
//     // lock proc frame
//     rc = pthread_mutex_lock(timerParams->proc_frame_mutex);
//     if(rc != 0){
//         syslog(LOG_ERR,"ERRROR fps calc thread: proc mutex lock function failed: %s",strerror(rc));
//         // cleanup(false,0,0,local_w_file_fd);
//         // raise(SIGINT);
//         exit(SYSTEM_ERROR);
//     }
//     // lock disp frame
//     rc = pthread_mutex_lock(timerParams->disp_frame_mutex);
//     if(rc != 0){
//         syslog(LOG_ERR,"ERRROR fps calc thread: display mutex lock function failed: %s",strerror(rc));
//         // cleanup(false,0,0,local_w_file_fd);
//         // raise(SIGINT);
//         exit(SYSTEM_ERROR);
//     }

//   /////////////////////// perform needed operations /////////////////////
//   // timerParams->call_count++;
//   temp_proc_frames = *timerParams->proc_frame_cnt;
//   temp_disp_frames = *timerParams->disp_frame_cnt;

//   // tot_proc += temp_proc_frames;
//   // tot_disp += temp_disp_frames;

//   // temp_tot_disp = 

//   *timerParams->proc_frame_cnt = 0;
//   *timerParams->disp_frame_cnt = 0;
//   // cout << "timer thread called: " << timerParams->call_count << "times" << endl;

//   /////////////////////// unlock /////////////////////
//     //unlock disp
//     rc = pthread_mutex_unlock(timerParams->disp_frame_mutex);
//     if(rc != 0){
//         syslog(LOG_ERR,"ERRROR fps calc thread: display mutex lock function failed: %s",strerror(rc));
//         // cleanup(false,0,0,local_w_file_fd);
//         // raise(SIGINT);
//         exit(SYSTEM_ERROR);
//     }

//     // unlock proc
//     rc = pthread_mutex_unlock(timerParams->proc_frame_mutex);
//     if(rc != 0){
//         syslog(LOG_ERR,"ERRROR fps calc thread: proc mutex lock function failed: %s",strerror(rc));
//         // cleanup(false,0,0,local_w_file_fd);
//         // raise(SIGINT);
//         exit(SYSTEM_ERROR);
//     }

//     // unlock timer
//     // rc = pthread_mutex_unlock(&timerParams->timer_mutex);
//     // if(rc != 0){
//     //     syslog(LOG_ERR,"ERRROR fps calc thread: mutex lock function failed: %s",strerror(rc));
//     //     // cleanup(false,0,0,local_w_file_fd);
//     //     // raise(SIGINT);
//     //     exit(SYSTEM_ERROR);
//     // }

//   /////////////////////// other calculations /////////////////////
//   syslog(LOG_NOTICE,"P-FPS (processing): %d frames in 1 second",temp_proc_frames);
//   syslog(LOG_NOTICE,"D-FPS (displaying): %d frames in 1 second",temp_disp_frames);


// }

// static void *frame_time_calc_1S(void){

//   static int tot_proc_frames;
//   static int tot_disp_frames;

//   int temp_proc = proc_frame_num;
//   int temp_disp = disp_frame_num;

//   syslog(LOG_NOTICE, "FPS: %d frames processed in 1 second", (temp_proc - tot_proc_frames));
//   syslog(LOG_NOTICE, "FPS: %d frames displayed in 1 second", (temp_disp - tot_disp_frames));
  
//   tot_proc_frames += temp_proc;
//   tot_disp_frames += temp_disp;
  
//   return NULL;
  
  
// }

// static void *frame_time_calc_60S(void){

//   static int tot_proc_frames;
//   static int tot_disp_frames;

//   int temp_proc = proc_frame_num;
//   int temp_disp = disp_frame_num;

//   syslog(LOG_NOTICE, "FPS: %d frames processed in 1 minute", (temp_proc - tot_proc_frames));
//   syslog(LOG_NOTICE, "FPS: %d frames displayed in 1 minute", (temp_disp - tot_disp_frames));
  
//   tot_proc_frames += temp_proc;
//   tot_disp_frames += temp_disp;
  
//   return NULL;
  
// }


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
int stop_dist(Rect rectangle, float f,Size in_size){

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
          
          dist = stop_dist(threadParams->stop_signs[k],threadParams->focal,threadParams->in_size);
          syslog(LOG_NOTICE, "STOP_SIGN: stop sign #%d/%d distance: %02f mm", k+1,detected_num, dist);
        
      }
    }

    return NULL;

}


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
    int Rx = threadParams->in_size.width;
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

  struct timespec end_frame;
  int rc= 0;

  // rc = pthread_setcancelstate(PTHREAD_CANCEL_ENABLE, NULL);
  // if(!rc){
  //   syslog(LOG_ERR, "ERROR thread ID: %ld: unable to enable cancel on thread ID: %s", frameParams->thread_ID, strerror(rc));
  // }
  // rc = pthread_setcanceltype(PTHREAD_CANCEL_ASYNCHRONOUS,NULL);
  // if(!rc){
  //   syslog(LOG_ERR, "ERROR thread ID: %ld: unable to set async cancel on thread ID: %s", frameParams->thread_ID, strerror(rc));
  // }
  
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

  // resize(frameParams->frame,frameParams->frame,Size(320,240));
  resize(frameParams->frame,frameParams->frame,frameParams->des_size);

  clock_gettime(CLOCK_MONOTONIC, &end_frame);

  // call mutex lock -> lock frame time calculation for each frame
  rc = pthread_mutex_lock(&time_mutex);
  // rc = pthread_mutex_lock(frameParams->proc_frame_mutex);
  if(rc != 0){
      syslog(LOG_ERR,"ERRROR fps calculation: mutex lock function failed: %s",strerror(rc));
      // cleanup(false,0,0,local_w_file_fd);
      // raise(SIGINT);
      exit(SYSTEM_ERROR);
  }

  frame_time_calc(&start_60S,&start_1S,&end_frame,type_proc);
  // *frameParams->proc_frame_cnt = *frameParams->proc_frame_cnt + 1;

  rc = pthread_mutex_unlock(&time_mutex);
  // rc = pthread_mutex_unlock(frameParams->proc_frame_mutex);
  if(rc != 0){
      syslog(LOG_ERR,"ERRROR fps calculation: mutex unlock function failed: %s",strerror(rc));
      // cleanup(false,0,0,local_w_file_fd);
      // raise(SIGINT);
      exit(SYSTEM_ERROR);
  }

  frameParams->complete = true;
  // global_frame = frameParams->frame.clone();

  // global_check = true;

  return NULL;

}

void *frame_disp_thread(void *display_Params){

  /* TODO:
        should pressing esc immediately stop display or should it display any remaining frames?
        currently doing latter, but if former:
            add code to go through and kill remaining frame threads
            delete frameParams
  */

  int rc = 0;

  displayParams_t *dispParams = (displayParams_t*)display_Params;
  struct timespec end_frame;
  uint16_t current_frames = 0;

  VideoWriter  output_vid;

  if(dispParams->save_to_video)
    output_vid.open(dispParams->out_file, dispParams->fourcc_code_out, dispParams->in_fps, dispParams->des_size,true);

  while((!TAILQ_EMPTY(&frame_list)) || (!*dispParams->done)){
  // while((!*dispParams->done)){

    // if(TAILQ_EMPTY(&frame_list))
    //       cout << "this list EMPTY" << endl;

    // if(*dispParams->done == true)
    //   cout << "why is it still here" << endl;
    // cout << "disp thread alive" << endl;
    frameParams_t *frameParams = TAILQ_FIRST(&frame_list);
    // cout << "frame pointer: " << frameParams << endl;


    if(frameParams != NULL){
    
      if(frameParams->complete){
        pthread_join(frameParams->thread_ID, NULL);

        ///////////////////////////// display results from detection ///////////////////////////////// 

        if(dispParams->save_to_video){
          output_vid.write(frameParams->frame);
        }
        else{
          imshow(final_out,frameParams->frame);
        }

        TAILQ_REMOVE(&frame_list, frameParams, next_frame);

        // if(TAILQ_EMPTY(&frame_list))
        //   cout << "this list EMPTY" << endl;

        delete frameParams;

        // rc = pthread_mutex_lock(dispParams->disp_frame_mutex);
        // if(rc != 0){
        //     syslog(LOG_ERR,"ERRROR display thread: display mutex lock function failed: %s",strerror(rc));
        //     exit(SYSTEM_ERROR);
        // }

        // *dispParams->disp_frame_cnt = *dispParams->disp_frame_cnt +1;

        // rc = pthread_mutex_unlock(dispParams->disp_frame_mutex);
        // if(rc != 0){
        //     syslog(LOG_ERR,"ERRROR display thread: display mutex unlock function failed: %s",strerror(rc));
        //     exit(SYSTEM_ERROR);
        // }

        clock_gettime(CLOCK_MONOTONIC, &end_frame);

        // call mutex lock -> lock frame time calculation for each frame
        rc = pthread_mutex_lock(&time_mutex);
        // rc = pthread_mutex_lock(frameParams->proc_frame_mutex);
        if(rc != 0){
            syslog(LOG_ERR,"ERRROR fps calculation: mutex lock function failed: %s",strerror(rc));
            // cleanup(false,0,0,local_w_file_fd);
            // raise(SIGINT);
            exit(SYSTEM_ERROR);
        }

        frame_time_calc(&start_60S,&start_1S,&end_frame,type_disp);
        // *frameParams->proc_frame_cnt = *frameParams->proc_frame_cnt + 1;

        rc = pthread_mutex_unlock(&time_mutex);
        // rc = pthread_mutex_unlock(frameParams->proc_frame_mutex);
        if(rc != 0){
            syslog(LOG_ERR,"ERRROR fps calculation: mutex unlock function failed: %s",strerror(rc));
            // cleanup(false,0,0,local_w_file_fd);
            // raise(SIGINT);
            exit(SYSTEM_ERROR);
        }

        current_frames++;

        // syslog(LOG_NOTICE, "Display: frame #%d completed", current_frames);
      }
    }
      
    if(!(current_frames % 500) && dispParams->save_to_video)
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
    // timerParams_t timerParams;
    // struct sigevent sev;
    // timer_t timerid;
    // int clock_id;
    // // struct timespec start_time;
    // struct itimerspec timer;

    // memset(&timerParams,0,sizeof(timerParams_t));
    // memset(&sev,0,sizeof(struct sigevent));
    // memset(&timer, 0, sizeof(struct itimerspec));

  //////////////////// declare opencv variables ////////////  
    VideoCapture input;
    // VideoWriter  output_vid;
    displayParams_t dispParams;
    pthread_t disp_thread_ID;
    //VideoCapture cap;
    //int frame_tot = 0; 
    int proc_frame_cnt = 0;
    int disp_frame_cnt = 0;

    pthread_mutex_t proc_frame_mutex;
    pthread_mutex_t disp_frame_mutex;

    bool OUT_WRITE = false;
    int in_fps = 0;
    int fourcc_code_out = 0;
    Size desired_size = SIZE_240P;
    //CascadeClassifier stop_cascade;
    Size in_size;
    //mode = false;
    uint16_t total_frames = 0;
    bool done = false;

  //////////////////// initialize timer variables ////////////  

    /* the following code was based off of the example at:
        https://github.com/cu-ecen-aeld/aesd-lectures/blob/master/lecture9/timer_thread.c
    */
    // clock_id = CLOCK_MONOTONIC;
    // sev.sigev_notify = SIGEV_THREAD;
    // sev.sigev_value.sival_ptr = &timerParams;
    // sev.sigev_notify_function = frame_time_calc_thread;

    // timer.it_interval.tv_sec = 1;
    // timer.it_interval.tv_nsec = 0;

    // if ( timer_create(clock_id,&sev,&timerid) != 0 ) {
    //   // printf("Error creating timer\n",errno,strerror(errno));
    //   cout << "ERROR: could not create timer: " << strerror(errno) << endl;
    //   exit(SYSTEM_ERROR);
    // } 


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
  

  //////////////////// create output file if necessary ////////////  
    // create output file if input given
    if(!out_file.empty()){
      fourcc_code_out = VideoWriter::fourcc('m','p','4','v');
      // output_vid.open(out_file, fourcc_code_out, in_fps, in_size,true);
      OUT_WRITE = true;
      desired_size = in_size;

    }

  //////////////////// get input propeties ////////////  
    // find properties of input
    in_fps = input.get(CAP_PROP_FPS);
    in_size = Size( (int)input.get(CAP_PROP_FRAME_WIDTH), (int)input.get(CAP_PROP_FRAME_HEIGHT)); //height and width of video 
    total_frames = input.get(CAP_PROP_FRAME_COUNT); // frame count

    cout << endl << "************** properties **************" << endl; 
    cout << "total frames: " << total_frames << endl;
    cout << "input fps: " << in_fps << endl;
    cout << "input size: " << in_size << endl;
    cout << "output size: "<< desired_size << endl << endl;



  
  //////////////////// create display windows ////////////  
    if(disp){
      namedWindow("original",WINDOW_NORMAL);
      namedWindow("grayscale",WINDOW_NORMAL);
    }
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
                  
    
    // DISPLAY TOGGLES
    // bool DISP_LINES = false;
    // bool DISP_STOP = false;
    
    //int TPR = 0;
    //int FPR = 0;

  
  ///////////////////////////////// initialize mutex /////////////////////////////////    
  
    // cpu_set_t cpuset;
    // int core_id;
    // int max_prio, rc;
    
    rc = pthread_mutex_init(&time_mutex,NULL);
    if(rc != 0){

        syslog(LOG_ERR, "ERROR: could not succesfully initialize mutex, error number: %d", rc);
        return SYSTEM_ERROR;

    }

    // rc = pthread_mutex_init(&timerParams.timer_mutex,NULL);
    // if(rc != 0){

    //     syslog(LOG_ERR, "ERROR: could not succesfully initialize timer mutex, error number: %d", rc);
    //     return SYSTEM_ERROR;

    // }

    // rc = pthread_mutex_init(&proc_frame_mutex,NULL);
    // if(rc != 0){

    //     syslog(LOG_ERR, "ERROR: could not succesfully initialize processing frame mutex, error number: %d", rc);
    //     return SYSTEM_ERROR;

    // }

    // rc = pthread_mutex_init(&disp_frame_mutex,NULL);
    // if(rc != 0){

    //     syslog(LOG_ERR, "ERROR: could not succesfully initialize display frame mutex, error number: %d", rc);
    //     return SYSTEM_ERROR;

    // }

  ///////////////////////////////// initialize and create other threads ///////////////////////////////// 
    // initialize linked list
    TAILQ_INIT(&frame_list);
    // struct timespec start_60S,start_1S, end_frame;
    // dispParams.total_frames = total_frames;
    dispParams.done = &done;
    // dispParams.output_vid = output_vid;
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

    // timerParams.disp_frame_cnt = &disp_frame_cnt;
    // timerParams.proc_frame_cnt = &proc_frame_cnt;

    // timerParams.disp_frame_mutex = &disp_frame_mutex;
    // timerParams.proc_frame_mutex = &proc_frame_mutex;                
  
  ///////////////////////////////// begin frame processing and timer ///////////////////////////////// 
    syslog(LOG_NOTICE, "general: ********************************* APPLICATION START ****************************");

    Mat frame, frame_gray;

    // start time for 1 minute comparison
    clock_gettime(CLOCK_MONOTONIC, &start_60S);
    
    // start time for frame
    clock_gettime(CLOCK_MONOTONIC, &start_1S);

    // clock_gettime(clock_id, &timer.it_value);
    
    // timer.it_value.tv_sec++;

    // if(timer_settime(timerid, TIMER_ABSTIME, &timer, NULL) != 0){
    //   cout << "ERROR: Could not set timer: " << strerror(errno);
    //   exit(SYSTEM_ERROR);
    // }

    while(!done){
    //for(int i=0; i<NUM_FEAT; i++)
    //{
    
    
        // read in frame
      if(!input.read(frame)){
        cout << "all frames have finished processing, displaying remaining frames" << endl;
        //while(1){
        //  if((winInput = waitKey(0)) == ESCAPE_KEY){ 
        //    break;
        //  }
        //}
        // waitKey(0);
        done = true;
        // while(!done){}

        break;
      } 
      
      if(disp)
        imshow("original", frame);
      
      // grayscale and blur
      cvtColor( frame, frame_gray, COLOR_BGR2GRAY );
      GaussianBlur(frame_gray, frame_gray, Size(9,9), 2, 2);

      if(disp)
        imshow("grayscale", frame_gray );
      
      
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
      
      // threadParams.stop_frame = frame_gray(stop_roi);
      
      frameParams->stop_roi = stop_roi;
      
      //threadParams[STOP_DETECT].in_size = in_size;
      
      frameParams->focal = focal;
      
      frameParams->disp = disp;
      frameParams->hog = hog;
      frameParams->complete = false;
      
      // cout << "thread params set" << endl;

      pthread_create(&frameParams->thread_ID,   // pointer to thread descriptor
                    NULL,     // use default attributes
                    frame_proc_thread, // thread function entry point
                    (void *)(frameParams) // parameters to pass in
                    );

      // cout << "created frame thread with ID: " << frameParams->thread_ID << endl;

      TAILQ_INSERT_TAIL(&frame_list, frameParams,next_frame);

      
    }
    
    pthread_join(disp_thread_ID, NULL);
    cout << "cleaning up...." << endl;

    // if(timer_delete(timerid) != 0) {
    //   cout << "ERROR: could not delete timer: " << strerror(errno) << endl;
    // }    
    // cout << "joined disp_thread" << endl;

    if(!TAILQ_EMPTY(&frame_list))
      cout << "finishing remaining frame processing threads..." << endl;

    while(!TAILQ_EMPTY(&frame_list)){
      frameParams_t *frameParams = TAILQ_FIRST(&frame_list);

      if(frameParams != NULL){
        // cout << "attempting to cancel thread ID: " << frameParams->thread_ID << endl;
        // rc = pthread_cancel(frameParams->thread_ID);
        // if(!rc){
        //   cout << "failed to cancel thread id" <<  frameParams->thread_ID << endl;
        // }
        pthread_join(frameParams->thread_ID, NULL);

        // cout << "thread ID: " << frameParams->thread_ID << " cancelled" << endl;

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
