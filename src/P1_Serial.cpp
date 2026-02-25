//C++ version of the serial video

#include <iostream>
#include <vector>
#include <chrono>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <torch/torch.h>
#include <torch/script.h>

using namespace std;
using namespace cv;
using namespace torch;
using namespace torch::jit;

//configuration
const string MODEL_PATH = "models/RemoteCLIP-ViT-L-14_scripted.pt";
const string VIDEO_PATH = "data/short_video.mp4";
const string OUTPUT_PATH = "data/short_video_embeddings.pt";
const int FRAME_INTERVAL = 30; // Extract 1 frame per second (assuming the video is 30 fps)
const int BATCH_SIZE = 16;
const int CROP_SIZE = 224; // ViT-L-14 input size

