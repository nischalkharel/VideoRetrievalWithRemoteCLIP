//C++ version of the serial video

#include <iostream>
#include <vector>
#include <chrono>
#include <fstream>
#include <iomanip>
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

//CLIP normalization constants - verified from open_clip output
const float MEAN[3] = {0.48145466f, 0.4578275f, 0.40821073f};
const float STD[3] = {0.26862954f, 0.26130258f, 0.27577711f};

//Preprocessing (all on CPU)
Tensor preprocess_frame(const Mat& bgr_frame){
    // step 1: convert BGR to RGB: opencv reads in BGR format, but CLIP expects RGB
    Mat rgb_frame;
    cvtColor(bgr_frame, rgb_frame, COLOR_BGR2RGB);


    // step 2: Resize shortest edge to 224 while maintaining aspect ratio
    int h = rgb_frame.rows;
    int w = rgb_frame.cols;
    int new_h, new_w;
    if (h < w) {
        new_h = CROP_SIZE;
        new_w = static_cast<int>(w * (CROP_SIZE / static_cast<float>(h)));
    } else {
        new_w = CROP_SIZE;
        new_h = static_cast<int>(h * (CROP_SIZE / static_cast<float>(w)));
    }

    Mat resized;
    resize(rgb_frame, resized, Size(new_w, new_h), 0, 0, INTER_CUBIC);

    // step 3: Center crop to 224x224
    int crop_x = (new_w - CROP_SIZE) / 2;
    int crop_y = (new_h - CROP_SIZE) / 2;
    Mat cropped = resized(Rect(crop_x, crop_y, CROP_SIZE, CROP_SIZE));

    // step 4: HWC uint8 -> CHW float32, scale to [0,1] (all on CPU)
    // .clone() is critical — from_blob() does NOT own the memory.
    // Without it the tensor data will corrupt when Mat goes out of scope.
    auto options_u8 = torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU);
    torch::Tensor tensor = torch::from_blob(
    cropped.data,
    {CROP_SIZE, CROP_SIZE, 3},
    options_u8
    ).clone()
    .permute({2, 0, 1})
    .to(torch::kFloat32)
    .div_(255.0f);

    // Step 5: Normalize with CLIP mean and std (all on CPU)
    // Using tensor() directly to avoid from_blob const-cast issues
    auto options_f32 = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
    torch::Tensor mean_t = torch::from_blob(
    (void*)MEAN, {3}, options_f32
).clone().view({3, 1, 1});

torch::Tensor std_t = torch::from_blob(
    (void*)STD, {3}, options_f32
).clone().view({3, 1, 1});

    tensor = (tensor - mean_t) / std_t;

    // Add batch dimension: [C,H,W] -> [1,C,H,W]
    // NOTE: Do NOT move to device yet. This happens when batch is ready.
    cout << "Preprocessed tensor device: " << tensor.device() << endl;  // Should print "cpu" cause we want to do all preprocessing on CPU
    return tensor.unsqueeze(0);
}

// Save embeddings to disk
// Writes a simple binary file: [num_embeddings, embed_dim, indices..., data...]
// Read back in Python with:
//   num_emb = np.frombuffer(f.read(4), dtype=np.int32)[0]
//   emb_dim = np.frombuffer(f.read(4), dtype=np.int32)[0]
//   indices = np.frombuffer(f.read(4 * num_emb), dtype=np.int32)
//   data    = np.frombuffer(f.read(), dtype=np.float32).reshape(num_emb, emb_dim)

void save_embeddings(
    const string& path,
    const vector<vector<float>>& embeddings,
    const vector<int>& frame_indices
)
{
    ofstream file(path, ios::binary);
    if (!file.is_open()) {
        cerr << "ERROR: Cannot open output file: " << path << endl;
        return;
    }

    int num_embeddings = static_cast<int>(embeddings.size());
    int embed_dim      = num_embeddings > 0 ? static_cast<int>(embeddings[0].size()) : 0;

    // Write header
    file.write(reinterpret_cast<const char*>(&num_embeddings), sizeof(int));
    file.write(reinterpret_cast<const char*>(&embed_dim),      sizeof(int));

    // Write frame indices
    for (int idx : frame_indices)
        file.write(reinterpret_cast<const char*>(&idx), sizeof(int));

    // Write embedding rows
    for (const auto& emb : embeddings)
        file.write(reinterpret_cast<const char*>(emb.data()), emb.size() * sizeof(float));

    file.close();
    cout << "Saved " << num_embeddings << " embeddings (dim=" << embed_dim
         << ") to " << path << endl;
}

// main
int main() {
    // Device setup
    torch::Device device(torch::kCPU);
if (torch::cuda::is_available()) {
    device = torch::Device(torch::kCUDA);
}
    std::cout << "Using device: " << (device.is_cuda() ? "cuda" : "cpu") << "\n\n";

    // Load TorchScript model
    cout << "Loading model from: " << MODEL_PATH << endl;
    script::Module model;
    try {
        model = load(MODEL_PATH);
    } catch (const c10::Error& e) {
        cerr << "ERROR: Failed to load model: " << e.what() << endl;
        return -1;
    }
    model.to(device);
    model.eval();
    cout << "Model loaded successfully and is ready for inference.\n\n";

    // Open video file
    VideoCapture cap(VIDEO_PATH);
    if (!cap.isOpened()) {
        cerr << "ERROR: Cannot open video file: " << VIDEO_PATH << endl;
        return -1;
    }

    // storage
    vector<vector<float>> frame_embeddings; // List of normalized embeddings
    vector<int> frame_indices; // To retrieve/display the frames later
    vector<Tensor> frame_batch; // For batching frames before inference

    int frame_count = 0;
    auto start_time = chrono::high_resolution_clock::now();
    
    // Timing variables
    double total_preprocess_time = 0.0;
    double total_inference_time = 0.0;

    // batch inference lambda --THIS IS A FUNCTION BTW
    auto run_batch = [&]() {
        auto inference_start = chrono::high_resolution_clock::now();
        if (frame_batch.empty()) return;

        // Stack [1,C,H,W] tensors into [B,C,H,W] on CPU
        Tensor batch_tensor = cat(frame_batch, 0); 
        
        // Move entire batch to GPU at once (more efficient than moving each frame individually)
        batch_tensor = batch_tensor.to(device);

        NoGradGuard no_grad; // mirrors: with torch.no_grad()

        // Forward pass through the visual encoder (GPU inference)
        vector<IValue> inputs = {batch_tensor};
        Tensor image_features = model.forward(inputs).toTensor();

        // L2 normalize: features /= features.norm(dim=-1, keepdim=True)
        Tensor norm = image_features.norm(2, -1, true);
        image_features = image_features / norm;

        // Copy each row to a plain float vector for storage
        image_features = image_features.cpu().contiguous();
        int batch_n   = image_features.size(0);
        int embed_dim = image_features.size(1);

        for (int i = 0; i < batch_n; i++) {
            float* ptr = image_features[i].data_ptr<float>();
            frame_embeddings.emplace_back(ptr, ptr + embed_dim);
        }

        frame_batch.clear();
        
        auto inference_end = chrono::high_resolution_clock::now();
        total_inference_time += chrono::duration<double>(inference_end - inference_start).count();
    };

    // main loop: read video frames, preprocess, batch, and run inference
    Mat frame;
    while (cap.read(frame)) {

        if (frame_count % FRAME_INTERVAL == 0) {
            auto preprocess_start = chrono::high_resolution_clock::now();
            
            Tensor preprocessed = preprocess_frame(frame);  // CPU preprocessing
            
            auto preprocess_end = chrono::high_resolution_clock::now();
            total_preprocess_time += chrono::duration<double>(preprocess_end - preprocess_start).count();
            
            frame_batch.push_back(preprocessed);
            frame_indices.push_back(frame_count);

            // Run inference when batch is full
            if (static_cast<int>(frame_batch.size()) == BATCH_SIZE)
                run_batch();
        }

        frame_count++;
    }
    
    // Process remaining frames
    if (!frame_batch.empty()) {
        run_batch();
    }
    cap.release();
    
    auto end_time = chrono::high_resolution_clock::now();
    double total_time = chrono::duration<double>(end_time - start_time).count();
    int num_frames = static_cast<int>(frame_embeddings.size());
    
    // Print timing results
    cout << "\n=== Serial Version Timing Results ===" << endl;
    cout << "Total time: " << fixed << setprecision(2) << total_time << " seconds" << endl;
    cout << "  Preprocessing time: " << total_preprocess_time << " seconds (" 
         << (total_preprocess_time / total_time * 100) << "%)" << endl;
    cout << "  Inference time: " << total_inference_time << " seconds (" 
         << (total_inference_time / total_time * 100) << "%)" << endl;
    cout << "Total frames processed: " << frame_indices.size() << endl;
    cout << "Frames per second (FPS): " << (num_frames / total_time) << endl;

    // Save embeddings
    save_embeddings(OUTPUT_PATH, frame_embeddings, frame_indices);

    return 0;
}