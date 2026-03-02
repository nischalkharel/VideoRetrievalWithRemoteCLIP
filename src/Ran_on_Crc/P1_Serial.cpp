//C++ version of the serial video
// Modified to use cv_wrapper (C-linkage) to avoid GCC ABI mismatch
// between LibTorch (ABI=0) and OpenCV (ABI=1)

#include <iostream>
#include <vector>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <torch/torch.h>
#include <torch/script.h>
#include "cv_wrapper.h"

using namespace std;
using namespace torch;
using namespace torch::jit;

//configuration
const char* MODEL_PATH = "models/RemoteCLIP-ViT-L-14_scripted.pt";
const char* VIDEO_PATH = "data/short_video.mp4";
const char* OUTPUT_PATH = "data/short_video_embeddings.pt";
const int FRAME_INTERVAL = 30; // Extract 1 frame per second (assuming the video is 30 fps)
const int BATCH_SIZE = 16;
const int CROP_SIZE = 224; // ViT-L-14 input size

//CLIP normalization constants - verified from open_clip output
const float MEAN[3] = {0.48145466f, 0.4578275f, 0.40821073f};
const float STD[3] = {0.26862954f, 0.26130258f, 0.27577711f};

//Preprocessing (all on CPU)
// Takes raw frame data instead of cv::Mat
Tensor preprocess_frame(unsigned char* data, int rows, int cols) {
    // All image processing done in torch to avoid ABI boundary issues.
    // Only video I/O uses the C wrapper.

    // Step 1: Load BGR data into tensor [H, W, 3]
    auto options_u8 = torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU);
    torch::Tensor bgr_tensor = torch::from_blob(
        data, {rows, cols, 3}, options_u8
    ).clone();  // clone to own the memory

    // Step 2: BGR -> RGB (swap channel 0 and 2)
    torch::Tensor rgb_tensor = bgr_tensor.index({
        torch::indexing::Slice(),
        torch::indexing::Slice(),
        torch::tensor({2, 1, 0})
    });

    // Step 3: Resize shortest edge to 224, maintain aspect ratio
    // We'll use torch interpolation instead of OpenCV resize
    int h = rows;
    int w = cols;
    int new_h, new_w;
    if (h < w) {
        new_h = CROP_SIZE;
        new_w = static_cast<int>(w * (CROP_SIZE / static_cast<float>(h)));
    } else {
        new_w = CROP_SIZE;
        new_h = static_cast<int>(h * (CROP_SIZE / static_cast<float>(w)));
    }

    // Convert to float [1, 3, H, W] for interpolation
    torch::Tensor for_resize = rgb_tensor.permute({2, 0, 1}).unsqueeze(0).to(torch::kFloat32);
    torch::Tensor resized = torch::nn::functional::interpolate(
        for_resize,
        torch::nn::functional::InterpolateFuncOptions()
            .size(std::vector<int64_t>{new_h, new_w})
            .mode(torch::kBicubic)
            .align_corners(false)
    );  // [1, 3, new_h, new_w]

    // Step 4: Center crop to 224x224
    int crop_y = (new_h - CROP_SIZE) / 2;
    int crop_x = (new_w - CROP_SIZE) / 2;
    torch::Tensor cropped = resized.index({
        torch::indexing::Slice(),
        torch::indexing::Slice(),
        torch::indexing::Slice(crop_y, crop_y + CROP_SIZE),
        torch::indexing::Slice(crop_x, crop_x + CROP_SIZE)
    });  // [1, 3, 224, 224]

    // Step 5: Scale to [0,1]
    torch::Tensor tensor = cropped.div_(255.0f);

    // Step 6: Normalize with CLIP mean and std
    auto options_f32 = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
    torch::Tensor mean_t = torch::from_blob(
        (void*)MEAN, {3}, options_f32
    ).clone().view({1, 3, 1, 1});

    torch::Tensor std_t = torch::from_blob(
        (void*)STD, {3}, options_f32
    ).clone().view({1, 3, 1, 1});

    tensor = (tensor - mean_t) / std_t;

    // Already has batch dim [1, 3, 224, 224]
    cout << "Preprocessed tensor device: " << tensor.device() << endl;

    return tensor;
}

// Save embeddings to disk
void save_embeddings(
    const char* path,
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

    // Open video file using C wrapper
    CvCapHandle cap = cv_cap_open(VIDEO_PATH);
    if (cap == nullptr) {
        cerr << "ERROR: Cannot open video file: " << VIDEO_PATH << endl;
        return -1;
    }

    // storage
    vector<vector<float>> frame_embeddings;
    vector<int> frame_indices;
    vector<Tensor> frame_batch;

    int frame_count = 0;
    auto start_time = chrono::high_resolution_clock::now();

    // Timing variables
    double total_preprocess_time = 0.0;
    double total_inference_time = 0.0;

    // batch inference lambda
    auto run_batch = [&]() {
        auto inference_start = chrono::high_resolution_clock::now();
        if (frame_batch.empty()) return;

        Tensor batch_tensor = cat(frame_batch, 0);
        batch_tensor = batch_tensor.to(device);

        NoGradGuard no_grad;

        vector<IValue> inputs = {batch_tensor};
        Tensor image_features = model.forward(inputs).toTensor();

        Tensor norm = image_features.norm(2, -1, true);
        image_features = image_features / norm;

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

    // main loop: read video frames via wrapper
    CvMatHandle frame_mat = cv_mat_create();
    while (cv_cap_read(cap, frame_mat)) {

        if (frame_count % FRAME_INTERVAL == 0) {
            auto preprocess_start = chrono::high_resolution_clock::now();

            int rows = cv_mat_rows(frame_mat);
            int cols = cv_mat_cols(frame_mat);
            unsigned char* data = cv_mat_data(frame_mat);

            Tensor preprocessed = preprocess_frame(data, rows, cols);

            auto preprocess_end = chrono::high_resolution_clock::now();
            total_preprocess_time += chrono::duration<double>(preprocess_end - preprocess_start).count();

            frame_batch.push_back(preprocessed);
            frame_indices.push_back(frame_count);

            if (static_cast<int>(frame_batch.size()) == BATCH_SIZE)
                run_batch();
        }

        frame_count++;
    }

    // Process remaining frames
    if (!frame_batch.empty()) {
        run_batch();
    }

    cv_mat_destroy(frame_mat);
    cv_cap_release(cap);

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
