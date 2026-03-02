// Compiled separately with _GLIBCXX_USE_CXX11_ABI=1 so OpenCV links correctly.
// Exposes only C-linkage functions — no std::string crosses the boundary.

#include <opencv2/opencv.hpp>
#include "cv_wrapper.h"

extern "C" {

CvCapHandle cv_cap_open(const char* path) {
    auto* cap = new cv::VideoCapture();
    cap->open(std::string(path));
    if (!cap->isOpened()) {
        delete cap;
        return nullptr;
    }
    return static_cast<CvCapHandle>(cap);
}

int cv_cap_read(CvCapHandle cap, CvMatHandle mat) {
    return static_cast<cv::VideoCapture*>(cap)->read(*static_cast<cv::Mat*>(mat)) ? 1 : 0;
}

void cv_cap_release(CvCapHandle cap) {
    auto* c = static_cast<cv::VideoCapture*>(cap);
    c->release();
    delete c;
}

CvMatHandle cv_mat_create() {
    return static_cast<CvMatHandle>(new cv::Mat());
}

void cv_mat_destroy(CvMatHandle mat) {
    delete static_cast<cv::Mat*>(mat);
}

int cv_mat_rows(CvMatHandle mat) {
    return static_cast<cv::Mat*>(mat)->rows;
}

int cv_mat_cols(CvMatHandle mat) {
    return static_cast<cv::Mat*>(mat)->cols;
}

unsigned char* cv_mat_data(CvMatHandle mat) {
    return static_cast<cv::Mat*>(mat)->data;
}

int cv_mat_type(CvMatHandle mat) {
    return static_cast<cv::Mat*>(mat)->type();
}

void cv_cvt_color(CvMatHandle src, CvMatHandle dst, int code) {
    cv::cvtColor(*static_cast<cv::Mat*>(src), *static_cast<cv::Mat*>(dst), code);
}

void cv_resize(CvMatHandle src, CvMatHandle dst, int width, int height, int interpolation) {
    cv::resize(*static_cast<cv::Mat*>(src), *static_cast<cv::Mat*>(dst),
               cv::Size(width, height), 0, 0, interpolation);
}

CvMatHandle cv_mat_roi(CvMatHandle mat, int x, int y, int w, int h) {
    cv::Mat* m = static_cast<cv::Mat*>(mat);
    cv::Mat* roi = new cv::Mat((*m)(cv::Rect(x, y, w, h)).clone());
    return static_cast<CvMatHandle>(roi);
}

int cv_COLOR_BGR2RGB() { return cv::COLOR_BGR2RGB; }
int cv_INTER_CUBIC()   { return cv::INTER_CUBIC; }

} // extern "C"
