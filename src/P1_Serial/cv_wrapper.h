#pragma once
#ifdef __cplusplus
extern "C" {
#endif

// Opaque handle types
typedef void* CvCapHandle;
typedef void* CvMatHandle;

// VideoCapture
CvCapHandle cv_cap_open(const char* path);
int         cv_cap_read(CvCapHandle cap, CvMatHandle mat);
void        cv_cap_release(CvCapHandle cap);

// Mat
CvMatHandle cv_mat_create();
void        cv_mat_destroy(CvMatHandle mat);
int         cv_mat_rows(CvMatHandle mat);
int         cv_mat_cols(CvMatHandle mat);
unsigned char* cv_mat_data(CvMatHandle mat);
int         cv_mat_type(CvMatHandle mat);

// Image processing
void cv_cvt_color(CvMatHandle src, CvMatHandle dst, int code);
void cv_resize(CvMatHandle src, CvMatHandle dst, int width, int height, int interpolation);
CvMatHandle cv_mat_roi(CvMatHandle mat, int x, int y, int w, int h);

// Constants
int cv_COLOR_BGR2RGB();
int cv_INTER_CUBIC();

#ifdef __cplusplus
}
#endif
