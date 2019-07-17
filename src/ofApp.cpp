#include "ofApp.h"
//#include <omp.h>

using namespace cv;
using namespace ofxCv;

//---------------------------------------------------------
cv::Point2d phaseCorrelate1D(InputArray _src1, InputArray _src2, InputArray _window, int k=65535, int flag_pixel = false);

static void divSpectrums1D( Mat &srcA, Mat &srcB, Mat &dst, int flags, bool conjB);

static void fftShift1D(Mat &out);

static void magSpectrums1D( Mat &src, Mat &dst);

//--------------------------------------------------------------
void ofApp::setup(){

    // Parameters

    temp1.load("/Users/mun/Desktop/rr.jpg");
    temp2.load("/Users/mun/Desktop/ll.jpg");
    temp1.setImageType(OF_IMAGE_GRAYSCALE);
    temp2.setImageType(OF_IMAGE_GRAYSCALE);
    Mat tempI= toCv(temp1);
    Mat tempJ= toCv(temp2);

    double mult_scale = 2;      // １階層ごとの画像サイズの縮尺
    int thresh = tempI.cols/3;  // 階層の深さ決定の目安
    int ww = getOptimalDFTSize(32); // 窓の幅 半値幅が有効幅
    int wh = 17; // 窓の高さ
    
    // Decision of the Image Hierarchical Level
    int lmax=0;
    for(;;){
        int value = (int)(pow(mult_scale,-(double)(lmax++))*tempI.cols);
        if(thresh>value) break;
    }
    
    vector<int> width, height;
    width.resize(lmax);
    height.resize(lmax);
    
    for(int i=0; i<lmax; i++){
        width[i] = (int)(pow(mult_scale,-(i))*tempI.cols);
        height[i] = tempI.rows;
    }
    
    Mat hann = Mat(wh,ww,CV_64F);
    for(int j=0; j<wh; j++){
        for(int i=0; i<ww; i++){
            hann.at<double>(j,i) = 0.5 - 0.5*cos(2.0 * CV_PI * static_cast<double>(i) / ww); // ww-1
        }
    }
    
    // Step1: Image Pyramid
    vector<Mat> I, J, LI, LJ;
    LI.resize(lmax);
    LJ.resize(lmax);
    I.resize(lmax);
    J.resize(lmax);
    
    for(int l=0; l<lmax; l++){
        resize(tempI, I[l], cv::Size(width[l], height[l]), 0, 0, 1);
        resize(tempJ, J[l], cv::Size(width[l], height[l]), 0, 0, 1);
        copyMakeBorder(I[l], LI[l], wh/2, wh/2, ww/2, ww/2, BORDER_CONSTANT, Scalar::all(0));
        copyMakeBorder(J[l], LJ[l], wh/2, wh/2, ww/2, ww/2, BORDER_CONSTANT, Scalar::all(0));
    }
    
    Mat disparity = Mat::zeros( cv::Size(tempI.cols, tempI.rows), CV_64F);
    
    for(int v= 0; v<(int)I[0].rows; v++){
        
        for(int u=0; u<(int)I[0].cols; u++){
            int error = 0;
            
            vector<Point2d> P, Q, Qd;
            P.resize(lmax+1);
            Q.resize(lmax+1);
            Qd.resize(lmax+1);
            
            // Step2:
            P[lmax] = Point2d(pow(mult_scale,-lmax)*u,v);
            Q[lmax] = Point2d(pow(mult_scale,-lmax)*u,v);
            
            for(int l=lmax-1; l>=0; l--){
                // Step3:
                P[l] = Point2d(pow(mult_scale,-l)*u,v);
                Qd[l] = Point2d(mult_scale*Q[l+1].x,Q[l+1].y);
                
                // Step4:
                cv::Rect i_roi((int)P[l].x,(int)P[l].y,ww,wh);
                Mat Iroi = LI[l](i_roi);
                
                if((Qd[l].x<0)||(LJ[l].cols<Qd[l].x+ww)){
                    Q[l] = P[l];
                    continue;
                }
                cv::Rect j_roi = cv::Rect((int)Qd[l].x,(int)Qd[l].y,ww,wh);
                Mat Jroi = LJ[l](j_roi);
                
                Mat I64f, J64f;
                Iroi.convertTo(I64f, CV_64F);
                Jroi.convertTo(J64f, CV_64F);
                
                Point2d shift = phaseCorrelate1D(I64f, J64f, hann, ww/2, true);
                
                if( P[l].x < (shift.x+Qd[l].x)){
                    Q[l] = Qd[l]+Point2d(shift.x, 0);
                }
                else {
                    Q[l] = Qd[l];
                }
            } // Step5
            { // Step6
                cv::Rect i_roi((int)P[0].x,(int)P[0].y,ww,wh);
                Mat Iroi = LI[0](i_roi);
                
                if((Q[0].x<0)||(LJ[0].cols<Q[0].x+ww)){
                    error = 1;
                    goto end;
                }
                cv::Rect j_roi = cv::Rect((int)Q[0].x,(int)Q[0].y,ww,wh);
                Mat Jroi = LJ[0](j_roi);
                
                Mat I64f, J64f;
                Iroi.convertTo(I64f, CV_64F);
                Jroi.convertTo(J64f, CV_64F);
                
                Point2d shift = phaseCorrelate1D(I64f, J64f, hann, ww/2, false);
                
                if( P[0].x < (shift.x+Q[0].x)){
                    Q[0] = Q[0]+Point2d(shift.x, 0);
                }
                
#if _DEBUG
                Mat matchedImg = Mat::zeros(Size(I[0].cols*2,0), CV_8UC1);
                
                vconcat(I[0],J[0],matchedImg);
                cvtColor( matchedImg,  matchedImg, COLOR_GRAY2RGB);
                line(matchedImg,Point(P[0].x,P[0].y),Point(Q[0].x,matchedImg.rows/2+Q[0].y),Scalar(0,0,255));
                //imshow("matched", matchedImg);
                toOf(matchedImg, matchImage);
                
#endif
            }
        end:
            double d = Q[0].x-P[0].x;
            double view_band = 0.6;  // エラー閾値、表示調整用
            if(view_band*ww<d||error) d=0;
            disparity.at<double>(v, u) = d;
        }
    }
    
    double min, max;
    cv::minMaxIdx(disparity, &min, &max);
    Mat adjMap;
    double scale = 255/(max-min);
    disparity.convertTo(adjMap, CV_8UC1, scale, -min*scale);
    
    cv::Mat falseColorMap;
    applyColorMap(adjMap, falseColorMap, cv::COLORMAP_JET);
    
    //imshow("disparity", falseColorMap);
    toOf(falseColorMap, dispa);
    //imwrite("disparity.bmp", falseColorMap);
    
}


//--------------------------------------------------------------
void ofApp::update(){

}

//--------------------------------------------------------------
void ofApp::draw(){

    ofSetColor(255);
    temp1.draw(0, 0, 512, 512);
    temp2.draw(512, 0, 512, 512);
    dispa.draw(1024, 0, 512, 512);
}

//--------------------------------------------------------------
void ofApp::keyPressed(int key){

}

//--------------------------------------------------------------
void ofApp::keyReleased(int key){

}

//--------------------------------------------------------------
void ofApp::mouseMoved(int x, int y ){

}

//--------------------------------------------------------------
void ofApp::mouseDragged(int x, int y, int button){

}

//--------------------------------------------------------------
void ofApp::mousePressed(int x, int y, int button){

}

//--------------------------------------------------------------
void ofApp::mouseReleased(int x, int y, int button){

}

//--------------------------------------------------------------
void ofApp::mouseEntered(int x, int y){

}

//--------------------------------------------------------------
void ofApp::mouseExited(int x, int y){

}

//--------------------------------------------------------------
void ofApp::windowResized(int w, int h){

}

//--------------------------------------------------------------
void ofApp::gotMessage(ofMessage msg){

}

//--------------------------------------------------------------
void ofApp::dragEvent(ofDragInfo dragInfo){ 

}

// 1D POC -----------------------------------------------------------------
cv::Point2d phaseCorrelate1D(InputArray _src1, InputArray _src2, InputArray _window, int k, int flag_pixel)
{
    Mat src1 = _src1.getMat();
    Mat src2 = _src2.getMat();
    Mat window = _window.getMat();
    
    CV_Assert( src1.type() == src2.type());
    CV_Assert( src1.type() == CV_32FC1 || src1.type() == CV_64FC1 );
    CV_Assert( src1.size == src2.size);
    
    if(!window.empty())
    {
        CV_Assert( src1.type() == window.type());
        CV_Assert( src1.size == window.size);
    }
    
    int M = src1.rows;
    int N = getOptimalDFTSize(src1.cols);
    
    Mat padded1, padded2, paddedWin;
    
    if(M != src1.rows || N != src1.cols)
    {
        copyMakeBorder(src1, padded1, 0, 0, 0, N - src1.cols, BORDER_CONSTANT, Scalar::all(0));
        copyMakeBorder(src2, padded2, 0, 0, 0, N - src2.cols, BORDER_CONSTANT, Scalar::all(0));
        
        if(!window.empty())
        {
            copyMakeBorder(window, paddedWin, 0, 0, 0, N - window.cols, BORDER_CONSTANT, Scalar::all(0));
        }
    }
    else
    {
        padded1 = src1;
        padded2 = src2;
        paddedWin = window;
    }
    
    // perform window multiplication if available
    if(!paddedWin.empty())
    {
        // apply window to both images before proceeding...
        multiply(paddedWin, padded1, padded1);
        multiply(paddedWin, padded2, padded2);
    }
    
    vector<Mat> C;
    C.resize(M);
    
//#pragma omp parallel for
    for(int i=0; i<M; i++){
        Mat FFT1, FFT2, P, Pm;
        
        cv::Rect roi(0,i,N,1);
        Mat padded1_roi = padded1(roi);
        Mat padded2_roi = padded2(roi);
        
        // execute phase correlation equation
        // Reference: http://en.wikipedia.org/wiki/Phase_correlation
        dft(padded1_roi, FFT1, DFT_REAL_OUTPUT);
        dft(padded2_roi, FFT2, DFT_REAL_OUTPUT);
        
        mulSpectrums(FFT1, FFT2, P, 0, true);
        
        magSpectrums1D(P, Pm);
        divSpectrums1D(P, Pm, C[i], 0, false); // FF* / |FF*| (phase correlation equation completed here...)
        
        if(0<k&&k<C[i].cols){
            cv::Rect lpf_roi(k,0,C[i].cols-k,1);
            Mat zero = C[i](lpf_roi);
            zero = Mat::zeros(zero.size(), zero.type());
        }
    }
    
    Mat Rall;
    if(M<11){
//#pragma omp parallel for
        for(int i=0; i<M; i++){
            Rall = Rall + C[i]  / M;
        }
    }
    else{
        vector<double> hann;
        hann.resize(M);
        double sum_hann = 0;
        for(int i=0; i<M; i++){
            hann[i] = 0.5 - 0.5*cos(2.0 * CV_PI * static_cast<double>(i) / M); // ww-1
            sum_hann += hann[i];
        }
//#pragma omp parallel for
        for(int i=0; i<M; i++){
            Rall = Rall + C[i] * hann[i] / sum_hann;
        }
    }
    
    idft(Rall, Rall); // gives us the nice peak shift location...
    
    fftShift1D(Rall); // shift the energy to the center of the frame.
    
    // locate the highest peak
    cv::Point peakLoc;
    minMaxLoc(Rall, NULL, NULL, NULL, &peakLoc);
    
    Point2d t(0,0);
    if(!flag_pixel&&(peakLoc.x>=1)&&(peakLoc.x<Rall.cols-1)){
        t.x = (Rall.at<double>(0, peakLoc.x-1)-Rall.at<double>(0, peakLoc.x+1))/(2.0*Rall.at<double>(0,peakLoc.x-1)-4.0*Rall.at<double>(0,peakLoc.x)+2.0*Rall.at<double>(0,peakLoc.x+1));
    }
    
    // adjust shift relative to image center...
    Point2d center((double)padded1.cols / 2.0, (double)padded1.rows / 2.0);
    
    C.clear();
    
    t.x = t.x + peakLoc.x;
    return (center - t);
}


// 1D div spectrum for dividing ---------------------------------------------
static void divSpectrums1D( Mat &srcA, Mat &srcB, Mat &dst, int flags, bool conjB)
{
    int depth = srcA.depth(), cn = srcA.channels(), type = srcA.type();
    int rows = srcA.rows, cols = srcA.cols;
    int j, k;
    
    CV_Assert( type == srcB.type() && srcA.size() == srcB.size() );
    CV_Assert( type == CV_32FC1 || type == CV_32FC2 || type == CV_64FC1 || type == CV_64FC2 );
    
    dst.create( srcA.rows, srcA.cols, type );
    
    bool is_1d = (flags & DFT_ROWS) || (rows == 1 || (cols == 1 &&
                                                      srcA.isContinuous() && srcB.isContinuous() && dst.isContinuous()));
    
    if( is_1d && !(flags & DFT_ROWS) )
        cols = cols + rows - 1, rows = 1;
    
    int ncols = cols*cn;
    int j0 = cn == 1;
    int j1 = ncols - (cols % 2 == 0 && cn == 1);
    
    const double* dataA = (const double*)srcA.data;
    const double* dataB = (const double*)srcB.data;
    double* dataC = (double*)dst.data;
    double eps = DBL_EPSILON; // prevent div0 problems
    
    size_t stepA = srcA.step/sizeof(dataA[0]);
    size_t stepB = srcB.step/sizeof(dataB[0]);
    size_t stepC = dst.step/sizeof(dataC[0]);
    
    for( ; rows--; dataA += stepA, dataB += stepB, dataC += stepC )
    {
        if( is_1d && cn == 1 )
        {
            dataC[0] = dataA[0] / (dataB[0] + eps);
            if( cols % 2 == 0 )
                dataC[j1] = dataA[j1] / (dataB[j1] + eps);
        }
        
        if( !conjB )
            for( j = j0; j < j1; j += 2 )
            {
                double denom = dataB[j]*dataB[j] + dataB[j+1]*dataB[j+1] + eps;
                double re = dataA[j]*dataB[j] + dataA[j+1]*dataB[j+1];
                double im = dataA[j+1]*dataB[j] - dataA[j]*dataB[j+1];
                dataC[j] = re / denom;
                dataC[j+1] = im / denom;
            }
        else
            for( j = j0; j < j1; j += 2 )
            {
                double denom = dataB[j]*dataB[j] + dataB[j+1]*dataB[j+1] + eps;
                double re = dataA[j]*dataB[j] - dataA[j+1]*dataB[j+1];
                double im = dataA[j+1]*dataB[j] + dataA[j]*dataB[j+1];
                dataC[j] = re / denom;
                dataC[j+1] = im / denom;
            }
    }
}

// fftshift 1D -----------------------------------------------------------
static void fftShift1D(Mat &out)
{
    if(out.rows == 1 && out.cols == 1)
    {
        // trivially shifted.
        return;
    }
    
    std::vector<Mat> planes;
    split(out, planes);
    
    int xMid = out.cols >> 1;
    int yMid = out.rows >> 1;
    
    bool is_1d = xMid == 0 || yMid == 0;
    
    xMid = xMid + yMid;
    
    for(size_t i = 0; i < planes.size(); i++)
    {
        Mat tmp;
        Mat half0(planes[i], cv::Rect(0, 0, xMid, 1));
        Mat half1(planes[i], cv::Rect(xMid, 0, xMid, 1));
        
        half0.copyTo(tmp);
        half1.copyTo(half0);
        tmp.copyTo(half1);
    }
    
    merge(planes, out);
}

// 1D spectrum --------------------------------------------------------
static void magSpectrums1D( Mat &src, Mat &dst)
{
    int depth = src.depth(), cn = src.channels(), type = src.type();
    int rows = src.rows, cols = src.cols;
    int j, k;
    
    CV_Assert( type == CV_32FC1 || type == CV_32FC2 || type == CV_64FC1 || type == CV_64FC2 );
    
    dst.create( src.rows, src.cols,  CV_64FC1 );
    
    dst.setTo(0);//Mat elements are not equal to zero by default!
    
    bool is_1d = (rows == 1 || (cols == 1 && src.isContinuous() && dst.isContinuous()));
    
    if( is_1d )
        cols = cols + rows - 1, rows = 1;
    
    int ncols = cols*cn;
    int j0 = cn == 1;
    int j1 = ncols - (cols % 2 == 0 && cn == 1);
    
    const double* dataSrc = (const double*)src.data;
    double* dataDst = (double*)dst.data;
    
    size_t stepSrc = src.step/sizeof(dataSrc[0]);
    size_t stepDst = dst.step/sizeof(dataDst[0]);
    
    for( ; rows--; dataSrc += stepSrc, dataDst += stepDst )
    {
        if( is_1d && cn == 1 )
        {
            dataDst[0] = dataSrc[0]*dataSrc[0];
            if( cols % 2 == 0 )
                dataDst[j1] = dataSrc[j1]*dataSrc[j1];
        }
        
        for( j = j0; j < j1; j += 2 )
        {
            dataDst[j] = std::sqrt(dataSrc[j]*dataSrc[j] + dataSrc[j+1]*dataSrc[j+1]);
        }
    }
}
