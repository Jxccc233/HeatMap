#ifndef PLAYBACKHEATMAPGENERATOR_H  
#define PLAYBACKHEATMAPGENERATOR_H  

#include <vector>  
#include <opencv2/opencv.hpp>  
#include <QtGui/QPixmap>
#include <QtGui/QImage>

// �궨�峣��  
#define HEATMAP_SIZE        100     // �ƶ˸�������ֵ����Ϊ100*100  
#define GREEN_THRESHOLD     0.35f  
#define TRANSITION_WIDTH    0.15f   
#define BLUR_SIGMA          2.0  
#define BLUR_SIGMA_MASK     1.0  
#define ALPHA_MIN           51  
#define ALPHA_RANGE         204  
#define WHITE_TO_GREEN_START    0.2f    // ��ɫ����ɫ�Ĺ�����ʼ�� (0.0~1.0)  
#define WHITE_TO_GREEN_END      0.3f    // ��ɫ����ɫ�Ĺ��ɽ�����  
#define GREEN_TO_YELLOW_END     0.4f    // ��ɫ����ɫ�Ĺ��ɽ����㣨���ơ�����ʼ�㣩  

class PlaybackHeatMapGenerator {  
public:  
   cv::Mat qPixmapToCvMat(const QPixmap& pixmap);  
   cv::Mat loadBackgroundImage(const QString& path);  
   cv::Mat normalizeData(const cv::Mat& originalData);  
   void applyGaussianBlur(const cv::Mat& src, const cv::Mat& mask, cv::Mat& dst, cv::Mat& blurredMask);  
   cv::Mat createColorMap();  
   void applyColorMapManually(const cv::Mat& src, const cv::Mat& colorMap, cv::Mat& dst);  
   cv::Mat createGreenThresholdMask(const cv::Mat& blurredData);  
   cv::Mat calculateAlphaChannel(const cv::Mat& blurredData, const cv::Mat& blurredZeroMask);  
   bool createRGBAImage(const cv::Mat& rgbHeatmap, const cv::Mat& alphaChannel, cv::Mat& rgba);  
   bool blendWithBackground(const cv::Mat& rgbaHeatmap, cv::Mat& bgImage, cv::Mat& result);  
   cv::Mat applyColorMapping(const cv::Mat& blurredData);  
   bool generateHeatmapRGBA(const std::vector<int>& data, cv::Mat& rgbaHeatmap);  
   QPixmap convertToQPixmap(const cv::Mat& image);  
   QPixmap generateHeatMap(const std::vector<int>& data, const QPixmap& background);  

private:  
   void createDataMatrices(const std::vector<int>& data, cv::Mat& originalData, cv::Mat& zeroMask);  
};  

#endif // PLAYBACKHEATMAPGENERATOR_H