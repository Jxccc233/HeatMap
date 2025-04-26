#include "PlaybackHeatMapGenerator.h"

QPixmap PlaybackHeatMapGenerator::generateHeatMap(const std::vector<int>& data, const QPixmap& background)
{
    if (data.size() != HEATMAP_SIZE * HEATMAP_SIZE)
    {
        //TPWLog(TPWLOG_MODULE_IPCAPP, TPWLOG_LEVEL_ERROR, "Invalid data size, expected %d, got %d\n", HEATMAP_SIZE * HEATMAP_SIZE, data.size());
        return QPixmap();
    }

    // 将QPixmap转换为OpenCV Mat
    cv::Mat bgImage = qPixmapToCvMat(background);
    if (bgImage.empty())
    {
        //TPWLog(TPWLOG_MODULE_IPCAPP, TPWLOG_LEVEL_ERROR, "Invalid background pixmap\n");
        return QPixmap();
    }

    // 处理原始数据生成热力图RGBA
    cv::Mat rgbaHeatmap;
    if (!generateHeatmapRGBA(data, rgbaHeatmap))
    {
        //TPWLog(TPWLOG_MODULE_IPCAPP, TPWLOG_LEVEL_ERROR, "Failed to generate heatmap\n");
        return QPixmap();
    }

    // 调整热力图尺寸并混合到缩略图
    cv::Mat blendedImage;
    if (!blendWithBackground(rgbaHeatmap, bgImage, blendedImage))
    {
        //TPWLog(TPWLOG_MODULE_IPCAPP, TPWLOG_LEVEL_ERROR, "Failed to blend images\n");
        return QPixmap();
    }

    // 返回嵌入了热力图的缩略图
    return convertToQPixmap(blendedImage);
}

cv::Mat PlaybackHeatMapGenerator::qPixmapToCvMat(const QPixmap& pixmap)
{
    QImage image = pixmap.toImage().convertToFormat(QImage::Format_RGBA8888);
    cv::Mat mat(image.height(), image.width(), CV_8UC4, (void*)image.constBits(), image.bytesPerLine());
    cv::Mat result;
    cv::cvtColor(mat, result, cv::COLOR_RGBA2BGRA); // 转换为OpenCV标准BGRA
    return result;
}

void PlaybackHeatMapGenerator::createDataMatrices(const std::vector<int>& data, cv::Mat& originalData, cv::Mat& zeroMask)
{
    originalData.create(HEATMAP_SIZE, HEATMAP_SIZE, CV_32F);  // 浮点数据矩阵
    zeroMask.create(HEATMAP_SIZE, HEATMAP_SIZE, CV_32F);      // 零值掩膜矩阵

    for (int i = 0; i < HEATMAP_SIZE; ++i)
    {
        auto* origRow = originalData.ptr<float>(i);
        auto* maskRow = zeroMask.ptr<float>(i);
        for (int j = 0; j < HEATMAP_SIZE; ++j)
        {
            const int value = data[i * HEATMAP_SIZE + j];
            origRow[j] = static_cast<float>(value);
            maskRow[j] = (value == 0) ? 0.0f : 1.0f;    // 生成零值掩膜，用于将热力图0值区域置位无色
        }
    }
}

cv::Mat PlaybackHeatMapGenerator::loadBackgroundImage(const QString& path)
{
    cv::Mat image = cv::imread(path.toStdString(), cv::IMREAD_UNCHANGED);
    if (image.empty()) return cv::Mat();

    if (image.channels() == 3)
    {
        cv::cvtColor(image, image, cv::COLOR_BGR2BGRA);
    }
    return image;
}

cv::Mat PlaybackHeatMapGenerator::normalizeData(const cv::Mat& originalData)
{
    double minVal, maxVal;
    cv::minMaxLoc(originalData, &minVal, &maxVal);  // 获取矩阵最小/最大值

    // 处理异常数据情况
    if (maxVal <= minVal)
    {
        return cv::Mat::zeros(originalData.size(), CV_32F);
    }
    return (originalData - minVal) / (maxVal - minVal);  // 执行线性归一化
}

void PlaybackHeatMapGenerator::applyGaussianBlur(const cv::Mat& src, const cv::Mat& mask, cv::Mat& dst, cv::Mat& blurredMask)
{
    const cv::Size kernelSize(0, 0);
    cv::GaussianBlur(src, dst, kernelSize, BLUR_SIGMA);
    cv::GaussianBlur(mask, blurredMask, kernelSize, BLUR_SIGMA);
}

cv::Mat PlaybackHeatMapGenerator::createColorMap()
{
    cv::Mat colorMap(256, 1, CV_8UC3);
    for (int i = 0; i < 256; ++i)
    {
        const float norm = i / 255.0f;
        cv::Vec3b color;

        // 阶段1: 白→绿 (BGR: 255,255,255 → 0,255,0)
        if (norm <= WHITE_TO_GREEN_END)
        {
            if (norm < WHITE_TO_GREEN_START)
            {
                color = cv::Vec3b(255, 255, 255); // 纯白
            }
            else
            {
                float t = (norm - WHITE_TO_GREEN_START) / (WHITE_TO_GREEN_END - WHITE_TO_GREEN_START);
                color = cv::Vec3b(
                    255 * (1 - t),  // B通道从255→0
                    255,            // G保持255
                    255 * (1 - t)   // R通道从255→0
                );
            }
        }
        // 阶段2: 绿→黄 (BGR: 0,255,0 → 0,255,255)
        else if (norm <= GREEN_TO_YELLOW_END)
        {
            float t = (norm - WHITE_TO_GREEN_END) / (GREEN_TO_YELLOW_END - WHITE_TO_GREEN_END);
            color = cv::Vec3b(
                0,          // B保持0
                255,        // G保持255
                255 * t     // R从0→255
            );
        }
        // 阶段3: 黄→红 (BGR: 0,255,255 → 0,0,255)
        else
        {
            float t = (norm - GREEN_TO_YELLOW_END) / (1 - GREEN_TO_YELLOW_END);
            color = cv::Vec3b(
                0,              // B保持0
                255 * (1 - t),  // G从255→0
                255             // R保持255
            );
        }

        colorMap.at<cv::Vec3b>(i, 0) = color;
    }
    return colorMap;
}

void PlaybackHeatMapGenerator::applyColorMapManually(const cv::Mat& src, const cv::Mat& colorMap, cv::Mat& dst)
{
    // 若不使用OpenCV的LUT方法，可使用此方法手动映射，不适用于数据量较大的场景
    dst.create(src.size(), CV_8UC3);
    for (int y = 0; y < src.rows; ++y)
    {
        const uchar* srcRow = src.ptr<uchar>(y);
        cv::Vec3b* dstRow = dst.ptr<cv::Vec3b>(y);
        for (int x = 0; x < src.cols; ++x)
        {
            dstRow[x] = colorMap.at<cv::Vec3b>(srcRow[x], 0);
        }
    }
}

cv::Mat PlaybackHeatMapGenerator::createGreenThresholdMask(const cv::Mat& blurredData)
{
    // 掩膜作用：
    // 1. 抑制零值背景：剔除无数据区域。
    // 2. 过滤低值噪声：仅保留高于绿色阈值的有效数据区域。
    cv::Mat mask(blurredData.size(), CV_32F);
    const float lowerBound = GREEN_THRESHOLD - TRANSITION_WIDTH / 2;
    const float upperBound = GREEN_THRESHOLD + TRANSITION_WIDTH / 2;

    for (int y = 0; y < mask.rows; ++y)
    {
        const float* dataRow = blurredData.ptr<float>(y);
        float* maskRow = mask.ptr<float>(y);
        for (int x = 0; x < mask.cols; ++x)
        {
            const float val = dataRow[x];

            if (val < lowerBound)
            {
                maskRow[x] = 0.0f;  // 低于下限：完全透明
            }
            else if (val < upperBound)
            {
                maskRow[x] = (val - lowerBound) / TRANSITION_WIDTH;  // 线性过渡
            }
            else
            {
                maskRow[x] = 1.0f;  // 高于上限：完全不透明
            }
        }
    }
    cv::GaussianBlur(mask, mask, cv::Size(0, 0), BLUR_SIGMA_MASK);
    return mask;
}

cv::Mat PlaybackHeatMapGenerator::calculateAlphaChannel(const cv::Mat& blurredData, const cv::Mat& blurredZeroMask)
{
    cv::Mat alpha(HEATMAP_SIZE, HEATMAP_SIZE, CV_8U);
    cv::Mat greenMask = createGreenThresholdMask(blurredData);  // 生成绿色掩膜，用于平滑掉杂乱数据

    cv::Mat combinedMask;
    cv::multiply(greenMask, blurredZeroMask, combinedMask);     // 混合矩阵，生成联合掩膜

    for (int y = 0; y < alpha.rows; ++y)
    {
        const float* dataRow = blurredData.ptr<float>(y);
        const float* maskRow = combinedMask.ptr<float>(y);
        uchar* alphaRow = alpha.ptr<uchar>(y);
        for (int x = 0; x < alpha.cols; ++x)
        {
            const float baseAlpha = ALPHA_MIN + dataRow[x] * ALPHA_RANGE;
            alphaRow[x] = cv::saturate_cast<uchar>(baseAlpha * maskRow[x]);
        }
    }
    return alpha;
}

bool PlaybackHeatMapGenerator::createRGBAImage(const cv::Mat& rgbHeatmap, const cv::Mat& alphaChannel, cv::Mat& rgba)
{
    if (rgbHeatmap.channels() != 3 || alphaChannel.type() != CV_8U)
    {
        return false;
    }

    std::vector<cv::Mat> channels(3);
    cv::split(rgbHeatmap, channels);
    channels.push_back(alphaChannel);
    cv::merge(channels, rgba);
    return true;
}

bool PlaybackHeatMapGenerator::blendWithBackground(const cv::Mat& rgbaHeatmap, cv::Mat& bgImage, cv::Mat& result)
{
    // 统一尺寸调整逻辑
    cv::Mat resizedHeatmap;
    const cv::Size targetSize(bgImage.cols, bgImage.rows);
    cv::resize(rgbaHeatmap, resizedHeatmap, targetSize, 0, 0, cv::INTER_LINEAR);

    // 确保背景格式正确
    if (bgImage.channels() == 3)
    {
        cv::cvtColor(bgImage, bgImage, cv::COLOR_BGR2BGRA);
    }
    else if (bgImage.channels() != 4)
    {
        TPWLog(TPWLOG_MODULE_IPCAPP, TPWLOG_LEVEL_ERROR, "Unsupported background channel count:%d\n", bgImage.channels());
        return false;
    }

    for (int y = 0; y < bgImage.rows; ++y)
    {
        cv::Vec4b* bg = bgImage.ptr<cv::Vec4b>(y);
        const cv::Vec4b* ht = resizedHeatmap.ptr<cv::Vec4b>(y);

        for (int x = 0; x < bgImage.cols; ++x)
        {
            const float alpha = ht[x][3] / 255.0f;
            bg[x][0] = cv::saturate_cast<uchar>(bg[x][0] * (1 - alpha) + ht[x][0] * alpha); // B
            bg[x][1] = cv::saturate_cast<uchar>(bg[x][1] * (1 - alpha) + ht[x][1] * alpha); // G
            bg[x][2] = cv::saturate_cast<uchar>(bg[x][2] * (1 - alpha) + ht[x][2] * alpha); // R
        }
    }

    bgImage.copyTo(result);
    return true;
}

cv::Mat PlaybackHeatMapGenerator::applyColorMapping(const cv::Mat& blurredData)
{
    cv::Mat colorMap = createColorMap();
    cv::Mat heatmap8U;
    blurredData.convertTo(heatmap8U, CV_8U, 255);
    cv::Mat coloredHeatmap;

    // 在 OpenCV 4.x 中，支持单通道输入 + 三通道颜色表
    //try {
    //    cv::LUT(heatmap8U, colorMap, coloredHeatmap);
    //}
    //catch (const cv::Exception&) {
    //    applyColorMapManually(heatmap8U, colorMap, coloredHeatmap);
    //}

    // 由于目前版本为 OpenCV 3.4.5 ，故需手动将单通道输入转换为三通道后，才可使用LUT函数（性能更强）
    cv::Mat heatmap8U_3C;
    cv::cvtColor(heatmap8U, heatmap8U_3C, cv::COLOR_GRAY2BGR); // 单通道转三通道
    cv::LUT(heatmap8U_3C, colorMap, coloredHeatmap);           // 输入和颜色表均为三通道
    return coloredHeatmap;
}

bool PlaybackHeatMapGenerator::generateHeatmapRGBA(const std::vector<int>& data, cv::Mat& rgbaHeatmap)
{
    cv::Mat originalData, zeroMask;
    // 将data转换为OpenCV矩阵格式
    createDataMatrices(data, originalData, zeroMask);
    // 归一化
    cv::Mat normalizedData = normalizeData(originalData);
    if (normalizedData.empty())
    {
        return false;
    }

    // 高斯模糊
    cv::Mat blurredData, blurredZeroMask;
    applyGaussianBlur(normalizedData, zeroMask, blurredData, blurredZeroMask);

    // 颜色映射
    cv::Mat coloredHeatmap = applyColorMapping(blurredData);
    // 透明度计算
    cv::Mat alphaChannel = calculateAlphaChannel(blurredData, blurredZeroMask);

    // 将三通道的RGB热力图与单通道的Alpha透明度通道合并为四通道的 RGBA 图像，实现带有透明度的热力图输出。
    return createRGBAImage(coloredHeatmap, alphaChannel, rgbaHeatmap);
}

QPixmap PlaybackHeatMapGenerator::convertToQPixmap(const cv::Mat& image)
{
    cv::Mat rgbaImage;
    cv::cvtColor(image, rgbaImage, cv::COLOR_BGRA2RGBA);

    QImage qtImage(
        rgbaImage.data,
        rgbaImage.cols,
        rgbaImage.rows,
        rgbaImage.step,
        QImage::Format_RGBA8888
    );
    return QPixmap::fromImage(qtImage.copy());
}