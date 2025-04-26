#include "PlaybackHeatMapGenerator.h"

QPixmap PlaybackHeatMapGenerator::generateHeatMap(const std::vector<int>& data, const QPixmap& background)
{
    if (data.size() != HEATMAP_SIZE * HEATMAP_SIZE)
    {
        //TPWLog(TPWLOG_MODULE_IPCAPP, TPWLOG_LEVEL_ERROR, "Invalid data size, expected %d, got %d\n", HEATMAP_SIZE * HEATMAP_SIZE, data.size());
        return QPixmap();
    }

    // ��QPixmapת��ΪOpenCV Mat
    cv::Mat bgImage = qPixmapToCvMat(background);
    if (bgImage.empty())
    {
        //TPWLog(TPWLOG_MODULE_IPCAPP, TPWLOG_LEVEL_ERROR, "Invalid background pixmap\n");
        return QPixmap();
    }

    // ����ԭʼ������������ͼRGBA
    cv::Mat rgbaHeatmap;
    if (!generateHeatmapRGBA(data, rgbaHeatmap))
    {
        //TPWLog(TPWLOG_MODULE_IPCAPP, TPWLOG_LEVEL_ERROR, "Failed to generate heatmap\n");
        return QPixmap();
    }

    // ��������ͼ�ߴ粢��ϵ�����ͼ
    cv::Mat blendedImage;
    if (!blendWithBackground(rgbaHeatmap, bgImage, blendedImage))
    {
        //TPWLog(TPWLOG_MODULE_IPCAPP, TPWLOG_LEVEL_ERROR, "Failed to blend images\n");
        return QPixmap();
    }

    // ����Ƕ��������ͼ������ͼ
    return convertToQPixmap(blendedImage);
}

cv::Mat PlaybackHeatMapGenerator::qPixmapToCvMat(const QPixmap& pixmap)
{
    QImage image = pixmap.toImage().convertToFormat(QImage::Format_RGBA8888);
    cv::Mat mat(image.height(), image.width(), CV_8UC4, (void*)image.constBits(), image.bytesPerLine());
    cv::Mat result;
    cv::cvtColor(mat, result, cv::COLOR_RGBA2BGRA); // ת��ΪOpenCV��׼BGRA
    return result;
}

void PlaybackHeatMapGenerator::createDataMatrices(const std::vector<int>& data, cv::Mat& originalData, cv::Mat& zeroMask)
{
    originalData.create(HEATMAP_SIZE, HEATMAP_SIZE, CV_32F);  // �������ݾ���
    zeroMask.create(HEATMAP_SIZE, HEATMAP_SIZE, CV_32F);      // ��ֵ��Ĥ����

    for (int i = 0; i < HEATMAP_SIZE; ++i)
    {
        auto* origRow = originalData.ptr<float>(i);
        auto* maskRow = zeroMask.ptr<float>(i);
        for (int j = 0; j < HEATMAP_SIZE; ++j)
        {
            const int value = data[i * HEATMAP_SIZE + j];
            origRow[j] = static_cast<float>(value);
            maskRow[j] = (value == 0) ? 0.0f : 1.0f;    // ������ֵ��Ĥ�����ڽ�����ͼ0ֵ������λ��ɫ
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
    cv::minMaxLoc(originalData, &minVal, &maxVal);  // ��ȡ������С/���ֵ

    // �����쳣�������
    if (maxVal <= minVal)
    {
        return cv::Mat::zeros(originalData.size(), CV_32F);
    }
    return (originalData - minVal) / (maxVal - minVal);  // ִ�����Թ�һ��
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

        // �׶�1: �ס��� (BGR: 255,255,255 �� 0,255,0)
        if (norm <= WHITE_TO_GREEN_END)
        {
            if (norm < WHITE_TO_GREEN_START)
            {
                color = cv::Vec3b(255, 255, 255); // ����
            }
            else
            {
                float t = (norm - WHITE_TO_GREEN_START) / (WHITE_TO_GREEN_END - WHITE_TO_GREEN_START);
                color = cv::Vec3b(
                    255 * (1 - t),  // Bͨ����255��0
                    255,            // G����255
                    255 * (1 - t)   // Rͨ����255��0
                );
            }
        }
        // �׶�2: �̡��� (BGR: 0,255,0 �� 0,255,255)
        else if (norm <= GREEN_TO_YELLOW_END)
        {
            float t = (norm - WHITE_TO_GREEN_END) / (GREEN_TO_YELLOW_END - WHITE_TO_GREEN_END);
            color = cv::Vec3b(
                0,          // B����0
                255,        // G����255
                255 * t     // R��0��255
            );
        }
        // �׶�3: �ơ��� (BGR: 0,255,255 �� 0,0,255)
        else
        {
            float t = (norm - GREEN_TO_YELLOW_END) / (1 - GREEN_TO_YELLOW_END);
            color = cv::Vec3b(
                0,              // B����0
                255 * (1 - t),  // G��255��0
                255             // R����255
            );
        }

        colorMap.at<cv::Vec3b>(i, 0) = color;
    }
    return colorMap;
}

void PlaybackHeatMapGenerator::applyColorMapManually(const cv::Mat& src, const cv::Mat& colorMap, cv::Mat& dst)
{
    // ����ʹ��OpenCV��LUT��������ʹ�ô˷����ֶ�ӳ�䣬���������������ϴ�ĳ���
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
    // ��Ĥ���ã�
    // 1. ������ֵ�������޳�����������
    // 2. ���˵�ֵ������������������ɫ��ֵ����Ч��������
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
                maskRow[x] = 0.0f;  // �������ޣ���ȫ͸��
            }
            else if (val < upperBound)
            {
                maskRow[x] = (val - lowerBound) / TRANSITION_WIDTH;  // ���Թ���
            }
            else
            {
                maskRow[x] = 1.0f;  // �������ޣ���ȫ��͸��
            }
        }
    }
    cv::GaussianBlur(mask, mask, cv::Size(0, 0), BLUR_SIGMA_MASK);
    return mask;
}

cv::Mat PlaybackHeatMapGenerator::calculateAlphaChannel(const cv::Mat& blurredData, const cv::Mat& blurredZeroMask)
{
    cv::Mat alpha(HEATMAP_SIZE, HEATMAP_SIZE, CV_8U);
    cv::Mat greenMask = createGreenThresholdMask(blurredData);  // ������ɫ��Ĥ������ƽ������������

    cv::Mat combinedMask;
    cv::multiply(greenMask, blurredZeroMask, combinedMask);     // ��Ͼ�������������Ĥ

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
    // ͳһ�ߴ�����߼�
    cv::Mat resizedHeatmap;
    const cv::Size targetSize(bgImage.cols, bgImage.rows);
    cv::resize(rgbaHeatmap, resizedHeatmap, targetSize, 0, 0, cv::INTER_LINEAR);

    // ȷ��������ʽ��ȷ
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

    // �� OpenCV 4.x �У�֧�ֵ�ͨ������ + ��ͨ����ɫ��
    //try {
    //    cv::LUT(heatmap8U, colorMap, coloredHeatmap);
    //}
    //catch (const cv::Exception&) {
    //    applyColorMapManually(heatmap8U, colorMap, coloredHeatmap);
    //}

    // ����Ŀǰ�汾Ϊ OpenCV 3.4.5 �������ֶ�����ͨ������ת��Ϊ��ͨ���󣬲ſ�ʹ��LUT���������ܸ�ǿ��
    cv::Mat heatmap8U_3C;
    cv::cvtColor(heatmap8U, heatmap8U_3C, cv::COLOR_GRAY2BGR); // ��ͨ��ת��ͨ��
    cv::LUT(heatmap8U_3C, colorMap, coloredHeatmap);           // �������ɫ���Ϊ��ͨ��
    return coloredHeatmap;
}

bool PlaybackHeatMapGenerator::generateHeatmapRGBA(const std::vector<int>& data, cv::Mat& rgbaHeatmap)
{
    cv::Mat originalData, zeroMask;
    // ��dataת��ΪOpenCV�����ʽ
    createDataMatrices(data, originalData, zeroMask);
    // ��һ��
    cv::Mat normalizedData = normalizeData(originalData);
    if (normalizedData.empty())
    {
        return false;
    }

    // ��˹ģ��
    cv::Mat blurredData, blurredZeroMask;
    applyGaussianBlur(normalizedData, zeroMask, blurredData, blurredZeroMask);

    // ��ɫӳ��
    cv::Mat coloredHeatmap = applyColorMapping(blurredData);
    // ͸���ȼ���
    cv::Mat alphaChannel = calculateAlphaChannel(blurredData, blurredZeroMask);

    // ����ͨ����RGB����ͼ�뵥ͨ����Alpha͸����ͨ���ϲ�Ϊ��ͨ���� RGBA ͼ��ʵ�ִ���͸���ȵ�����ͼ�����
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