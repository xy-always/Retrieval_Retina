#ifndef CSP_API_H
#define CSP_API_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define CSP_MAX_STR_LEN         (256)

typedef struct {
    uint32_t x;         // 图像左上角第一个像素的x坐标
    uint32_t y;         // 图像左上角第一个像素的y坐标
    uint32_t width;     // 图像宽度
    uint32_t height;    // 图像高度
    void* data;         // 图像数据
    uint32_t dataLen;   // 图像数据长度
} CspImageInfo;

typedef struct {
    uint32_t tileWidth;         // 切块基本宽度，典型值256
    uint32_t tileHeight;        // 切块基本高度，典型值256
    uint32_t imageWidth;        // 扫描图像宽度
    uint32_t imageHeight;       // 扫描图像高度
    float scanRatio;            // 扫描倍率，如40.0, 20.0等
    uint8_t downsamplingMode;   // 下采样模式
    float downsamplingRatio;    // 下采样倍率
    float mpp;                  // 每个像素的微米数(microns per pixel)
} CspConfig;

typedef struct {
    char manufacturer[CSP_MAX_STR_LEN];     // 设备厂商
    char modelName[CSP_MAX_STR_LEN];        // 设备型号
    char serialNumber[CSP_MAX_STR_LEN];     // 设备序列号
    char softwareVersion[CSP_MAX_STR_LEN];  // 扫描仪软件版本
} CspScannerInfo;

/* ============================================== 阅片接口 =================================================== */
/**
 * @brief 创建csp读句柄，先获取句柄后，才能进行后续操作
 * @param[in]       fileName        csp文件路径，必须为以\0结尾的字符串
 * @return          为NULL表示创建失败，非NULL表示创建成功
 */
void* GetCspReader(const char* fileName);

/**
 * @brief 销毁csp读句柄，所有操作执行完毕后被调用，销毁后csp读句柄将不可用
 * @param[in]       fp          csp读句柄
 * @return          void
 */
void DestroyCspReader(void* fp);

/**
 * @brief 读取预览图
 * @param[in]       fp          csp读句柄
 * @param[out]      prevImg     预览图信息
 *                              sdk内部申请prevImg->data，并由接口CspDestroyImage负责释放该内存
 * @return          0表示成功，非0表示失败
 */
int32_t CspReadPreview(void* fp, CspImageInfo* prevImg);

/**
 * @brief 读取标签图
 * @param[in]       fp          csp读句柄
 * @param[out]      labelImg    标签图信息
 *                              sdk内部申请labelImg->data，并由接口CspDestroyImage负责释放该内存
 * @return          0表示成功，非0表示失败
 */
int32_t CspReadLabel(void* fp, CspImageInfo* labelImg);

/**
 * @brief 读取缩略图
 * @param[in]       fp          csp读句柄
 * @param[out]      thumbImg    缩略图信息
 *                              sdk内部申请thumbImg->data，并由接口CspDestroyImage负责释放该内存
 * @return          0表示成功，非0表示失败
 */
int32_t CspReadThumb(void* fp, CspImageInfo* thumbImg);

/**
 * @brief 销毁图像块内存
 * @param[in]       img         待销毁的图像块
 * @return          void
 */
void CspDestroyImage(CspImageInfo* img);

/**
 * @brief 读取扫描仪信息
 * @param[in]       fp          csp读句柄
 * @param[out]      cfg         扫描配置信息
 * @return          0表示成功，非0表示失败
 */
int32_t CspReadScannerInfo(void* fp, CspScannerInfo* scannerInfo);

/**
 * @brief 读取扫描配置信息
 * @param[in]       fp          csp读句柄
 * @param[out]      cfg         扫描配置信息
 * @return          0表示成功，非0表示失败
 */
int32_t CspReadConfig(void* fp, CspConfig* cfg);

/**
 * @brief 获取扫描层数
 * @param[in]       fp          csp读句柄
 * @param[out]      layerNum    扫描层数
 * @return          0表示成功，非0表示失败
 */
int32_t CspGetLayerNum(void* fp, uint32_t* layerNum);

/**
 * @brief 读取一个区域的数据，返回数据为JPEG格式
 * @param[in]       fp          csp读句柄
 * @param[in]       scale       图像块的扫描倍率
 * @param[in/out]   img         图像块信息，需要设置请求的x, y, width, height，
 *                              sdk内部申请img->data，并由接口CspDestroyImage负责释放该内存
 * @return          0表示成功，非0表示失败
 */
int32_t CspReadImg(void* fp, float scale, CspImageInfo* img);

/* ============================================== 扫描接口 =================================================== */
/**
 * @brief 创建csp写句柄
 * @param[in]       fileName    csp文件名
 * @param[in]       parallelNum 控制内部并发数量
 * @return          为NULL表示创建失败，非NULL表示创建成功
 */
void* GetCspWriter(const char* fileName, uint32_t parallelNum);

/**
 * @brief 销毁csp写句柄
 * @param[in]       fp          csp写句柄
 * @return          void
 */
void DestroyCspWriter(void* fp);

/**
 * @brief 写入扫描仪信息
 * @param[in]       fp                  csp写句柄
 * @param[in]       manufacturer        设备厂商
 * @param[in]       modelName           设备型号
 * @param[in]       serialNumber        设备序列号
 * @param[in]       softwareVersion     扫描仪软件版本
 * @return          0表示成功，非0表示失败
 */
int32_t CspWriteEquipInfo(void* fp, CspScannerInfo* scannerInfo);

/**
 * @brief 写入预览图
 * @param[in]       fp              csp写句柄
 * @param[in]       width           预览图宽
 * @param[in]       height          预览图高
 * @param[in]       dataLen         预览图数据字节长度
 * @param[in]       data            预览图数据
 * @return          0表示成功，非0表示失败
 */
int32_t CspWritePrevImg(void* fp, uint32_t width, uint32_t height, uint32_t dataLen, const void* data);

/**
 * @brief 写入标签图
 * @param[in]       fp              csp写句柄
 * @param[in]       width           标签图宽
 * @param[in]       height          标签图高
 * @param[in]       dataLen         标签图数据字节长度
 * @param[in]       data            标签图数据
 * @return          0表示成功，非0表示失败
 */
int32_t CspWriteLabelImg(void* fp, uint32_t width, uint32_t height, uint32_t dataLen, const void* data);

/**
 * @brief 写入缩略图
 * @param[in]       fp              csp写句柄
 * @param[in]       width           缩略图宽
 * @param[in]       height          缩略图高
 * @param[in]       dataLen         缩略图数据字节长度
 * @param[in]       data            缩略图数据
 * @return          0表示成功，非0表示失败
 */
int32_t CspWriteThumbImg(void* fp, uint32_t width, uint32_t height, uint32_t dataLen, const void* data);

/**
 * @brief 设置扫描配置信息
 * @param[in]       fp              csp写句柄
 * @param[in]       tileWidth       图像块切块标准宽度，如256
 * @param[in]       tileHeight      图像块切块标准高度，如256
 * @param[in]       imgWidth        图像宽度
 * @param[in]       imgHeight       图像高度
 * @param[in]       scale           最高帧扫描倍率，如40.0
 * @return          0表示成功，非0表示失败
 */
int32_t CspSetScanConfig(void* fp, CspConfig* cfg);

/**
 * @brief 添加ROI信息
 * @param[in]       fp              csp写句柄
 * @param[in]       x               ROI起始坐标X
 * @param[in]       y               ROI起始坐标Y
 * @param[in]       width           ROI宽度
 * @param[in]       height          ROI高度
 * @return          0表示成功，非0表示失败
 */
int32_t CspAddRoiInfo(void* fp, uint32_t x, uint32_t y, uint32_t width, uint32_t height);

/**
 * @brief 写入单个图像块，图像块的宽高必须与扫描配置信息中设置的一致（边界位置除外）
 * @param[in]       fp              csp写句柄
 * @param[in]       tile            要写入的图像块信息
 * @return          0表示成功，非0表示失败
 */
int32_t CspWriteImageSingle(void* fp, const CspImageInfo* tile);

/**
 * @brief 写入一组图像块，图像块的宽高必须与扫描配置信息中设置的一致（边界位置除外）
 * @param[in]       fp              csp写句柄
 * @param[in]       tileNum         图像块数目
 * @param[in]       tiles           要写入的图像块信息
 * @return          0表示成功，非0表示失败
 */
int32_t CspWriteImageMulti(void* fp, uint32_t tileNum, const CspImageInfo* tiles);

/**
 * @brief 结束写入数据
 * @param[in]       fp          csp写句柄
 * @return          0表示成功，非0表示失败
 */
int32_t CspFlushFile(void* fp);

/**
 * @brief 扫描过程中停止扫描，停止后直接调用DestroyCspWriter进行销毁，停止后不保证生成的文件可用
 * @param[in]       fp          csp写句柄
 * @return          void
 */
void CspStopWrite(void* fp);

#ifdef __cplusplus
}
#endif

#endif // CSP_API_H
