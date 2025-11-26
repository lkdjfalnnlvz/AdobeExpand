#ifndef _WEBM_PREMIERE_IMPORT_H_
#define _WEBM_PREMIERE_IMPORT_H_

#include "PrSDKImport.h"
#include "PrSDKTypes.h"
#include "PrSDKPPixCreatorSuite.h"
#include "PrSDKPPixCacheSuite.h"
#include "PrSDKPPixSuite.h"
#include "PrSDKTimeSuite.h"

extern "C" {
    #include <libavformat/avformat.h>
    #include <libavcodec/avcodec.h>
    #include <libswscale/swscale.h>
    #include <libavutil/imgutils.h>
    #include <libavutil/opt.h>
}

typedef struct
{
    csSDK_int32             importerID;
    
    AVFormatContext*        fmt_ctx;       
    
    int                     videoStreamIdx;
    AVCodecContext*         videoCodecCtx;  
    AVStream*               videoStream;  
    SwsContext*             swsCtx;        
    
    int                     audioStreamIdx;
    AVCodecContext*         audioCodecCtx;
    AVStream*               audioStream;

    csSDK_int32             width;
    csSDK_int32             height;
    int64_t                 duration;
    PlugMemoryFuncsPtr      memFuncs;
    SPBasicSuite            *BasicSuite;
    PrSDKPPixCreatorSuite   *PPixCreatorSuite;
    PrSDKPPixCacheSuite     *PPixCacheSuite;
    PrSDKPPixSuite          *PPixSuite;
    PrSDKTimeSuite          *TimeSuite;

} ImporterLocalRec8, *ImporterLocalRec8Ptr, **ImporterLocalRec8H;

#endif

