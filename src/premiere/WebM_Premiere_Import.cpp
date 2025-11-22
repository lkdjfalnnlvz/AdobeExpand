#include "WebM_Premiere_Import.h"
#include <vector>
#include <string>

#ifdef PRWIN_ENV
#include <windows.h>
std::string Utf16ToUtf8(const prUTF16Char* utf16) {
    if (!utf16) return "";
    int size_needed = WideCharToMultiByte(CP_UTF8, 0, (LPCWSTR)utf16, -1, NULL, 0, NULL, NULL);
    std::string strTo(size_needed, 0);
    WideCharToMultiByte(CP_UTF8, 0, (LPCWSTR)utf16, -1, &strTo[0], size_needed, NULL, NULL);
    if (!strTo.empty() && strTo.back() == 0) strTo.pop_back();
    return strTo;
}

static prMALError 
SDKGetIndFormat(imStdParms* stdParms, csSDK_size_t index, imIndFormatRec* SDKIndFormatRec)
{
    char formatname[255];
    char shortname[32];
    char platformXten[256];

    switch(index) {
        case 0: // MP4
            SDKIndFormatRec->filetype = 'MP4 ';
            strcpy(formatname, "MP4 Video (FFmpeg)"); strcpy(shortname, "MP4"); memcpy(platformXten, "mp4\0\0", 5);
            break;
        case 1: // MKV
            SDKIndFormatRec->filetype = 'MKV ';
            strcpy(formatname, "Matroska Video"); strcpy(shortname, "MKV"); memcpy(platformXten, "mkv\0\0", 5);
            break;
        case 2: // MOV
            SDKIndFormatRec->filetype = 'MooV';
            strcpy(formatname, "QuickTime MOV"); strcpy(shortname, "MOV"); memcpy(platformXten, "mov\0\0", 5);
            break;
        case 3: // WebM
            SDKIndFormatRec->filetype = 'WebM';
            strcpy(formatname, "WebM Video"); strcpy(shortname, "WebM"); memcpy(platformXten, "webm\0\0", 6);
            break;
        case 4: // AVI
            SDKIndFormatRec->filetype = 'AVI ';
            strcpy(formatname, "AVI Video"); strcpy(shortname, "AVI"); memcpy(platformXten, "avi\0\0", 5);
            break;
        case 5: // AV1 (Raw)
            SDKIndFormatRec->filetype = 'AV1 ';
            strcpy(formatname, "AV1 Raw Video"); strcpy(shortname, "AV1"); memcpy(platformXten, "av1\0\0", 5);
            break;
        default:
            return imBadFormatIndex;
    }

    SDKIndFormatRec->flags = xfCanImport | xfIsMovie;
    
    #ifdef PRWIN_ENV
    strcpy_s(SDKIndFormatRec->FormatName, 255, formatname);
    strcpy_s(SDKIndFormatRec->FormatShortName, 32, shortname);
    memcpy_s(SDKIndFormatRec->PlatformExtension, 256, platformXten, 256);
    #else
    strcpy(SDKIndFormatRec->FormatName, formatname);
    strcpy(SDKIndFormatRec->FormatShortName, shortname);
    strcpy(SDKIndFormatRec->PlatformExtension, platformXten);
    #endif

    return malNoError;
}

static prMALError 
SDKOpenFile8(imStdParms* stdParms, imFileRef* SDKfileRef, imFileOpenRec8* SDKfileOpenRec8)
{
    ImporterLocalRec8H localRecH = NULL;
    ImporterLocalRec8Ptr localRecP = NULL;

    if(SDKfileOpenRec8->privatedata)
    {
        localRecH = (ImporterLocalRec8H)SDKfileOpenRec8->privatedata;
        stdParms->piSuites->memFuncs->lockHandle(reinterpret_cast<char**>(localRecH));
        localRecP = reinterpret_cast<ImporterLocalRec8Ptr>(*localRecH);
    }
    else
    {
        localRecH = (ImporterLocalRec8H)stdParms->piSuites->memFuncs->newHandle(sizeof(ImporterLocalRec8));
        SDKfileOpenRec8->privatedata = (PrivateDataPtr)localRecH;
        
        stdParms->piSuites->memFuncs->lockHandle(reinterpret_cast<char**>(localRecH));
        localRecP = reinterpret_cast<ImporterLocalRec8Ptr>(*localRecH);
        
        localRecP->BasicSuite = stdParms->piSuites->utilFuncs->getSPBasicSuite();
        localRecP->BasicSuite->AcquireSuite(kPrSDKPPixCreatorSuite, kPrSDKPPixCreatorSuiteVersion, (const void**)&localRecP->PPixCreatorSuite);
        localRecP->BasicSuite->AcquireSuite(kPrSDKPPixCacheSuite, kPrSDKPPixCacheSuiteVersion, (const void**)&localRecP->PPixCacheSuite);
        localRecP->BasicSuite->AcquireSuite(kPrSDKPPixSuite, kPrSDKPPixSuiteVersion, (const void**)&localRecP->PPixSuite);
        localRecP->BasicSuite->AcquireSuite(kPrSDKTimeSuite, kPrSDKTimeSuiteVersion, (const void**)&localRecP->TimeSuite);
        localRecP->memFuncs = stdParms->piSuites->memFuncs;

        localRecP->fmt_ctx = NULL;
        localRecP->videoCodecCtx = NULL;
        localRecP->swsCtx = NULL;
    }

    if (localRecP->fmt_ctx) {
        stdParms->piSuites->memFuncs->unlockHandle(reinterpret_cast<char**>(localRecH));
        return malNoError;
    }

    std::string path8;
    #ifdef PRWIN_ENV
        path8 = Utf16ToUtf8(SDKfileOpenRec8->fileinfo.filepath);
    #else
    #endif

    if (avformat_open_input(&localRecP->fmt_ctx, path8.c_str(), NULL, NULL) < 0) {
        stdParms->piSuites->memFuncs->unlockHandle(reinterpret_cast<char**>(localRecH));
        return imFileOpenFailed;
    }

    if (avformat_find_stream_info(localRecP->fmt_ctx, NULL) < 0) {
        stdParms->piSuites->memFuncs->unlockHandle(reinterpret_cast<char**>(localRecH));
        return imFileOpenFailed;
    }

    localRecP->videoStreamIdx = -1;
    for (unsigned int i = 0; i < localRecP->fmt_ctx->nb_streams; i++) {
        if (localRecP->fmt_ctx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
            localRecP->videoStreamIdx = i;
            localRecP->videoStream = localRecP->fmt_ctx->streams[i];
            
            const AVCodec* codec = avcodec_find_decoder(localRecP->videoStream->codecpar->codec_id);
            if (!codec) continue;

            localRecP->videoCodecCtx = avcodec_alloc_context3(codec);
            avcodec_parameters_to_context(localRecP->videoCodecCtx, localRecP->videoStream->codecpar);

            localRecP->videoCodecCtx->thread_count = 0;

            if (avcodec_open2(localRecP->videoCodecCtx, codec, NULL) < 0) {
                continue;
            }

            localRecP->width = localRecP->videoCodecCtx->width;
            localRecP->height = localRecP->videoCodecCtx->height;
            localRecP->duration = localRecP->videoStream->duration;
            
            break;
        }
    }

    if (localRecP->videoStreamIdx == -1) {
        stdParms->piSuites->memFuncs->unlockHandle(reinterpret_cast<char**>(localRecH));
        return imFileHasNoImportableStreams;
    }

    stdParms->piSuites->memFuncs->unlockHandle(reinterpret_cast<char**>(localRecH));
    return malNoError;
}
static prMALError 
SDKGetInfo8(imStdParms* stdParms, imFileAccessRec8* fileAccessInfo8, imFileInfoRec8* SDKFileInfo8)
{
    ImporterLocalRec8H ldataH = reinterpret_cast<ImporterLocalRec8H>(SDKFileInfo8->privatedata);
    stdParms->piSuites->memFuncs->lockHandle(reinterpret_cast<char**>(ldataH));
    ImporterLocalRec8Ptr localRecP = reinterpret_cast<ImporterLocalRec8Ptr>(*ldataH);

    SDKFileInfo8->hasVideo = kPrTrue;
    SDKFileInfo8->hasAudio = kPrFalse;
    SDKFileInfo8->vidInfo.supportsAsyncIO = kPrFalse;
    SDKFileInfo8->vidInfo.supportsGetSourceVideo = kPrTrue;

    SDKFileInfo8->vidInfo.imageWidth = localRecP->width;
    SDKFileInfo8->vidInfo.imageHeight = localRecP->height;
    
    SDKFileInfo8->vidInfo.subType = PrPixelFormat_BGRA_4444_8u; 

    if (localRecP->videoStream->avg_frame_rate.num > 0) {
        SDKFileInfo8->vidScale = localRecP->videoStream->avg_frame_rate.num;
        SDKFileInfo8->vidSampleSize = localRecP->videoStream->avg_frame_rate.den;
    } else {
        SDKFileInfo8->vidScale = 30;
        SDKFileInfo8->vidSampleSize = 1;
    }

    PrTime ticksPerSec = 0;
    localRecP->TimeSuite->GetTicksPerSecond(&ticksPerSec);
    
    double durationSeconds = (double)localRecP->duration * av_q2d(localRecP->videoStream->time_base);
    SDKFileInfo8->vidDuration = (PrTime)(durationSeconds * ticksPerSec);

    SDKFileInfo8->vidInfo.pixelAspectNum = localRecP->videoCodecCtx->sample_aspect_ratio.num;
    SDKFileInfo8->vidInfo.pixelAspectDen = localRecP->videoCodecCtx->sample_aspect_ratio.den;
    if (SDKFileInfo8->vidInfo.pixelAspectNum == 0) {
        SDKFileInfo8->vidInfo.pixelAspectNum = 1;
        SDKFileInfo8->vidInfo.pixelAspectDen = 1;
    }

    stdParms->piSuites->memFuncs->unlockHandle(reinterpret_cast<char**>(ldataH));
    return malNoError;
}
static prMALError 
SDKGetSourceVideo(imStdParms* stdParms, imFileRef fileRef, imSourceVideoRec* sourceVideoRec)
{
    ImporterLocalRec8H ldataH = reinterpret_cast<ImporterLocalRec8H>(sourceVideoRec->inPrivateData);
    stdParms->piSuites->memFuncs->lockHandle(reinterpret_cast<char**>(ldataH));
    ImporterLocalRec8Ptr localRecP = reinterpret_cast<ImporterLocalRec8Ptr>(*ldataH);

    PrTime ticksPerSec = 0;
    localRecP->TimeSuite->GetTicksPerSecond(&ticksPerSec);

    int64_t targetPTS = av_rescale_q(sourceVideoRec->inFrameTime, 
                                     (AVRational){1, (int)ticksPerSec}, 
                                     localRecP->videoStream->time_base);

    av_seek_frame(localRecP->fmt_ctx, localRecP->videoStreamIdx, targetPTS, AVSEEK_FLAG_BACKWARD);
    avcodec_flush_buffers(localRecP->videoCodecCtx);

    AVPacket* packet = av_packet_alloc();
    AVFrame* frame = av_frame_alloc();
    bool frameFound = false;

    while (av_read_frame(localRecP->fmt_ctx, packet) >= 0) {
        if (packet->stream_index == localRecP->videoStreamIdx) {
            if (avcodec_send_packet(localRecP->videoCodecCtx, packet) == 0) {
                while (avcodec_receive_frame(localRecP->videoCodecCtx, frame) == 0) {
                    if (frame->pts >= targetPTS) {
                        
                        PPixHand ppix;
                        prRect theRect;
                        prSetRect(&theRect, 0, 0, localRecP->width, localRecP->height);
                        localRecP->PPixCreatorSuite->CreatePPix(&ppix, PrPPixBufferAccess_ReadWrite, PrPixelFormat_BGRA_4444_8u, &theRect);

                        char* pixData;
                        csSDK_int32 rowBytes;
                        localRecP->PPixSuite->GetPixels(ppix, PrPPixBufferAccess_ReadWrite, &pixData);
                        localRecP->PPixSuite->GetRowBytes(ppix, &rowBytes);

                        if (localRecP->swsCtx == NULL) {
                            localRecP->swsCtx = sws_getContext(
                                localRecP->width, localRecP->height, localRecP->videoCodecCtx->pix_fmt,
                                localRecP->width, localRecP->height, AV_PIX_FMT_BGRA,
                                SWS_BILINEAR, NULL, NULL, NULL
                            );
                        }

                        uint8_t* dest[4] = { (uint8_t*)pixData, NULL, NULL, NULL };
                        int destLinesize[4] = { rowBytes, 0, 0, 0 };
                        
                        sws_scale(localRecP->swsCtx, frame->data, frame->linesize, 0, localRecP->height, dest, destLinesize);

                        *sourceVideoRec->outFrame = ppix;
                        frameFound = true;
                        goto cleanup;
                    }
                }
            }
        }
        av_packet_unref(packet);
    }

cleanup:
    av_packet_free(&packet);
    av_frame_free(&frame);
    stdParms->piSuites->memFuncs->unlockHandle(reinterpret_cast<char**>(ldataH));

    return frameFound ? malNoError : imFileReadFailed;
}
static prMALError 
SDKCloseFile(imStdParms* stdParms, imFileRef* SDKfileRef, void* privateData) 
{
    ImporterLocalRec8H ldataH = reinterpret_cast<ImporterLocalRec8H>(privateData);
    if(ldataH && *ldataH) {
        stdParms->piSuites->memFuncs->lockHandle(reinterpret_cast<char**>(ldataH));
        ImporterLocalRec8Ptr localRecP = reinterpret_cast<ImporterLocalRec8Ptr>(*ldataH);

        if (localRecP->swsCtx) sws_freeContext(localRecP->swsCtx);
        if (localRecP->videoCodecCtx) avcodec_free_context(&localRecP->videoCodecCtx);
        if (localRecP->fmt_ctx) avformat_close_input(&localRecP->fmt_ctx);

        if(localRecP->BasicSuite) {
            localRecP->BasicSuite->ReleaseSuite(kPrSDKPPixCreatorSuite, kPrSDKPPixCreatorSuiteVersion);
            localRecP->BasicSuite->ReleaseSuite(kPrSDKPPixCacheSuite, kPrSDKPPixCacheSuiteVersion);
            localRecP->BasicSuite->ReleaseSuite(kPrSDKPPixSuite, kPrSDKPPixSuiteVersion);
            localRecP->BasicSuite->ReleaseSuite(kPrSDKTimeSuite, kPrSDKTimeSuiteVersion);
        }

        stdParms->piSuites->memFuncs->disposeHandle(reinterpret_cast<PrMemoryHandle>(ldataH));
    }
    return malNoError;
}
static prMALError 
SDKGetIndPixelFormat(imStdParms* stdParms, csSDK_size_t idx, imIndPixelFormatRec* SDKIndPixelFormatRec) 
{
    if(idx == 0) {
        SDKIndPixelFormatRec->outPixelFormat = PrPixelFormat_BGRA_4444_8u;
        return malNoError;
    }
    return imBadFormatIndex;
}
static prMALError 
SDKInit(imStdParms* stdParms, imImportInfoRec* importInfo)
{
    importInfo->canSave          = kPrFalse;
    importInfo->canDelete        = kPrFalse;
    importInfo->canCalcSizes     = kPrFalse;
    importInfo->canTrim          = kPrFalse;
    importInfo->hasSetup         = kPrFalse;
    importInfo->setupOnDblClk    = kPrFalse;
    importInfo->dontCache        = kPrFalse;
    importInfo->keepLoaded       = kPrFalse;
    importInfo->priority         = 0;
    importInfo->avoidAudioConform = kPrTrue; 

    return malNoError;
}

static prMALError 
SDKQuietFile(imStdParms* stdParms, imFileRef* SDKfileRef, void* privateData)
{
    ImporterLocalRec8H ldataH = reinterpret_cast<ImporterLocalRec8H>(privateData);
    if(ldataH && *ldataH) {
        stdParms->piSuites->memFuncs->lockHandle(reinterpret_cast<char**>(ldataH));
        ImporterLocalRec8Ptr localRecP = reinterpret_cast<ImporterLocalRec8Ptr>(*ldataH);

        if (localRecP->swsCtx) {
            sws_freeContext(localRecP->swsCtx);
            localRecP->swsCtx = NULL;
        }
        if (localRecP->videoCodecCtx) {
            avcodec_free_context(&localRecP->videoCodecCtx);
            localRecP->videoCodecCtx = NULL;
        }
        if (localRecP->fmt_ctx) {
            avformat_close_input(&localRecP->fmt_ctx);
            localRecP->fmt_ctx = NULL;
        }

        stdParms->piSuites->memFuncs->unlockHandle(reinterpret_cast<char**>(ldataH));
    }
    return malNoError;
}

static prMALError 
SDKAnalysis(imStdParms* stdParms, imFileRef SDKfileRef, imAnalysisRec* SDKAnalysisRec)
{
    return malNoError;
}

PREMPLUGENTRY DllExport xImportEntry (
    csSDK_int32     selector, 
    imStdParms      *stdParms, 
    void            *param1, 
    void            *param2)
{
    prMALError result = imUnsupported;

    try {
        switch (selector)
        {
            case imInit:
                result = SDKInit(stdParms, reinterpret_cast<imImportInfoRec*>(param1));
                break;

            case imGetInfo8:
                result = SDKGetInfo8(stdParms, 
                                     reinterpret_cast<imFileAccessRec8*>(param1), 
                                     reinterpret_cast<imFileInfoRec8*>(param2));
                break;

            case imOpenFile8:
                result = SDKOpenFile8(stdParms, 
                                      reinterpret_cast<imFileRef*>(param1), 
                                      reinterpret_cast<imFileOpenRec8*>(param2));
                break;
            
            case imQuietFile:
                result = SDKQuietFile(stdParms, 
                                      reinterpret_cast<imFileRef*>(param1), 
                                      param2); 
                break;

            case imCloseFile:
                result = SDKCloseFile(stdParms, 
                                      reinterpret_cast<imFileRef*>(param1), 
                                      param2);
                break;

            case imAnalysis:
                result = SDKAnalysis(stdParms,
                                     reinterpret_cast<imFileRef>(param1),
                                     reinterpret_cast<imAnalysisRec*>(param2));
                break;

            case imGetIndFormat:
                result = SDKGetIndFormat(stdParms, 
                                         reinterpret_cast<csSDK_size_t>(param1),
                                         reinterpret_cast<imIndFormatRec*>(param2));
                break;

            case imGetIndPixelFormat:
                result = SDKGetIndPixelFormat(stdParms,
                                              reinterpret_cast<csSDK_size_t>(param1),
                                              reinterpret_cast<imIndPixelFormatRec*>(param2));
                break;

            case imGetSupports8:
                result = malSupports8;
                break;

            case imGetPreferredFrameSize:
                result = imIterateFrameSizes; 
                break;

            case imGetSourceVideo:
                result = SDKGetSourceVideo(stdParms,
                                           reinterpret_cast<imFileRef>(param1),
                                           reinterpret_cast<imSourceVideoRec*>(param2));
                break;
                
            case imImportAudio7:
                result = imUnsupported; 
                break;

            case imCreateAsyncImporter:
                result = imUnsupported;
                break;
        }
    
    } catch(...) { 
        result = imOtherErr; 
    }

    return result;
}
