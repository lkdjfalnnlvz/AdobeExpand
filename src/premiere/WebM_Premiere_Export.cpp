///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2013-2024, Brendan Bolles
// 
// All rights reserved.
// 
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
// *	   Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
// *	   Redistributions in binary form must reproduce the above
// copyright notice, this list of conditions and the following disclaimer
// in the documentation and/or other materials provided with the
// distribution.
// 
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
///////////////////////////////////////////////////////////////////////////

// ------------------------------------------------------------------------
//
// WebM plug-in for Premiere
//
// by Brendan Bolles <brendan@fnordware.com>
//
// ------------------------------------------------------------------------


#include "WebM_Premiere_Export.h"

#include "WebM_Premiere_Export_Params.h"
#include "WebM_Premiere_Codecs.h"


#ifdef PRMAC_ENV
	#include <mach/mach.h>
#else
	#include <algorithm>

	#include <assert.h>
	#include <time.h>
	#include <math.h>

	#define LONG_LONG_MAX LLONG_MAX
#endif

#include <string>
#include <queue>


#include "mkvmuxer/mkvmuxer.h"



class PrMkvWriter : public mkvmuxer::IMkvWriter
{
  public:
	PrMkvWriter(PrSDKExportFileSuite *fileSuite, csSDK_uint32 fileObject);
	virtual ~PrMkvWriter();
	
	virtual int32_t Write(const void* buf, uint32_t len);
	virtual int64_t Position() const;
	virtual int32_t Position(int64_t position); // seek
	virtual bool Seekable() const { return true; }
	virtual void ElementStartNotify(uint64_t element_id, int64_t position);
	
  private:
	const PrSDKExportFileSuite *_fileSuite;
	const csSDK_uint32 _fileObject;
};

PrMkvWriter::PrMkvWriter(PrSDKExportFileSuite *fileSuite, csSDK_uint32 fileObject) :
	_fileSuite(fileSuite),
	_fileObject(fileObject)
{
	prSuiteError err = _fileSuite->Open(_fileObject);
	
	if(err != malNoError)
		throw err;
}

PrMkvWriter::~PrMkvWriter()
{
	prSuiteError err = _fileSuite->Close(_fileObject);
	
	assert(err == malNoError);
}

int32_t
PrMkvWriter::Write(const void* buf, uint32_t len)
{
	prSuiteError err = _fileSuite->Write(_fileObject, (void *)buf, len);
	
	return err;
}

int64_t
PrMkvWriter::Position() const
{
	prInt64 pos = 0;

#if kPrSDKExportFileSuiteVersion == kPrSDKExportFileSuiteVersion1
#define PR_SEEK_CURRENT fileSeekMode_End // there was a bug in Premiere - fileSeekMode_End was really fileSeekMode_Current
#else
#define PR_SEEK_CURRENT fileSeekMode_Current
#endif

	prSuiteError err = _fileSuite->Seek(_fileObject, 0, pos, PR_SEEK_CURRENT);
	
	if(err != malNoError)
		throw err;
	
	return pos;
}

int32_t
PrMkvWriter::Position(int64_t position)
{
	prInt64 pos = 0;

	prSuiteError err = _fileSuite->Seek(_fileObject, position, pos, fileSeekMode_Begin);
	
	return err;
}

void
PrMkvWriter::ElementStartNotify(uint64_t element_id, int64_t position)
{
	// ummm, should I do something?
}


#pragma mark-


static const csSDK_int32 WebM_ID = 'WebM';
static const csSDK_int32 WebM_Export_Class = 'WebM';

extern int g_num_cpus;


// http://matroska.org/technical/specs/notes.html#TimecodeScale
// Time (in nanoseconds) = TimeCode * TimeCodeScale
// When we call finctions like GetTime, we're given Time in Nanoseconds.
static const uint64_t S2NS = 1000000000LL;


static void
utf16ncpy(prUTF16Char *dest, const char *src, int max_len)
{
	prUTF16Char *d = dest;
	const char *c = src;
	
	do{
		*d++ = *c;
	}while(*c++ != '\0' && --max_len);
}

static void
ncpyUTF16(char *dest, const prUTF16Char *src, int max_len)
{
	char *d = dest;
	const prUTF16Char *c = src;
	
	do{
		*d++ = *c;
	}while(*c++ != '\0' && --max_len);
}


static prMALError
exSDKStartup(
	exportStdParms		*stdParmsP, 
	exExporterInfoRec	*infoRecP)
{
	int fourCC = 0;
	VersionInfo version = {0, 0, 0};

	PrSDKAppInfoSuite *appInfoSuite = NULL;
	stdParmsP->getSPBasicSuite()->AcquireSuite(kPrSDKAppInfoSuite, kPrSDKAppInfoSuiteVersion, (const void**)&appInfoSuite);
	
	if(appInfoSuite)
	{
		appInfoSuite->GetAppInfo(PrSDKAppInfoSuite::kAppInfo_AppFourCC, (void *)&fourCC);

		appInfoSuite->GetAppInfo(PrSDKAppInfoSuite::kAppInfo_Version, (void *)&version);
	
		stdParmsP->getSPBasicSuite()->ReleaseSuite(kPrSDKAppInfoSuite, kPrSDKAppInfoSuiteVersion);
		
		// not a good idea to try to run a MediaCore exporter in AE
		if(fourCC == kAppAfterEffects)
			return exportReturn_IterateExporterDone;
	}
	

	infoRecP->fileType			= WebM_ID;
	
	utf16ncpy(infoRecP->fileTypeName, "WebM", 255);
	utf16ncpy(infoRecP->fileTypeDefaultExtension, "webm", 255);
	
	infoRecP->classID = WebM_Export_Class;
	
	infoRecP->exportReqIndex	= 0;
	infoRecP->wantsNoProgressBar = kPrFalse;
	infoRecP->hideInUI			= kPrFalse;
	infoRecP->doesNotSupportAudioOnly = kPrFalse;
	infoRecP->canExportVideo	= kPrTrue;
	infoRecP->canExportAudio	= kPrTrue;
	infoRecP->singleFrameOnly	= kPrFalse;
	
	infoRecP->interfaceVersion	= EXPORTMOD_VERSION;
	
	infoRecP->isCacheable		= kPrFalse;
	
	if(stdParmsP->interfaceVer >= 6 &&
		((fourCC == kAppPremierePro && version.major >= 9) ||
		 (fourCC == kAppMediaEncoder && version.major >= 9)))
	{
	#if EXPORTMOD_VERSION >= 6
		infoRecP->canConformToMatchParams = kPrTrue;
	#else
		// in earlier SDKs, we'll cheat and set this ourselves
		csSDK_uint32 *info = &infoRecP->isCacheable;
		info[1] = kPrTrue; // one spot past isCacheable
	#endif
	}

	VideoEncoder::initialize();

	return malNoError;
}


static prMALError
exSDKBeginInstance(
	exportStdParms			*stdParmsP, 
	exExporterInstanceRec	*instanceRecP)
{
	prMALError				result				= malNoError;
	SPErr					spError				= kSPNoError;
	ExportSettings			*mySettings;
	PrSDKMemoryManagerSuite	*memorySuite;
	csSDK_int32				exportSettingsSize	= sizeof(ExportSettings);
	SPBasicSuite			*spBasic			= stdParmsP->getSPBasicSuite();
	
	if(spBasic != NULL)
	{
		spError = spBasic->AcquireSuite(
			kPrSDKMemoryManagerSuite,
			kPrSDKMemoryManagerSuiteVersion,
			const_cast<const void**>(reinterpret_cast<void**>(&memorySuite)));
			
		mySettings = reinterpret_cast<ExportSettings *>(memorySuite->NewPtrClear(exportSettingsSize));

		if(mySettings)
		{
			mySettings->spBasic		= spBasic;
			mySettings->memorySuite	= memorySuite;
			
			spError = spBasic->AcquireSuite(
				kPrSDKExportParamSuite,
				kPrSDKExportParamSuiteVersion,
				const_cast<const void**>(reinterpret_cast<void**>(&(mySettings->exportParamSuite))));
			spError = spBasic->AcquireSuite(
				kPrSDKExportFileSuite,
				kPrSDKExportFileSuiteVersion,
				const_cast<const void**>(reinterpret_cast<void**>(&(mySettings->exportFileSuite))));
			spError = spBasic->AcquireSuite(
				kPrSDKExportInfoSuite,
				kPrSDKExportInfoSuiteVersion,
				const_cast<const void**>(reinterpret_cast<void**>(&(mySettings->exportInfoSuite))));
			spError = spBasic->AcquireSuite(
				kPrSDKExportProgressSuite,
				kPrSDKExportProgressSuiteVersion,
				const_cast<const void**>(reinterpret_cast<void**>(&(mySettings->exportProgressSuite))));
			spError = spBasic->AcquireSuite(
				kPrSDKPPixCreatorSuite,
				kPrSDKPPixCreatorSuiteVersion,
				const_cast<const void**>(reinterpret_cast<void**>(&(mySettings->ppixCreatorSuite))));
			spError = spBasic->AcquireSuite(
				kPrSDKPPixSuite,
				kPrSDKPPixSuiteVersion,
				const_cast<const void**>(reinterpret_cast<void**>(&(mySettings->ppixSuite))));
			spError = spBasic->AcquireSuite(
				kPrSDKPPix2Suite,
				kPrSDKPPix2SuiteVersion,
				const_cast<const void**>(reinterpret_cast<void**>(&(mySettings->ppix2Suite))));
			spError = spBasic->AcquireSuite(
				kPrSDKSequenceRenderSuite,
				kPrSDKSequenceRenderSuiteVersion,
				const_cast<const void**>(reinterpret_cast<void**>(&(mySettings->sequenceRenderSuite))));
			spError = spBasic->AcquireSuite(
				kPrSDKSequenceAudioSuite,
				kPrSDKSequenceAudioSuiteVersion,
				const_cast<const void**>(reinterpret_cast<void**>(&(mySettings->sequenceAudioSuite))));
			spError = spBasic->AcquireSuite(
				kPrSDKTimeSuite,
				kPrSDKTimeSuiteVersion,
				const_cast<const void**>(reinterpret_cast<void**>(&(mySettings->timeSuite))));
			spError = spBasic->AcquireSuite(
				kPrSDKWindowSuite,
				kPrSDKWindowSuiteVersion,
				const_cast<const void**>(reinterpret_cast<void**>(&(mySettings->windowSuite))));
		}

		instanceRecP->privateData = reinterpret_cast<void*>(mySettings);
	}
	else
	{
		result = exportReturn_ErrMemory;
	}
	
	return result;
}


static prMALError
exSDKEndInstance(
	exportStdParms			*stdParmsP, 
	exExporterInstanceRec	*instanceRecP)
{
	prMALError				result		= malNoError;
	ExportSettings			*lRec		= reinterpret_cast<ExportSettings *>(instanceRecP->privateData);
	SPBasicSuite			*spBasic	= stdParmsP->getSPBasicSuite();
	PrSDKMemoryManagerSuite	*memorySuite;
	if(spBasic != NULL && lRec != NULL)
	{
		if(lRec->exportParamSuite)
		{
			result = spBasic->ReleaseSuite(kPrSDKExportParamSuite, kPrSDKExportParamSuiteVersion);
		}
		if(lRec->exportFileSuite)
		{
			result = spBasic->ReleaseSuite(kPrSDKExportFileSuite, kPrSDKExportFileSuiteVersion);
		}
		if(lRec->exportInfoSuite)
		{
			result = spBasic->ReleaseSuite(kPrSDKExportInfoSuite, kPrSDKExportInfoSuiteVersion);
		}
		if(lRec->exportProgressSuite)
		{
			result = spBasic->ReleaseSuite(kPrSDKExportProgressSuite, kPrSDKExportProgressSuiteVersion);
		}
		if(lRec->ppixCreatorSuite)
		{
			result = spBasic->ReleaseSuite(kPrSDKPPixCreatorSuite, kPrSDKPPixCreatorSuiteVersion);
		}
		if(lRec->ppixSuite)
		{
			result = spBasic->ReleaseSuite(kPrSDKPPixSuite, kPrSDKPPixSuiteVersion);
		}
		if(lRec->ppix2Suite)
		{
			result = spBasic->ReleaseSuite(kPrSDKPPix2Suite, kPrSDKPPix2SuiteVersion);
		}
		if(lRec->sequenceRenderSuite)
		{
			result = spBasic->ReleaseSuite(kPrSDKSequenceRenderSuite, kPrSDKSequenceRenderSuiteVersion);
		}
		if(lRec->sequenceAudioSuite)
		{
			result = spBasic->ReleaseSuite(kPrSDKSequenceAudioSuite, kPrSDKSequenceAudioSuiteVersion);
		}
		if(lRec->timeSuite)
		{
			result = spBasic->ReleaseSuite(kPrSDKTimeSuite, kPrSDKTimeSuiteVersion);
		}
		if(lRec->windowSuite)
		{
			result = spBasic->ReleaseSuite(kPrSDKWindowSuite, kPrSDKWindowSuiteVersion);
		}
		if(lRec->memorySuite)
		{
			memorySuite = lRec->memorySuite;
			memorySuite->PrDisposePtr(reinterpret_cast<PrMemoryPtr>(lRec));
			result = spBasic->ReleaseSuite(kPrSDKMemoryManagerSuite, kPrSDKMemoryManagerSuiteVersion);
		}
	}

	return result;
}


static prMALError
exSDKFileExtension(
	exportStdParms					*stdParmsP, 
	exQueryExportFileExtensionRec	*exportFileExtensionRecP)
{
	utf16ncpy(exportFileExtensionRecP->outFileExtension, "webm", 255);
		
	return malNoError;
}


static void get_framerate(PrTime ticksPerSecond, PrTime ticks_per_frame, exRatioValue *fps)
{
	PrTime frameRates[] = {	10, 15, 23,
							24, 25, 29,
							30, 48, 48,
							50, 59, 60};
													
	static const PrTime frameRateNumDens[][2] = {	{10, 1}, {15, 1}, {24000, 1001},
													{24, 1}, {25, 1}, {30000, 1001},
													{30, 1}, {48000, 1001}, {48, 1},
													{50, 1}, {60000, 1001}, {60, 1}};
	
	int frameRateIndex = -1;
	
	for(csSDK_int32 i=0; i < sizeof(frameRates) / sizeof (PrTime); i++)
	{
		frameRates[i] = ticksPerSecond / frameRateNumDens[i][0] * frameRateNumDens[i][1];
		
		if(ticks_per_frame == frameRates[i])
			frameRateIndex = i;
	}
	
	if(frameRateIndex >= 0)
	{
		fps->numerator = frameRateNumDens[frameRateIndex][0];
		fps->denominator = frameRateNumDens[frameRateIndex][1];
	}
	else
	{
		fps->numerator = 1001 * ticksPerSecond / ticks_per_frame;
		fps->denominator = 1001;
	}
}


static prMALError
exSDKExport(
	exportStdParms	*stdParmsP,
	exDoExportRec	*exportInfoP)
{
	prMALError					result					= malNoError;
	ExportSettings				*mySettings				= reinterpret_cast<ExportSettings*>(exportInfoP->privateData);
	PrSDKExportParamSuite		*paramSuite				= mySettings->exportParamSuite;
	PrSDKSequenceRenderSuite	*renderSuite			= mySettings->sequenceRenderSuite;
	PrSDKSequenceAudioSuite		*audioSuite				= mySettings->sequenceAudioSuite;
	PrSDKMemoryManagerSuite		*memorySuite			= mySettings->memorySuite;
	PrSDKPPixSuite				*pixSuite				= mySettings->ppixSuite;
	PrSDKPPix2Suite				*pix2Suite				= mySettings->ppix2Suite;


	PrTime ticksPerSecond = 0;
	mySettings->timeSuite->GetTicksPerSecond(&ticksPerSecond);
	
	
	csSDK_uint32 exID = exportInfoP->exporterPluginID;
	csSDK_int32 gIdx = 0;
	
	exParamValues widthP, heightP, pixelAspectRatioP, fieldTypeP, frameRateP;
	
	paramSuite->GetParamValue(exID, gIdx, ADBEVideoWidth, &widthP);
	paramSuite->GetParamValue(exID, gIdx, ADBEVideoHeight, &heightP);
	paramSuite->GetParamValue(exID, gIdx, ADBEVideoAspect, &pixelAspectRatioP);
	paramSuite->GetParamValue(exID, gIdx, ADBEVideoFieldType, &fieldTypeP);
	paramSuite->GetParamValue(exID, gIdx, ADBEVideoFPS, &frameRateP);
	
	exRatioValue fps;
	get_framerate(ticksPerSecond, frameRateP.value.timeValue, &fps);

	exParamValues sampleRateP, channelTypeP;
	paramSuite->GetParamValue(exID, gIdx, ADBEAudioRatePerSecond, &sampleRateP);
	paramSuite->GetParamValue(exID, gIdx, ADBEAudioNumChannels, &channelTypeP);
	
	PrAudioChannelType audioFormat = (PrAudioChannelType)channelTypeP.value.intValue;
	
	if(audioFormat < kPrAudioChannelType_Mono || audioFormat > kPrAudioChannelType_51)
		audioFormat = kPrAudioChannelType_Stereo;
	
	const int audioChannels = (audioFormat == kPrAudioChannelType_51 ? 6 :
								audioFormat == kPrAudioChannelType_Stereo ? 2 :
								audioFormat == kPrAudioChannelType_Mono ? 1 :
								2);
	
	exParamValues videoCodecP, av1codecP, methodP, videoQualityP, bitrateP, colorSpaceP, twoPassP, keyframeMaxDistanceP, samplingP, bitDepthP, alphaP, customArgsP;
		
	paramSuite->GetParamValue(exID, gIdx, WebMVideoCodec, &videoCodecP);
	paramSuite->GetParamValue(exID, gIdx, WebMAV1Codec, &av1codecP);
	paramSuite->GetParamValue(exID, gIdx, WebMVideoMethod, &methodP);
	paramSuite->GetParamValue(exID, gIdx, WebMVideoQuality, &videoQualityP);
	paramSuite->GetParamValue(exID, gIdx, WebMVideoBitrate, &bitrateP);
	paramSuite->GetParamValue(exID, gIdx, WebMColorSpace, &colorSpaceP);
	paramSuite->GetParamValue(exID, gIdx, WebMVideoTwoPass, &twoPassP);
	paramSuite->GetParamValue(exID, gIdx, WebMVideoKeyframeMaxDistance, &keyframeMaxDistanceP);
	paramSuite->GetParamValue(exID, gIdx, WebMVideoSampling, &samplingP);
	paramSuite->GetParamValue(exID, gIdx, WebMVideoBitDepth, &bitDepthP);
	paramSuite->GetParamValue(exID, gIdx, ADBEVideoAlpha, &alphaP);
	paramSuite->GetParamValue(exID, gIdx, WebMCustomArgs, &customArgsP);
	
	exParamValues versionP;
	paramSuite->GetParamValue(exID, gIdx, WebMPluginVersion, &versionP);
	
	if(versionP.value.intValue < 0x00010100)
		keyframeMaxDistanceP.value.intValue = 128;
	
	if(bitDepthP.value.intValue < 8)
		bitDepthP.value.intValue = 8;
	
	
	const WebM_Video_Codec video_codec = (WebM_Video_Codec)videoCodecP.value.intValue;
	const bool use_vp8 = (video_codec == WEBM_CODEC_VP8);
	AV1_Codec av1_codec = (AV1_Codec)av1codecP.value.intValue;
	const bool av1_auto = (av1_codec == AV1_CODEC_AUTO);
	const bool nvenc_codec = (video_codec == WEBM_CODEC_AV1 && av1_codec == AV1_CODEC_NVENC);
	const WebM_Video_Method method = (WebM_Video_Method)methodP.value.intValue;
	const WebM_Chroma_Sampling chroma = ((use_vp8 || nvenc_codec) ? WEBM_420 : (WebM_Chroma_Sampling)samplingP.value.intValue);
	const int bit_depth = (use_vp8 ? 8 :
							(nvenc_codec && bitDepthP.value.intValue == 12) ? 10 :
							bitDepthP.value.intValue);
	const WebM_ColorSpace colorSpace = (WebM_ColorSpace)colorSpaceP.value.intValue;
	const bool use_alpha = alphaP.value.intValue;

	char customArgs[256];
	ncpyUTF16(customArgs, customArgsP.paramString, 255);
	customArgs[255] = '\0';
	

	exParamValues audioCodecP, audioMethodP, audioQualityP, audioBitrateP;
	paramSuite->GetParamValue(exID, gIdx, WebMAudioCodec, &audioCodecP);
	paramSuite->GetParamValue(exID, gIdx, WebMAudioMethod, &audioMethodP);
	paramSuite->GetParamValue(exID, gIdx, WebMAudioQuality, &audioQualityP);
	paramSuite->GetParamValue(exID, gIdx, WebMAudioBitrate, &audioBitrateP);
	
	exParamValues autoBitrateP, opusBitrateP;
	paramSuite->GetParamValue(exID, gIdx, WebMOpusAutoBitrate, &autoBitrateP);
	paramSuite->GetParamValue(exID, gIdx, WebMOpusBitrate, &opusBitrateP);
	
	const WebM_Audio_Codec audio_codec = (WebM_Audio_Codec)audioCodecP.value.intValue;
	
	
	const PrPixelFormat yuv_format8 = (use_alpha ? PrPixelFormat_BGRA_4444_16u :
										chroma == WEBM_444 ? (colorSpace == WEBM_REC709 ? PrPixelFormat_VUYX_4444_8u_709 : PrPixelFormat_VUYX_4444_8u) :
										chroma == WEBM_422 ? (colorSpace == WEBM_REC709 ? PrPixelFormat_UYVY_422_8u_709 : PrPixelFormat_UYVY_422_8u_601) :
										(colorSpace == WEBM_REC709 ? PrPixelFormat_YUV_420_MPEG2_FRAME_PICTURE_PLANAR_8u_709 : PrPixelFormat_YUV_420_MPEG2_FRAME_PICTURE_PLANAR_8u_601));

	const PrPixelFormat yuv_format16 = PrPixelFormat_BGRA_4444_16u; // can't trust PrPixelFormat_VUYA_4444_16u, only 16-bit YUV format
	
	const PrPixelFormat yuv_format = (bit_depth > 8 ? yuv_format16 : yuv_format8);
	
	SequenceRender_ParamsRec renderParms;
	PrPixelFormat pixelFormats[] = { yuv_format,
									PrPixelFormat_BGRA_4444_16u, // must support BGRA, even if I don't want to
									PrPixelFormat_BGRA_4444_8u };
									
	renderParms.inRequestedPixelFormatArray = pixelFormats;
	renderParms.inRequestedPixelFormatArrayCount = 3;
	renderParms.inWidth = widthP.value.intValue;
	renderParms.inHeight = heightP.value.intValue;
	renderParms.inPixelAspectRatioNumerator = pixelAspectRatioP.value.ratioValue.numerator;
	renderParms.inPixelAspectRatioDenominator = pixelAspectRatioP.value.ratioValue.denominator;
	renderParms.inRenderQuality = (exportInfoP->maximumRenderQuality ? kPrRenderQuality_Max : kPrRenderQuality_High);
	renderParms.inFieldType = fieldTypeP.value.intValue;
	renderParms.inDeinterlace = kPrFalse;
	renderParms.inDeinterlaceQuality = (exportInfoP->maximumRenderQuality ? kPrRenderQuality_Max : kPrRenderQuality_High);
	renderParms.inCompositeOnBlack = (use_alpha ? kPrFalse : kPrTrue);;
	
	
	csSDK_uint32 videoRenderID = 0;
	
	if(exportInfoP->exportVideo)
	{
		result = renderSuite->MakeVideoRenderer(exID, &videoRenderID, frameRateP.value.timeValue);
	}
	
	csSDK_uint32 audioRenderID = 0;
	
	if(exportInfoP->exportAudio)
	{
		PrAudioChannelLabel *channelOrder = AudioEncoder::channelOrder(audio_codec, audioFormat);
		
		result = audioSuite->MakeAudioRenderer(exID,
												exportInfoP->startTime,
												audioChannels,
												channelOrder,
												kPrAudioSampleType_32BitFloat,
												sampleRateP.value.floatValue,
												&audioRenderID);
	}

	
	PrMemoryPtr vbr_buffer = NULL;
	size_t vbr_buffer_size = 0;
	
	PrMemoryPtr alpha_vbr_buffer = NULL;
	size_t alpha_vbr_buffer_size = 0;


	PrMkvWriter *writer = NULL;

	mkvmuxer::Segment *muxer_segment = NULL;
	
	
	try{

	const bool multipass = (exportInfoP->exportVideo && twoPassP.value.intValue &&
							VideoEncoder::twoPassCapable(video_codec, av1_codec, method, chroma, bit_depth, widthP.value.intValue, heightP.value.intValue, use_alpha));
	
	const int passes = (multipass ? 2 : 1);
	
	for(int pass = 0; pass < passes && result == malNoError; pass++)
	{
		const bool vbr_pass = (passes > 1 && pass == 0);
		
		const uint64_t alpha_id = 1;
		
		
		VideoEncoder *videoEncoder = NULL;
		VideoEncoder *videoAlphaEncoder = NULL;
		
		if(exportInfoP->exportVideo && result == malNoError)
		{
			const int rgbBitrate = (use_alpha ? (bitrateP.value.intValue * 2 / 3) : bitrateP.value.intValue);
		
			videoEncoder = VideoEncoder::makeEncoder(widthP.value.intValue, heightP.value.intValue, pixelAspectRatioP.value.ratioValue,
														fps,
														video_codec, av1_codec,
														method, videoQualityP.value.intValue, rgbBitrate,
														multipass, vbr_pass, vbr_buffer, vbr_buffer_size,
														keyframeMaxDistanceP.value.intValue, use_alpha,
														chroma, bit_depth,
														colorSpace, customArgs,
														pixSuite, pix2Suite, false);
			
			if(use_alpha)
			{
				const int alphaBitrate = (bitrateP.value.intValue / 3);
			
				videoAlphaEncoder = VideoEncoder::makeEncoder(widthP.value.intValue, heightP.value.intValue, pixelAspectRatioP.value.ratioValue,
																fps,
																video_codec, av1_codec,
																method, videoQualityP.value.intValue, alphaBitrate,
																multipass, vbr_pass, vbr_buffer, vbr_buffer_size,
																keyframeMaxDistanceP.value.intValue, true,
																chroma, bit_depth,
																colorSpace, customArgs,
																pixSuite, pix2Suite, true);
			}
		}


		PrTime videoEncoderTime = exportInfoP->startTime;
		
		
		if(passes > 1)
		{
			const std::string msg = (vbr_pass ? "Analyzing video" : "Encoding WebM movie");

			prUTF16Char utf_str[256];

			utf16ncpy(utf_str, msg.c_str(), 255);

			mySettings->exportProgressSuite->SetProgressString(exID, utf_str);
		}

		
		AudioEncoder *audioEncoder = NULL;
		
		float *pr_audio_buffer[6] = {NULL, NULL, NULL, NULL, NULL, NULL};
		
		PrAudioSample extraAudioSamples = 0; // Needed for Opus because of the delay
		
		csSDK_int32 maxBlip = 100;
		
		if(exportInfoP->exportAudio && !vbr_pass && result == malNoError)
		{
			mySettings->sequenceAudioSuite->GetMaxBlip(audioRenderID, frameRateP.value.timeValue, &maxBlip);
			
			audioEncoder = AudioEncoder::makeEncoder(audioChannels,
													sampleRateP.value.floatValue,
													audio_codec,
													(Ogg_Method)audioMethodP.value.intValue,
													audioQualityP.value.floatValue,
													audioBitrateP.value.intValue,
													autoBitrateP.value.intValue);
													
			for(int i=0; i < audioChannels; i++)
			{
				pr_audio_buffer[i] = (float *)malloc(sizeof(float) * maxBlip);
				
				if(pr_audio_buffer[i] == NULL)
					throw exportReturn_ErrMemory;
			}
			
			extraAudioSamples = (audioEncoder->getCodecDelay() * sampleRateP.value.floatValue / S2NS);
		}
		
		csSDK_int32 audioFrameSize = maxBlip;
		
		bool fullAudioFrames = false;
		
		if(exportInfoP->exportAudio && audio_codec == WEBM_CODEC_OPUS)
		{
			// For Vorbis we can send in maxBlip and get a bunch of packets out,
			// but for Opus each packet will be the length of the input, so we'll
			// customize it to be the number of samples in a frame.
			
			// Honestly not sure why I set it up this way, but...
			const csSDK_int32 samplesPerFrame = sampleRateP.value.floatValue * fps.denominator / fps.numerator;
			
			audioFrameSize = sampleRateP.value.floatValue / 400;
			
			while(audioFrameSize * 2 < samplesPerFrame && audioFrameSize * 2 < maxBlip)
			{
				audioFrameSize *= 2;
			}
			
			fullAudioFrames = true; // Always need full-size audio frames
		}
		
		
		if(result == malNoError)
		{
			// I'd say think about lowering this to get better precision,
			// but I get some messed up stuff when I do that.  Maybe a bug in the muxer?
			// The WebM spec says to keep it at one million:
			// http://www.webmproject.org/docs/container/#muxer-guidelines
			const uint64_t timeCodeScale = 1000000LL;
			
			uint64_t vid_track = 0;
			uint64_t audio_track = 0;
			
			if(!vbr_pass)
			{
				writer = new PrMkvWriter(mySettings->exportFileSuite, exportInfoP->fileObject);
				
				muxer_segment = new mkvmuxer::Segment;
				
				muxer_segment->Init(writer);
				muxer_segment->set_mode(mkvmuxer::Segment::kFile);
				
				
				mkvmuxer::SegmentInfo * const info = muxer_segment->GetSegmentInfo();
				
				
				info->set_writing_app("fnord WebM for Premiere, built " __DATE__);
				
				
				// date_utc is defined as the number of nanoseconds since the beginning of the millenium (1 Jan 2001)
				// http://www.matroska.org/technical/specs/index.html
				struct tm date_utc_base;
				memset(&date_utc_base, 0, sizeof(struct tm));
				
				date_utc_base.tm_year = 2001 - 1900;
				
				time_t base = mktime(&date_utc_base);
				
				info->set_date_utc( (int64_t)difftime(time(NULL), base) * S2NS );
				
				
				assert(info->timecode_scale() == timeCodeScale);
				
				assert(!muxer_segment->estimate_file_duration());
				
		
				if(exportInfoP->exportVideo)
				{
					vid_track = muxer_segment->AddVideoTrack(renderParms.inWidth, renderParms.inHeight, 1);
					
					mkvmuxer::VideoTrack * const video = static_cast<mkvmuxer::VideoTrack *>(muxer_segment->GetTrackByNumber(vid_track));
					
					video->set_frame_rate((double)fps.numerator / (double)fps.denominator);

					video->set_codec_id(video_codec == WEBM_CODEC_AV1 ? mkvmuxer::Tracks::kAv1CodecId :
											video_codec == WEBM_CODEC_VP9 ? mkvmuxer::Tracks::kVp9CodecId :
											mkvmuxer::Tracks::kVp8CodecId);
					
					size_t privateSize = 0;
					void *privateData = videoEncoder->getPrivateData(privateSize);
					
					if(privateData != nullptr && privateSize > 0)
					{
						const bool copied = video->SetCodecPrivate((const uint8_t *)privateData, privateSize);
						
						assert(copied);
					}
											
					if(renderParms.inPixelAspectRatioNumerator != renderParms.inPixelAspectRatioDenominator)
					{
						const uint64_t display_width = ((double)renderParms.inWidth *
														(double)renderParms.inPixelAspectRatioNumerator /
														(double)renderParms.inPixelAspectRatioDenominator)
														+ 0.5;
					
						video->set_display_width(display_width);
						video->set_display_height(renderParms.inHeight);
					}
					
					if(use_alpha)
					{
						video->SetAlphaMode(mkvmuxer::VideoTrack::kAlpha);
						video->set_max_block_additional_id(alpha_id);
					}
					
					muxer_segment->CuesTrack(vid_track);
					
					
					
					// Color metadata!
					// https://mailarchive.ietf.org/arch/search/?email_list=cellar&q=colour
					
					mkvmuxer::Colour color;
					
					color.set_bits_per_channel(bit_depth);
					
					color.set_chroma_subsampling_horz(chroma == WEBM_444 ? 0 : 1);
					color.set_chroma_subsampling_vert(chroma == WEBM_420 ? 1 : 0);
					
					color.set_matrix_coefficients(colorSpace == WEBM_REC709 ? mkvmuxer::Colour::kBt709 : mkvmuxer::Colour::kBt470bg); // BT.470 is Rec.601?
					color.set_range(mkvmuxer::Colour::kBroadcastRange);
					color.set_transfer_characteristics(mkvmuxer::Colour::kIturBt709Tc); // Rec.709 and 601 have the same curve
					color.set_primaries(mkvmuxer::Colour::kIturBt709P); // and the same primaries
					
					video->SetColour(color);
				}
				
				
				if(exportInfoP->exportAudio)
				{
					if(audio_codec == WEBM_CODEC_OPUS)
					{
						assert(sampleRateP.value.floatValue == 48000.f);
						
						sampleRateP.value.floatValue = 48000.f; // we'll just go ahead and enforce that
					}
				
					audio_track = muxer_segment->AddAudioTrack(sampleRateP.value.floatValue, audioChannels, 2);
					
					mkvmuxer::AudioTrack * const audio = static_cast<mkvmuxer::AudioTrack *>(muxer_segment->GetTrackByNumber(audio_track));
					
					audio->set_codec_id(audio_codec == WEBM_CODEC_OPUS ? mkvmuxer::Tracks::kOpusCodecId :
										mkvmuxer::Tracks::kVorbisCodecId);
					
					size_t privateSize = 0;
					void *privateData = audioEncoder->getPrivateData(privateSize);
					
					if(privateData != nullptr && privateSize > 0)
					{
						const bool copied = audio->SetCodecPrivate((const uint8_t *)privateData, privateSize);
						
						assert(copied);
					}
					
					
					const uint64_t seekPreRoll = audioEncoder->getSeekPreRoll();
					
					if(seekPreRoll > 0)
						audio->set_seek_pre_roll(seekPreRoll);
					
					
					const uint64_t codecDelay = audioEncoder->getCodecDelay();
					
					if(codecDelay > 0)
						audio->set_codec_delay(codecDelay);
					
					
					if(!exportInfoP->exportVideo)
						muxer_segment->CuesTrack(audio_track);
				}
			}
			
			PrAudioSample currentAudioSample = 0;

			// Here's a question: what do we do when the number of audio samples doesn't match evenly
			// with the number of frames?  This could especially happen when the user changes the frame
			// rate to something other than what Premiere is using to specify the out time.  So you could
			// have just enough time to pop up one more frame, but not enough time to fill the audio for
			// that frame.  What to do?  We could extend the movie duration to the whole frame, but right now
			// we'll just encode the amount of audio originally requested.  One ramification is that you could
			// be done encoding all your audio but still have a final frame to encode.
			const PrAudioSample endAudioSample = ((exportInfoP->endTime - exportInfoP->startTime) /
													(ticksPerSecond / (PrAudioSample)sampleRateP.value.floatValue)) +
													extraAudioSamples;
													
			assert(ticksPerSecond % (PrAudioSample)sampleRateP.value.floatValue == 0);
			
			const AudioEncoder::Packet *audioPacket = NULL;
			
			
			bool summary_packet_written = false;
			
			PrTime videoTime = exportInfoP->startTime;
			
			while(videoTime <= exportInfoP->endTime && result == malNoError)
			{
				const PrTime fileTime = videoTime - exportInfoP->startTime;
				
				// Time (in nanoseconds) = TimeCode * TimeCodeScale.
				const uint64_t timeCode = ((fileTime * (S2NS / timeCodeScale)) + (ticksPerSecond / 2)) / ticksPerSecond;
				
				const uint64_t timeStamp = timeCode * timeCodeScale;
			
				
				// When writing WebM, we want blocks of audio and video interleaved.
				// But encoders don't always cooperate with our wishes.  We feed them some data,
				// but they may not be ready to produce output right away.  So what we do is keep
				// feeding in the data until the output we want is produced.
				
				if(exportInfoP->exportAudio && !vbr_pass && result == malNoError)
				{
					const bool last_frame = (videoTime > (exportInfoP->endTime - frameRateP.value.timeValue));
					
					uint64_t currentAudioTimeStamp = 0;
					
					while((currentAudioTimeStamp <= timeStamp || last_frame) && (audioPacket != NULL || currentAudioSample < endAudioSample) && result == malNoError)
					{
						if(audioPacket == NULL)
							audioPacket = audioEncoder->getPacket();
						
						if(audioPacket == NULL)
						{
							if(currentAudioSample < endAudioSample)
							{
								int samples = audioFrameSize;
								int discardSamples = 0;
								
								if(samples > (endAudioSample - currentAudioSample))
								{
									if(fullAudioFrames)
										discardSamples = (samples - (endAudioSample - currentAudioSample));
									else
										samples = (endAudioSample - currentAudioSample);
								}
								
								result = audioSuite->GetAudio(audioRenderID, samples, pr_audio_buffer, false);
								
								if(result == malNoError)
								{
									const bool success = audioEncoder->compressSamples(pr_audio_buffer, samples, discardSamples);
									
									if(!success)
										result = exportReturn_InternalError;
										
									currentAudioSample += samples;
								}
							}
							else
							{
								const bool success = audioEncoder->endOfStream();
								
								if(!success)
									result = exportReturn_InternalError;
							}
							
							audioPacket = audioEncoder->getPacket();
						}
						
						if(audioPacket != NULL)
						{
							const uint64_t audioPacketTimeStamp = audioPacket->sampleIndex * S2NS / (uint64_t)sampleRateP.value.floatValue;
							
							assert((audioPacketTimeStamp > currentAudioTimeStamp) || audioPacketTimeStamp == 0);
							
							currentAudioTimeStamp = audioPacketTimeStamp;
							
							if((audioPacketTimeStamp <= timeStamp) || last_frame)
							{
								assert((audioPacket->sampleIndex < endAudioSample) || last_frame); // Vorbis pumps out extra packets at the end?
							
								bool added = false;
								
								if(audioPacket->discardSamples > 0)
								{
									const int64_t discardPadding = audioPacket->discardSamples * S2NS / (int64_t)sampleRateP.value.floatValue;
								
									added = muxer_segment->AddFrameWithDiscardPadding((const uint8_t *)audioPacket->data, audioPacket->size,
																						discardPadding, audio_track, audioPacketTimeStamp, true);
								}
								else
								{
									added = muxer_segment->AddFrame((const uint8_t *)audioPacket->data, audioPacket->size,
																	audio_track, audioPacketTimeStamp, true);
								}
																		
								if(added)
								{
									audioEncoder->returnPacket(audioPacket);
									
									audioPacket = audioEncoder->getPacket();
								}
								else
									result = exportReturn_InternalError;
							}
						}
					}
				}
				
				
				if(exportInfoP->exportVideo && (videoTime < exportInfoP->endTime) && result == malNoError) // there will some audio after the last video frame
				{
					bool made_frame = false;
					
					while(!made_frame && result == malNoError)
					{
						if(videoEncoder->packetReady() && (!use_alpha || videoAlphaEncoder->packetReady()))
						{
							if(vbr_pass)
							{
								const VideoEncoder::Packet *packet = videoEncoder->getPacket();
								
								if(vbr_buffer_size == 0)
									vbr_buffer = memorySuite->NewPtr(packet->size);
								else
									memorySuite->SetPtrSize(&vbr_buffer, vbr_buffer_size + packet->size);
								
								memcpy(&vbr_buffer[vbr_buffer_size], packet->data, packet->size);
								
								vbr_buffer_size += packet->size;
								
								videoEncoder->returnPacket(packet);
								
								made_frame = true;
								
								if(use_alpha)
								{
									const VideoEncoder::Packet *alphaPacket = videoAlphaEncoder->getPacket();
									
									if(alpha_vbr_buffer_size == 0)
										alpha_vbr_buffer = memorySuite->NewPtr(alphaPacket->size);
									else
										memorySuite->SetPtrSize(&alpha_vbr_buffer, alpha_vbr_buffer_size + alphaPacket->size);
									
									memcpy(&alpha_vbr_buffer[alpha_vbr_buffer_size], alphaPacket->data, alphaPacket->size);
									
									alpha_vbr_buffer_size += alphaPacket->size;
									
									videoAlphaEncoder->returnPacket(alphaPacket);
								}
							}
							else
							{
								const VideoEncoder::Packet *packet = videoEncoder->getPacket();
								
								assert( packet->time == (videoTime - exportInfoP->startTime) * fps.numerator / (ticksPerSecond * fps.denominator) );
								assert( packet->duration == 1 ); // because of how we did the timescale
								
								if(use_alpha)
								{
									const VideoEncoder::Packet *alphaPacket = videoAlphaEncoder->getPacket();
									
									assert( alphaPacket->time == (videoTime - exportInfoP->startTime) * fps.numerator / (ticksPerSecond * fps.denominator) );
									assert( alphaPacket->duration == 1 );
									
									assert(packet->keyframe == alphaPacket->keyframe);
									
									bool added = muxer_segment->AddFrameWithAdditional((const uint8_t *)packet->data, packet->size,
																						(const uint8_t *)alphaPacket->data, alphaPacket->size, alpha_id,
																						vid_track, timeStamp,
																						packet->keyframe);
																		
									made_frame = true;
									
									if(!added)
										result = exportReturn_InternalError;
									
									videoAlphaEncoder->returnPacket(alphaPacket);
								}
								else
								{
									bool added = muxer_segment->AddFrame((const uint8_t *)packet->data, packet->size,
																		vid_track, timeStamp,
																		packet->keyframe);
																		
									made_frame = true;
									
									if(!added)
										result = exportReturn_InternalError;
								}
								
								videoEncoder->returnPacket(packet);
							}
						}
							
							
						bool encode_more = !made_frame;
						
						if(vbr_pass)
						{
							// if that was the last VBR packet, we have to finalize and write a summary packet,
							// so go through the loop once more
							// but only call the encoder with NULL once during VBR pass
							if(videoEncoderTime == LONG_LONG_MAX)
								encode_more = false;
							
							if(videoTime >= (exportInfoP->endTime - frameRateP.value.timeValue) && (videoEncoderTime >= exportInfoP->endTime) && !summary_packet_written)
							{
								assert(made_frame == true); // encode_more = false
								
								made_frame = false; // to trick the loop into going around once more
								
								summary_packet_written = true; // it will be
								
								if(videoEncoderTime != LONG_LONG_MAX)
								{
									encode_more = true;
									
									assert(!videoEncoder->packetReady());
									assert(!use_alpha || !videoAlphaEncoder->packetReady());
								}
								else
								{
									assert(videoEncoder->packetReady());
									assert(!use_alpha || videoAlphaEncoder->packetReady());
								}
							}
						}
						
						if(encode_more && result == suiteError_NoError)
						{
							// this is for the encoder, which does its own math based on config.g_timebase
							// let's do the math
							// time = timestamp * timebase :: time = videoTime / ticksPerSecond : timebase = 1 / fps
							// timestamp = time / timebase
							// timestamp = (videoTime / ticksPerSecond) * (fps.num / fps.den)
							const PrTime encoder_fileTime = videoEncoderTime - exportInfoP->startTime;
							const PrTime encoder_nextFileTime = encoder_fileTime + frameRateP.value.timeValue;
							
							const PrTime encoder_timeStamp = encoder_fileTime * fps.numerator / (ticksPerSecond * fps.denominator);
							const PrTime encoder_nextTimeStamp = encoder_nextFileTime * fps.numerator / (ticksPerSecond * fps.denominator);
							const PrTime encoder_duration = encoder_nextTimeStamp - encoder_timeStamp;
							
							// BUT, if we're setting timebase to 1/fps, then timestamp is just frame number.
							// And since frame number isn't going to overflow at big times the way encoder_timeStamp is,
							// let's just use that.
							const PrTime encoder_FrameNumber = encoder_fileTime / frameRateP.value.timeValue;
							const PrTime encoder_FrameDuration = 1;
							
							// these asserts will not be true for big time values (int64_t overflow)
							if(videoEncoderTime < LONG_MAX)
							{
								assert(encoder_FrameNumber == encoder_timeStamp);
								assert(encoder_FrameDuration == encoder_duration);
							}
							
				
							if(videoEncoderTime < exportInfoP->endTime)
							{
								SequenceRender_GetFrameReturnRec renderResult;
								
								result = renderSuite->RenderVideoFrame(videoRenderID,
																		videoEncoderTime,
																		&renderParms,
																		kRenderCacheType_None,
																		&renderResult);
								
								if(result == suiteError_NoError)
								{
								#ifndef NDEBUG
									prRect bounds;
									csSDK_uint32 parN, parD;
									
									pixSuite->GetBounds(renderResult.outFrame, &bounds);
									pixSuite->GetPixelAspectRatio(renderResult.outFrame, &parN, &parD);
									
									const int width = bounds.right - bounds.left;
									const int height = bounds.bottom - bounds.top;
									
									assert(width == widthP.value.intValue);
									assert(height == heightP.value.intValue);
									assert(parN == pixelAspectRatioP.value.ratioValue.numerator);  // Premiere sometimes screws this up
									assert(parD == pixelAspectRatioP.value.ratioValue.denominator);
								#endif // NDEBUG
									
									bool encoded = videoEncoder->compressFrame(renderResult.outFrame, encoder_FrameNumber, encoder_FrameDuration);
									
									if(encoded)
									{
										videoEncoderTime += frameRateP.value.timeValue;
										
										if(use_alpha)
										{
											encoded = videoAlphaEncoder->compressFrame(renderResult.outFrame, encoder_FrameNumber, encoder_FrameDuration);
											
											if(!encoded)
												result = exportReturn_InternalError;
										}
									}
									else
										result = exportReturn_InternalError;
									
									
									pixSuite->Dispose(renderResult.outFrame);
								}
							}
							else
							{
								// squeeze the last bit out of the encoder
								bool squeezed = videoEncoder->endOfStream();
								
								if(squeezed)
								{
									videoEncoderTime = LONG_LONG_MAX;
								
									if(use_alpha)
									{
										squeezed = videoAlphaEncoder->endOfStream();
										
										if(!squeezed)
											result = exportReturn_InternalError;
									}
								}
								else
									result = exportReturn_InternalError;
							}
						}
					}
				}
				
				
				if(result == malNoError)
				{
					float progress = (double)(videoTime - exportInfoP->startTime) / (double)(exportInfoP->endTime - exportInfoP->startTime);
					
					if(passes == 2)
					{
						const float firstpass_frac = ((video_codec == WEBM_CODEC_VP8) ? 0.3f : 0.1f);
					
						if(pass == 1)
						{
							progress = firstpass_frac + (progress * (1.f - firstpass_frac));
						}
						else
							progress = (progress * firstpass_frac);
					}

					result = mySettings->exportProgressSuite->UpdateProgressPercent(exID, progress);
					
					if(result == suiteError_ExporterSuspended)
					{
						result = mySettings->exportProgressSuite->WaitForResume(exID);
					}
				}
				
				
				videoTime += frameRateP.value.timeValue;
			}
			
			
			if(muxer_segment != NULL)
			{
				assert(!vbr_pass);
				
				const PrTime endTime = std::min<PrTime>(videoTime, exportInfoP->endTime);
				
				const PrTime fileTimeDuration = endTime - exportInfoP->startTime;
					
				const uint64_t timeCodeDuration = ((fileTimeDuration * (S2NS / timeCodeScale)) + (ticksPerSecond / 2)) / ticksPerSecond;
				
				// Thanks, Frank!
				// https://bugs.chromium.org/p/webm/issues/detail?id=1100
				
				muxer_segment->set_duration(timeCodeDuration);
			}
			
			
			// audio sanity check
			if(result == malNoError && exportInfoP->exportAudio && !vbr_pass)
			{
			
			}
		}
		else
			result = exportReturn_InternalError;
		


		if(exportInfoP->exportVideo)
		{
			delete videoEncoder;
			delete videoAlphaEncoder;
		}
			
		if(exportInfoP->exportAudio && !vbr_pass)
		{
			for(int i=0; i < audioChannels; i++)
			{
				if(pr_audio_buffer[i] != NULL)
					free(pr_audio_buffer[i]);
			}
		
			delete audioEncoder;
		}
	}

	
	if(muxer_segment != NULL)
	{
		bool fin = muxer_segment->Finalize();
		
		if(!fin)
			result = exportReturn_InternalError;
	}
	
	
	}catch(...) { result = exportReturn_InternalError; }


	if(vbr_buffer != NULL)
		memorySuite->PrDisposePtr(vbr_buffer);

	if(alpha_vbr_buffer != NULL)
		memorySuite->PrDisposePtr(alpha_vbr_buffer);
	
	
	if(exportInfoP->exportVideo)
		renderSuite->ReleaseVideoRenderer(exID, videoRenderID);

	if(exportInfoP->exportAudio)
		audioSuite->ReleaseAudioRenderer(exID, audioRenderID);
	

	return result;
}


DllExport PREMPLUGENTRY xSDKExport (
	csSDK_int32		selector, 
	exportStdParms	*stdParmsP, 
	void			*param1, 
	void			*param2)
{
	prMALError result = exportReturn_Unsupported;
	
	switch (selector)
	{
		case exSelStartup:
			result = exSDKStartup(	stdParmsP, 
									reinterpret_cast<exExporterInfoRec*>(param1));
			break;

		case exSelBeginInstance:
			result = exSDKBeginInstance(stdParmsP,
										reinterpret_cast<exExporterInstanceRec*>(param1));
			break;

		case exSelEndInstance:
			result = exSDKEndInstance(	stdParmsP,
										reinterpret_cast<exExporterInstanceRec*>(param1));
			break;

		case exSelGenerateDefaultParams:
			result = exSDKGenerateDefaultParams(stdParmsP,
												reinterpret_cast<exGenerateDefaultParamRec*>(param1));
			break;

		case exSelPostProcessParams:
		{
			result = exSDKPostProcessParams(stdParmsP,
											reinterpret_cast<exPostProcessParamsRec*>(param1),
											VideoEncoder::haveCodec(AV1_CODEC_NVENC));
			break;
		}

		case exSelGetParamSummary:
			result = exSDKGetParamSummary(	stdParmsP,
											reinterpret_cast<exParamSummaryRec*>(param1));
			break;

		case exSelQueryOutputSettings:
			result = exSDKQueryOutputSettings(	stdParmsP,
												reinterpret_cast<exQueryOutputSettingsRec*>(param1));
			break;

		case exSelQueryExportFileExtension:
			result = exSDKFileExtension(stdParmsP,
										reinterpret_cast<exQueryExportFileExtensionRec*>(param1));
			break;

		case exSelValidateParamChanged:
			result = exSDKValidateParamChanged(	stdParmsP,
												reinterpret_cast<exParamChangedRec*>(param1));
			break;

		case exSelValidateOutputSettings:
			result = malNoError;
			break;

		case exSelExport:
			result = exSDKExport(	stdParmsP,
									reinterpret_cast<exDoExportRec*>(param1));
			break;
	}
	
	return result;
}
