///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2013, Brendan Bolles
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


#include "vpx/vpx_encoder.h"
#include "vpx/vp8cx.h"

#include "aom/aom_codec.h"
#include "aom/aom_encoder.h"
#include "aom/aomcx.h"

#ifdef WEBM_HAVE_NVENC
#include <cuda.h>
#include <cuda_runtime.h>
CUdevice cudaDevice;

#include <nvEncodeAPI.h>
NV_ENCODE_API_FUNCTION_LIST nvenc = { 0 };
#endif

#include <vorbis/codec.h>
#include <vorbis/vorbisenc.h>

#include "opus_multistream.h"

#include "mkvmuxer/mkvmuxer.h"

void aom_to_vpx_img(vpx_image_t *vpx_img, const aom_image_t *aom_img);


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

static int
mylog2(int val)
{
	int ret = 0;
	
	while( pow(2.0, ret) < val )
	{
		ret++;
	}
	
	return ret;
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

#ifdef WEBM_HAVE_NVENC
	int driverVersion = 0;
	cudaError_t cudaErr = cudaDriverGetVersion(&driverVersion);

	if(cudaErr == cudaSuccess && driverVersion >= CUDART_VERSION)
	{
		int deviceNum = -1;
		cudaErr = cudaGetDevice(&deviceNum);

		if(cudaErr == cudaSuccess)
		{
			CUresult cuErr = cuDeviceGet(&cudaDevice, deviceNum);

			if(cuErr == CUDA_SUCCESS)
			{
				assert(nvenc.version == 0);

				const uint32_t sdkVersion = (NVENCAPI_MAJOR_VERSION << 4) | NVENCAPI_MINOR_VERSION;

				uint32_t version = 0;
				NVENCSTATUS nverr = NvEncodeAPIGetMaxSupportedVersion(&version);

				if(nverr == NV_ENC_SUCCESS && sdkVersion <= version)
				{
					nvenc.version = NV_ENCODE_API_FUNCTION_LIST_VER;

					nverr = NvEncodeAPICreateInstance(&nvenc);

					if(nverr != NV_ENC_SUCCESS)
						nvenc.version = 0;
				}
			}
		}
	}
#endif

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


// converting from Adobe 16-bit to regular 16-bit
#define PF_HALF_CHAN16			16384

static inline unsigned short
Promote(const unsigned short &val)
{
	return (val > PF_HALF_CHAN16 ? ( (val - 1) << 1 ) + 1 : val << 1);
}


template <typename BGRA_PIX, typename IMG_PIX>
static inline IMG_PIX
DepthConvert(const BGRA_PIX &val, const int &depth);

template<>
static inline unsigned short
DepthConvert<unsigned short, unsigned short>(const unsigned short &val, const int &depth)
{
	return (Promote(val) >> (16 - depth));
}

template<>
static inline unsigned short
DepthConvert<unsigned char, unsigned short>(const unsigned char &val, const int &depth)
{
	return ((unsigned short)val << (depth - 8)) | (val >> (16 - depth));
}

template<>
static inline unsigned char
DepthConvert<unsigned short, unsigned char>(const unsigned short &val, const int &depth)
{
	assert(depth == 8);
	return ( (((long)(val) * 255) + 16384) / 32768);
}

template<>
static inline unsigned char
DepthConvert<unsigned char, unsigned char>(const unsigned char &val, const int &depth)
{
	assert(depth == 8);
	return val;
}


template <typename VUYA_PIX, typename IMG_PIX>
static void
CopyVUYAToVPXImg(vpx_image_t *img, vpx_image_t *alpha_img, const char *frameBufferP, const csSDK_int32 rowbytes)
{
	const unsigned int sub_x = img->x_chroma_shift + 1;
	const unsigned int sub_y = img->y_chroma_shift + 1;
	
	for(int y = 0; y < img->d_h; y++)
	{
		IMG_PIX *imgY = (IMG_PIX *)(img->planes[VPX_PLANE_Y] + (img->stride[VPX_PLANE_Y] * y));
		IMG_PIX *imgU = (IMG_PIX *)(img->planes[VPX_PLANE_U] + (img->stride[VPX_PLANE_U] * (y / sub_y)));
		IMG_PIX *imgV = (IMG_PIX *)(img->planes[VPX_PLANE_V] + (img->stride[VPX_PLANE_V] * (y / sub_y)));
	
		const VUYA_PIX *prVUYA = (VUYA_PIX *)(frameBufferP + (rowbytes * (img->d_h - 1 - y)));
		
		const VUYA_PIX *prV = prVUYA + 0;
		const VUYA_PIX *prU = prVUYA + 1;
		const VUYA_PIX *prY = prVUYA + 2;
		
		for(int x=0; x < img->d_w; x++)
		{
			*imgY++ = DepthConvert<VUYA_PIX, IMG_PIX>(*prY, img->bit_depth);
			
			if( (y % sub_y == 0) && (x % sub_x == 0) )
			{
				*imgU++ = DepthConvert<VUYA_PIX, IMG_PIX>(*prU, img->bit_depth);
				*imgV++ = DepthConvert<VUYA_PIX, IMG_PIX>(*prV, img->bit_depth);
			}
			
			prY += 4;
			prU += 4;
			prV += 4;
		}
	}
	
	if(alpha_img != NULL)
	{
		assert(sub_x == alpha_img->x_chroma_shift + 1);
		assert(sub_y == alpha_img->y_chroma_shift + 1);
		
		for(int y = 0; y < img->d_h; y++)
		{
			IMG_PIX *imgY = (IMG_PIX *)(alpha_img->planes[VPX_PLANE_Y] + (alpha_img->stride[VPX_PLANE_Y] * y));
			IMG_PIX *imgU = (IMG_PIX *)(alpha_img->planes[VPX_PLANE_U] + (alpha_img->stride[VPX_PLANE_U] * (y / sub_y)));
			IMG_PIX *imgV = (IMG_PIX *)(alpha_img->planes[VPX_PLANE_V] + (alpha_img->stride[VPX_PLANE_V] * (y / sub_y)));
		
			const VUYA_PIX *prVUYA = (VUYA_PIX *)(frameBufferP + (rowbytes * (alpha_img->d_h - 1 - y)));
			
			const VUYA_PIX *prA = prVUYA + 3;
			
			for(int x=0; x < img->d_w; x++)
			{
				*imgY++ = DepthConvert<VUYA_PIX, IMG_PIX>(*prA, img->bit_depth);
				
				if( (y % sub_y == 0) && (x % sub_x == 0) )
				{
					*imgU++ = DepthConvert<unsigned char, IMG_PIX>(128, alpha_img->bit_depth);
					*imgV++ = DepthConvert<unsigned char, IMG_PIX>(128, alpha_img->bit_depth);
				}
				
				prA += 4;
			}
		}
	}
}


template <typename BGRA_PIX, typename IMG_PIX, bool isARGB>
static void
CopyBGRAToVPXImg(vpx_image_t *img, vpx_image_t *alpha_img, const char *frameBufferP, const csSDK_int32 rowbytes)
{
	const unsigned int sub_x = img->x_chroma_shift + 1;
	const unsigned int sub_y = img->y_chroma_shift + 1;

	for(int y = 0; y < img->d_h; y++)
	{
		IMG_PIX *imgY = (IMG_PIX *)(img->planes[VPX_PLANE_Y] + (img->stride[VPX_PLANE_Y] * y));
		IMG_PIX *imgU = (IMG_PIX *)(img->planes[VPX_PLANE_U] + (img->stride[VPX_PLANE_U] * (y / sub_y)));
		IMG_PIX *imgV = (IMG_PIX *)(img->planes[VPX_PLANE_V] + (img->stride[VPX_PLANE_V] * (y / sub_y)));
		
		const BGRA_PIX *prBGRA = (BGRA_PIX *)(frameBufferP + (rowbytes * (img->d_h - 1 - y)));
		
		const BGRA_PIX *prB = prBGRA + 0;
		const BGRA_PIX *prG = prBGRA + 1;
		const BGRA_PIX *prR = prBGRA + 2;
		
		if(isARGB)
		{
			// Media Encoder CS5 insists on handing us this format in some cases,
			// even though we didn't list it as an option
			prR = prBGRA + 1;
			prG = prBGRA + 2;
			prB = prBGRA + 3;
		}
		
		// These are the pixels below the current one for MPEG-2 chroma siting
		const BGRA_PIX *prBb = prB - (rowbytes / sizeof(BGRA_PIX));
		const BGRA_PIX *prGb = prG - (rowbytes / sizeof(BGRA_PIX));
		const BGRA_PIX *prRb = prR - (rowbytes / sizeof(BGRA_PIX));
		
		// unless this is the last line and there is no pixel below
		if(y == (img->d_h - 1) || sub_y != 2)
		{
			prBb = prB;
			prGb = prG;
			prRb = prR;
		}
		
		
		// using the conversion found here: http://www.fourcc.org/fccyvrgb.php
		// and 601 spec here: http://www.itu.int/dms_pubrec/itu-r/rec/bt/R-REC-BT.601-7-201103-I!!PDF-E.pdf
		
		// these are part of the RGBtoYUV math (uses Adobe 16-bit)
		const int Yadd = (sizeof(BGRA_PIX) > 1 ? 20565000 : 165000);    // to be divided by 10000
		const int UVadd = (sizeof(BGRA_PIX) > 1 ? 164495000 : 1285000); // includes extra 5000 for rounding
		
		for(int x=0; x < img->d_w; x++)
		{
			*imgY++ = DepthConvert<BGRA_PIX, IMG_PIX>( ((2568 * (int)*prR) + (5041 * (int)*prG) + ( 979 * (int)*prB) + Yadd) / 10000, img->bit_depth);
			
			if(sub_y > 1)
			{
				if( (y % sub_y == 0) && (x % sub_x == 0) )
				{
					*imgV++ = DepthConvert<BGRA_PIX, IMG_PIX>( (((4392 * (int)*prR) - (3678 * (int)*prG) - ( 714 * (int)*prB) + UVadd) +
										((4392 * (int)*prRb) - (3678 * (int)*prGb) - ( 714 * (int)*prBb) + UVadd)) / 20000, img->bit_depth);
					*imgU++ = DepthConvert<BGRA_PIX, IMG_PIX>( ((-(1482 * (int)*prR) - (2910 * (int)*prG) + (4392 * (int)*prB) + UVadd) +
										(-(1482 * (int)*prRb) - (2910 * (int)*prGb) + (4392 * (int)*prBb) + UVadd)) / 20000, img->bit_depth);
				}
				
				prRb += 4;
				prGb += 4;
				prBb += 4;
			}
			else
			{
				if(x % sub_x == 0)
				{
					*imgV++ = DepthConvert<BGRA_PIX, IMG_PIX>( (((4392 * (int)*prR) - (3678 * (int)*prG) - ( 714 * (int)*prB) + UVadd)) / 10000, img->bit_depth);
					*imgU++ = DepthConvert<BGRA_PIX, IMG_PIX>( ((-(1482 * (int)*prR) - (2910 * (int)*prG) + (4392 * (int)*prB) + UVadd) ) / 10000, img->bit_depth);
				}
			}
			
			prR += 4;
			prG += 4;
			prB += 4;
		}
	}
	
	
	if(alpha_img != NULL)
	{
		assert(sub_x == alpha_img->x_chroma_shift + 1);
		assert(sub_y == alpha_img->y_chroma_shift + 1);
		
		for(int y = 0; y < img->d_h; y++)
		{
			IMG_PIX *imgY = (IMG_PIX *)(alpha_img->planes[VPX_PLANE_Y] + (alpha_img->stride[VPX_PLANE_Y] * y));
			IMG_PIX *imgU = (IMG_PIX *)(alpha_img->planes[VPX_PLANE_U] + (alpha_img->stride[VPX_PLANE_U] * (y / sub_y)));
			IMG_PIX *imgV = (IMG_PIX *)(alpha_img->planes[VPX_PLANE_V] + (alpha_img->stride[VPX_PLANE_V] * (y / sub_y)));
			
			const BGRA_PIX *prBGRA = (BGRA_PIX *)(frameBufferP + (rowbytes * (alpha_img->d_h - 1 - y)));
			
			const BGRA_PIX *prA = prBGRA + 3;
			
			if(isARGB)
			{
				prA = prBGRA + 0;
			}
			
			for(int x=0; x < img->d_w; x++)
			{
				*imgY++ = DepthConvert<BGRA_PIX, IMG_PIX>(*prA, img->bit_depth);
				
				if( (y % sub_y == 0) && (x % sub_x == 0) )
				{
					*imgV++ = DepthConvert<unsigned char, IMG_PIX>(128, img->bit_depth);
					*imgU++ = DepthConvert<unsigned char, IMG_PIX>(128, img->bit_depth);
				}
				
				prA += 4;
			}
		}
	}
}


static void
CopyPixToVPXImg(vpx_image_t *img, vpx_image_t *alpha_img, const PPixHand &outFrame, PrSDKPPixSuite *pixSuite, PrSDKPPix2Suite *pix2Suite)
{
	prRect boundsRect;
	pixSuite->GetBounds(outFrame, &boundsRect);
	
	assert(boundsRect.right == img->d_w && boundsRect.bottom == img->d_h);

	PrPixelFormat pixFormat;
	pixSuite->GetPixelFormat(outFrame, &pixFormat);

	const unsigned int sub_x = img->x_chroma_shift + 1;
	const unsigned int sub_y = img->y_chroma_shift + 1;

	if(pixFormat == PrPixelFormat_YUV_420_MPEG2_FRAME_PICTURE_PLANAR_8u_601)
	{
		assert(sub_x == 2 && sub_y == 2);
		assert(img->bit_depth == 8);
		assert(alpha_img == NULL);
		
		char *Y_PixelAddress, *U_PixelAddress, *V_PixelAddress;
		csSDK_uint32 Y_RowBytes, U_RowBytes, V_RowBytes;
		
		pix2Suite->GetYUV420PlanarBuffers(outFrame, PrPPixBufferAccess_ReadOnly,
											&Y_PixelAddress, &Y_RowBytes,
											&U_PixelAddress, &U_RowBytes,
											&V_PixelAddress, &V_RowBytes);
		
		for(int y = 0; y < img->d_h; y++)
		{
			unsigned char *imgY = img->planes[VPX_PLANE_Y] + (img->stride[VPX_PLANE_Y] * y);
			
			const unsigned char *prY = (unsigned char *)Y_PixelAddress + (Y_RowBytes * y);
			
			memcpy(imgY, prY, img->d_w * sizeof(unsigned char));
		}
		
		const int chroma_width = (img->d_w / 2) + (img->d_w % 2);
		const int chroma_height = (img->d_h / 2) + (img->d_h % 2);
		
		for(int y = 0; y < chroma_height; y++)
		{
			unsigned char *imgU = img->planes[VPX_PLANE_U] + (img->stride[VPX_PLANE_U] * y);
			unsigned char *imgV = img->planes[VPX_PLANE_V] + (img->stride[VPX_PLANE_V] * y);
			
			const unsigned char *prU = (unsigned char *)U_PixelAddress + (U_RowBytes * y);
			const unsigned char *prV = (unsigned char *)V_PixelAddress + (V_RowBytes * y);
			
			memcpy(imgU, prU, chroma_width * sizeof(unsigned char));
			memcpy(imgV, prV, chroma_width * sizeof(unsigned char));
		}
	}
	else
	{
		char *frameBufferP = NULL;
		csSDK_int32 rowbytes = 0;
		
		pixSuite->GetPixels(outFrame, PrPPixBufferAccess_ReadOnly, &frameBufferP);
		pixSuite->GetRowBytes(outFrame, &rowbytes);
		
		
		if(pixFormat == PrPixelFormat_UYVY_422_8u_601)
		{
			assert(sub_x == 2 && sub_y == 1);
			assert(img->bit_depth == 8);
			assert(alpha_img == NULL);
			
			for(int y = 0; y < img->d_h; y++)
			{
				unsigned char *imgY = img->planes[VPX_PLANE_Y] + (img->stride[VPX_PLANE_Y] * y);
				unsigned char *imgU = img->planes[VPX_PLANE_U] + (img->stride[VPX_PLANE_U] * y);
				unsigned char *imgV = img->planes[VPX_PLANE_V] + (img->stride[VPX_PLANE_V] * y);
			
				const unsigned char *prUYVY = (unsigned char *)frameBufferP + (rowbytes * y);
				
				for(int x=0; x < img->d_w; x++)
				{
					if(x % 2 == 0)
						*imgU++ = *prUYVY++;
					else
						*imgV++ = *prUYVY++;
					
					*imgY++ = *prUYVY++;;
				}
			}
		}
		else if(pixFormat == PrPixelFormat_VUYX_4444_8u)
		{
			assert(sub_x == 1 && sub_y == 1);
			assert(img->bit_depth == 8);
			assert(alpha_img == NULL);
			
			CopyVUYAToVPXImg<unsigned char, unsigned char>(img, alpha_img, frameBufferP, rowbytes);
		}
		else if(pixFormat == PrPixelFormat_VUYA_4444_16u)
		{
			assert(img->bit_depth > 8);
			
			CopyVUYAToVPXImg<unsigned short, unsigned short>(img, alpha_img, frameBufferP, rowbytes);
		}
		else if(pixFormat == PrPixelFormat_BGRA_4444_16u || pixFormat == PrPixelFormat_BGRX_4444_16u)
		{
			assert(pixFormat == PrPixelFormat_BGRA_4444_16u || alpha_img == NULL);
		
			if(img->bit_depth > 8)
				CopyBGRAToVPXImg<unsigned short, unsigned short, false>(img, alpha_img, frameBufferP, rowbytes);
			else
				CopyBGRAToVPXImg<unsigned short, unsigned char, false>(img, alpha_img, frameBufferP, rowbytes);
		}
		else if(pixFormat == PrPixelFormat_BGRA_4444_8u || pixFormat == PrPixelFormat_BGRX_4444_8u)
		{
			assert(pixFormat == PrPixelFormat_BGRA_4444_8u || alpha_img == NULL);
		
			if(img->bit_depth > 8)
				CopyBGRAToVPXImg<unsigned char, unsigned short, false>(img, alpha_img, frameBufferP, rowbytes);
			else
				CopyBGRAToVPXImg<unsigned char, unsigned char, false>(img, alpha_img, frameBufferP, rowbytes);
		}
		else if(pixFormat == PrPixelFormat_ARGB_4444_8u || pixFormat == PrPixelFormat_XRGB_4444_8u)
		{
			assert(pixFormat == PrPixelFormat_ARGB_4444_8u || alpha_img == NULL);
		
			if(img->bit_depth > 8)
				CopyBGRAToVPXImg<unsigned char, unsigned short, true>(img, alpha_img, frameBufferP, rowbytes);
			else
				CopyBGRAToVPXImg<unsigned char, unsigned char, true>(img, alpha_img, frameBufferP, rowbytes);
		}
		else
			assert(false);
	}
}


static void
CopyPixToAOMImg(aom_image_t *img, aom_image_t *alpha_img, const PPixHand &outFrame, PrSDKPPixSuite *pixSuite, PrSDKPPix2Suite *pix2Suite)
{
	vpx_image_t vpx_img, vpx_alpha_img;
	
	aom_to_vpx_img(&vpx_img, img);
	
	if(alpha_img != NULL)
		aom_to_vpx_img(&vpx_alpha_img, alpha_img);
	
	CopyPixToVPXImg(&vpx_img, (alpha_img == NULL ? NULL : &vpx_alpha_img), outFrame, pixSuite, pix2Suite);
}


#ifdef WEBM_HAVE_NVENC
static void
NVENCBufToVPXImg(vpx_image_t &vpx_img, void *bufferDataPtr, uint32_t pitch, NV_ENC_BUFFER_FORMAT format, uint32_t width, uint32_t height)
{
	assert(format != NV_ENC_BUFFER_FORMAT_NV12);

	vpx_img.fmt = (format == NV_ENC_BUFFER_FORMAT_YV12 ? VPX_IMG_FMT_YV12 :
		format == NV_ENC_BUFFER_FORMAT_IYUV ? VPX_IMG_FMT_I420 :
		format == NV_ENC_BUFFER_FORMAT_YUV444 ? VPX_IMG_FMT_I444 :
		format == NV_ENC_BUFFER_FORMAT_YUV420_10BIT ? VPX_IMG_FMT_I42016 :
		format == NV_ENC_BUFFER_FORMAT_YUV444_10BIT ? VPX_IMG_FMT_I44416 :
		VPX_IMG_FMT_NONE);

	assert(vpx_img.fmt != VPX_IMG_FMT_NONE && vpx_img.fmt != VPX_IMG_FMT_YV12);

	vpx_img.d_w = vpx_img.w = width;
	vpx_img.d_h = vpx_img.h = height;

	vpx_img.bit_depth = (format == NV_ENC_BUFFER_FORMAT_YUV420_10BIT || format == NV_ENC_BUFFER_FORMAT_YUV444_10BIT) ? 16 : 8;

	const bool subsampled = !(format == NV_ENC_BUFFER_FORMAT_YUV444 || format == NV_ENC_BUFFER_FORMAT_YUV444_10BIT);

	vpx_img.x_chroma_shift = (subsampled ? 1 : 0);
	vpx_img.y_chroma_shift = (subsampled ? 1 : 0);

	vpx_img.stride[VPX_PLANE_Y] = pitch; // pitch == rowbytes
	vpx_img.stride[VPX_PLANE_U] = (pitch / (subsampled ? 2 : 1));
	vpx_img.stride[VPX_PLANE_V] = vpx_img.stride[VPX_PLANE_U];

	unsigned char *planarYUV = (unsigned char*)bufferDataPtr;

	if(format == NV_ENC_BUFFER_FORMAT_YUV420_10BIT)
	{
		// semi-planar?!?
		planarYUV = (unsigned char*)malloc((vpx_img.stride[VPX_PLANE_Y] * height) + (vpx_img.stride[VPX_PLANE_U] * height / 2) + (vpx_img.stride[VPX_PLANE_V] * height / 2));

		if(planarYUV == NULL)
			return;
	}

	vpx_img.planes[VPX_PLANE_Y] = (unsigned char*)planarYUV;
	vpx_img.planes[VPX_PLANE_U] = vpx_img.planes[VPX_PLANE_Y] + (vpx_img.stride[VPX_PLANE_Y] * height);
	vpx_img.planes[VPX_PLANE_V] = vpx_img.planes[VPX_PLANE_U] + (vpx_img.stride[VPX_PLANE_U] * height / (subsampled ? 2 : 1));
}

static void
CopyToYUV420_10BIT(vpx_image_t &vpx_img, void *bufferDataPtr, uint32_t pitch)
{
	unsigned char *semiPlanarYUV = (unsigned char *)bufferDataPtr;

	assert(vpx_img.bit_depth == 16);
	assert(vpx_img.stride[VPX_PLANE_Y] == (sizeof(unsigned char) * pitch));

	memcpy(semiPlanarYUV, vpx_img.planes[VPX_PLANE_Y], vpx_img.stride[VPX_PLANE_Y] * vpx_img.d_h);

	for(int y=0; y < (vpx_img.d_h / 2); y++)
	{
		unsigned short *pixUV = (unsigned short *)(semiPlanarYUV + (vpx_img.stride[VPX_PLANE_Y] * vpx_img.d_h) + (sizeof(unsigned char) * pitch * y));
		const unsigned short *pixU = (const unsigned short *)(vpx_img.planes[VPX_PLANE_U] + (vpx_img.stride[VPX_PLANE_U] * y));
		const unsigned short *pixV = (const unsigned short *)(vpx_img.planes[VPX_PLANE_V] + (vpx_img.stride[VPX_PLANE_V] * y));

		for(int x=0; x < (vpx_img.d_w / 2); x++)
		{
			*pixUV++ = *pixU++;
			*pixUV++ = *pixV++;
		}
	}
}

static void
CopyPixToNVENCBuf(void *bufferDataPtr, uint32_t pitch, NV_ENC_BUFFER_FORMAT format,
					void *alphaBufferDataPtr, uint32_t alphaPitch, NV_ENC_BUFFER_FORMAT alphaFormat,
					uint32_t width, uint32_t height,
					const PPixHand &outFrame, PrSDKPPixSuite *pixSuite, PrSDKPPix2Suite *pix2Suite)
{
	vpx_image_t vpx_img, alpha_vpx_img;

	NVENCBufToVPXImg(vpx_img, bufferDataPtr, pitch, format, width, height);

	if(alphaBufferDataPtr != NULL)
		NVENCBufToVPXImg(alpha_vpx_img, alphaBufferDataPtr, alphaPitch, alphaFormat, width, height);


	CopyPixToVPXImg(&vpx_img, (alphaBufferDataPtr != NULL ? &alpha_vpx_img : NULL), outFrame, pixSuite, pix2Suite);


	if(format == NV_ENC_BUFFER_FORMAT_YUV420_10BIT)
	{
		CopyToYUV420_10BIT(vpx_img, bufferDataPtr, pitch);

		free(vpx_img.planes[VPX_PLANE_Y]);
	}

	if(alphaBufferDataPtr != NULL && alphaFormat == NV_ENC_BUFFER_FORMAT_YUV420_10BIT)
	{
		CopyToYUV420_10BIT(alpha_vpx_img, alphaBufferDataPtr, alphaPitch);

		free(alpha_vpx_img.planes[VPX_PLANE_Y]);
	}
}
#endif // WEBM_HAVE_NVENC

static void
vorbis_get_limits(int audioChannels, float sampleRate, long &min_bitrate, long &max_bitrate)
{
	// must conform to bitrate profiles, see vorbisenc.c

	if(audioChannels == 6)
	{
		if(sampleRate < 8000.)
		{
			// ve_setup_XX_uncoupled
		
			assert(false);
			
			min_bitrate = -1;
			max_bitrate = -1;
		}
		else if(sampleRate < 9000.)
		{
			// ve_setup_8_uncoupled
			
			min_bitrate = 8000;
			max_bitrate = 42000;
		}
		else if(sampleRate < 15000.)
		{
			// ve_setup_11_uncoupled
			
			min_bitrate = 12000;
			max_bitrate = 50000;
		}
		else if(sampleRate < 19000.)
		{
			// ve_setup_16_uncoupled
			
			min_bitrate = 16000;
			max_bitrate = 100000;
		}
		else if(sampleRate < 26000.)
		{
			// ve_setup_22_uncoupled
			
			min_bitrate = 16000;
			max_bitrate = 90000;
		}
		else if(sampleRate < 40000.)
		{
			// ve_setup_32_uncoupled
			
			min_bitrate = 30000;
			max_bitrate = 190000;
		}
		else if(sampleRate < 50000.)
		{
			// ve_setup_44_51
			
			min_bitrate = 14000;
			max_bitrate = 240001;
		}
		else if(sampleRate < 200000.)
		{
			// ve_setup_X_uncoupled
			
			min_bitrate = -1;
			max_bitrate = -1;
		}
		else
		{
			assert(false);
			
			min_bitrate = -1;
			max_bitrate = -1;
		}
	}
	else if(audioChannels == 2)
	{
		if(sampleRate < 8000.)
		{
			// ve_setup_XX_stereo
		
			assert(false);
			
			min_bitrate = -1;
			max_bitrate = -1;
		}
		else if(sampleRate < 9000.)
		{
			// ve_setup_8_stereo
			
			min_bitrate = 6000;
			max_bitrate = 32000;
		}
		else if(sampleRate < 15000.)
		{
			// ve_setup_11_stereo
			
			min_bitrate = 8000;
			max_bitrate = 44000;
		}
		else if(sampleRate < 19000.)
		{
			// ve_setup_16_stereo
			
			min_bitrate = 12000;
			max_bitrate = 86000;
		}
		else if(sampleRate < 26000.)
		{
			// ve_setup_22_stereo
			
			min_bitrate = 15000;
			max_bitrate = 86000;
		}
		else if(sampleRate < 40000.)
		{
			// ve_setup_32_stereo
			
			min_bitrate = 18000;
			max_bitrate = 190000;
		}
		else if(sampleRate < 50000.)
		{
			// ve_setup_44_stereo
			
			min_bitrate = 22500;
			max_bitrate = 250001;
		}
		else if(sampleRate < 200000.)
		{
			// ve_setup_X_stereo
			
			min_bitrate = -1;
			max_bitrate = -1;
		}
		else
		{
			assert(false);
			
			min_bitrate = -1;
			max_bitrate = -1;
		}
	}
	else
	{
		assert(audioChannels == 1);
		
		if(sampleRate < 8000.)
		{
			// ve_setup_XX_uncoupled
		
			assert(false);
			
			min_bitrate = -1;
			max_bitrate = -1;
		}
		else if(sampleRate < 9000.)
		{
			// ve_setup_8_uncoupled
			
			min_bitrate = 8000;
			max_bitrate = 42000;
		}
		else if(sampleRate < 15000.)
		{
			// ve_setup_11_uncoupled
			
			min_bitrate = 12000;
			max_bitrate = 50000;
		}
		else if(sampleRate < 19000.)
		{
			// ve_setup_16_uncoupled
			
			min_bitrate = 16000;
			max_bitrate = 100000;
		}
		else if(sampleRate < 26000.)
		{
			// ve_setup_22_uncoupled
			
			min_bitrate = 16000;
			max_bitrate = 90000;
		}
		else if(sampleRate < 40000.)
		{
			// ve_setup_32_uncoupled
			
			min_bitrate = 30000;
			max_bitrate = 190000;
		}
		else if(sampleRate < 50000.)
		{
			// ve_setup_44_uncoupled
			
			min_bitrate = 32000;
			max_bitrate = 240001;
		}
		else if(sampleRate < 200000.)
		{
			// ve_setup_X_uncoupled
			
			min_bitrate = -1;
			max_bitrate = -1;
		}
		else
		{
			assert(false);
			
			min_bitrate = -1;
			max_bitrate = -1;
		}
	}
}

static int
xiph_len(int l)
{
    return 1 + l / 255 + l;
}

static void
xiph_lace(unsigned char **np, uint64_t val)
{
	unsigned char *p = *np;

	while(val >= 255)
	{
		*p++ = 255;
		val -= 255;
	}
	
	*p++ = val;
	
	*np = p;
}

static void *
MakePrivateData(ogg_packet &header, ogg_packet &header_comm, ogg_packet &header_code, size_t &size)
{
	size = 1 + xiph_len(header.bytes) + xiph_len(header_comm.bytes) + header_code.bytes;
	
	void *buf = malloc(size);
	
	if(buf)
	{
		unsigned char *p = (unsigned char *)buf;
		
		*p++ = 2;
		
		xiph_lace(&p, header.bytes);
		xiph_lace(&p, header_comm.bytes);
		
		memcpy(p, header.packet, header.bytes);
		p += header.bytes;
		memcpy(p, header_comm.packet, header_comm.bytes);
		p += header_comm.bytes;
		memcpy(p, header_code.packet, header_code.bytes);
	}
	
	return buf;
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
	
	exParamValues videoCodecP, av1codecP, methodP, videoQualityP, bitrateP, twoPassP, keyframeMaxDistanceP, samplingP, bitDepthP, alphaP, customArgsP;
		
	paramSuite->GetParamValue(exID, gIdx, WebMVideoCodec, &videoCodecP);
	paramSuite->GetParamValue(exID, gIdx, WebMAV1Codec, &av1codecP);
	paramSuite->GetParamValue(exID, gIdx, WebMVideoMethod, &methodP);
	paramSuite->GetParamValue(exID, gIdx, WebMVideoQuality, &videoQualityP);
	paramSuite->GetParamValue(exID, gIdx, WebMVideoBitrate, &bitrateP);
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
										chroma == WEBM_444 ? PrPixelFormat_VUYX_4444_8u :
										chroma == WEBM_422 ? PrPixelFormat_UYVY_422_8u_601 :
										PrPixelFormat_YUV_420_MPEG2_FRAME_PICTURE_PLANAR_8u_601);

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
		PrAudioChannelLabel monoOrder[1] = { kPrAudioChannelLabel_Discrete };
												
		PrAudioChannelLabel stereoOrder[2] = { kPrAudioChannelLabel_FrontLeft,
												kPrAudioChannelLabel_FrontRight };
													
		// Premiere uses Left, Right, Left Rear, Right Rear, Center, LFE
		// Opus and Vorbis use Left, Center, Right, Left Rear, Right Rear, LFE
		// http://www.xiph.org/vorbis/doc/Vorbis_I_spec.html#x1-800004.3.9
		PrAudioChannelLabel surroundOrder[6] = { kPrAudioChannelLabel_FrontLeft,
													kPrAudioChannelLabel_FrontCenter,
													kPrAudioChannelLabel_FrontRight,
													kPrAudioChannelLabel_RearSurroundLeft,
													kPrAudioChannelLabel_RearSurroundRight,
													kPrAudioChannelLabel_LowFrequency };
		
		PrAudioChannelLabel *channelOrder = (audioFormat == kPrAudioChannelType_51 ? surroundOrder :
												audioFormat == kPrAudioChannelType_Stereo ? stereoOrder :
												audioFormat == kPrAudioChannelType_Mono ? monoOrder :
												stereoOrder);
		
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

	bool multipass = (exportInfoP->exportVideo && twoPassP.value.intValue);

	std::string codecMessage;

#ifdef WEBM_HAVE_NVENC
	CUcontext cudaContext = NULL;

	NVENCSTATUS nv_err = NV_ENC_SUCCESS;

	void* nv_encoder = NULL;
	NV_ENC_BUFFER_FORMAT nv_input_format = NV_ENC_BUFFER_FORMAT_UNDEFINED;
	int nv_input_buffer_idx = 0;
	std::vector<NV_ENC_INPUT_PTR> nv_input_buffers;
	bool nv_output_available = false;
	int nv_output_buffer_idx = 0;
	std::vector<NV_ENC_OUTPUT_PTR> nv_output_buffers;
	std::queue<NV_ENC_LOCK_BITSTREAM> nv_encoder_queue;

	void* nv_alpha_encoder = NULL;
	NV_ENC_BUFFER_FORMAT nv_alpha_input_format = NV_ENC_BUFFER_FORMAT_UNDEFINED;
	int nv_alpha_input_buffer_idx = 0;
	std::vector<NV_ENC_INPUT_PTR> nv_alpha_input_buffers;
	bool nv_alpha_output_available = false;
	int nv_alpha_output_buffer_idx = 0;
	std::vector<NV_ENC_OUTPUT_PTR> nv_alpha_output_buffers;
	std::queue<NV_ENC_LOCK_BITSTREAM> nv_alpha_encoder_queue;

#endif // WEBM_HAVE_NVENC

	if(exportInfoP->exportVideo && video_codec == WEBM_CODEC_AV1 && av1_codec != AV1_CODEC_AOM)
	{
		// Checking NVENC stuff here because it doesn't actually appear to run through multiple passes
		// and want to make sure I'm actually going to use it before I cancel them.

		if(av1_auto)
		{
		#ifdef WEBM_HAVE_NVENC
			if(nvenc.version != 0)
				av1_codec = AV1_CODEC_NVENC;
			else
		#endif // WEBM_HAVE_NVENC
				av1_codec = AV1_CODEC_AOM;
		}

		const AV1_Codec fallback_codec = AV1_CODEC_AOM;

		if(av1_codec == AV1_CODEC_NVENC && (chroma != WEBM_420 || bit_depth > 10))
		{
			// 4:4:4 not supported yet
			codecMessage = "Incompatible NVENC pixel settings";

			if(av1_auto)
				av1_codec = fallback_codec;
			else
				result = exportReturn_InternalError;
		}

		if(av1_codec == AV1_CODEC_NVENC && result == malNoError)
		{
		#ifdef WEBM_HAVE_NVENC
			if(nvenc.version != 0)
			{
				CUresult cuErr = cuCtxCreate(&cudaContext, CU_CTX_SCHED_AUTO, cudaDevice);

				if(cuErr == CUDA_SUCCESS)
				{
					assert(cudaContext != NULL);

					NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS sessionParams = { 0 };

					sessionParams.version = NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS_VER;
					sessionParams.deviceType = NV_ENC_DEVICE_TYPE_CUDA;
					sessionParams.device = cudaContext;
					sessionParams.apiVersion = NVENCAPI_VERSION;

					nv_err = nvenc.nvEncOpenEncodeSessionEx(&sessionParams, &nv_encoder);

					if(use_alpha && nv_err == NV_ENC_SUCCESS)
						nv_err = nvenc.nvEncOpenEncodeSessionEx(&sessionParams, &nv_alpha_encoder);

					if(nv_err == NV_ENC_SUCCESS)
					{
						const GUID codecGUID = NV_ENC_CODEC_AV1_GUID;

						bool have_codec = false;

						uint32_t codec_count = 0;
						nv_err = nvenc.nvEncGetEncodeGUIDCount(nv_encoder, &codec_count);

						if(nv_err == NV_ENC_SUCCESS && codec_count > 0)
						{
							GUID *guids = new GUID[codec_count];

							uint32_t codec_count_again = 0;

							nv_err = nvenc.nvEncGetEncodeGUIDs(nv_encoder, guids, codec_count, &codec_count_again);

							assert(codec_count_again == codec_count);

							for(int i=0; i < codec_count && nv_err == NV_ENC_SUCCESS && !have_codec; i++)
							{
								if(guids[i] == codecGUID)
									have_codec = true;
							}

							delete[] guids;
						}


						const GUID profileGUID = NV_ENC_AV1_PROFILE_MAIN_GUID;

						bool have_profile = false;

						if(have_codec)
						{
							uint32_t profile_count = 0;

							nv_err = nvenc.nvEncGetEncodeProfileGUIDCount(nv_encoder, codecGUID, &profile_count);

							if(nv_err == NV_ENC_SUCCESS && profile_count > 0)
							{
								GUID *guids = new GUID[profile_count];

								uint32_t profile_count_again = 0;

								nv_err = nvenc.nvEncGetEncodeProfileGUIDs(nv_encoder, codecGUID, guids, profile_count, &profile_count_again);

								assert(profile_count_again == profile_count);

								for(int i=0; i < profile_count && nv_err == NV_ENC_SUCCESS && !have_profile; i++)
								{
									if(guids[i] == profileGUID)
										have_profile = true;
								}

								delete[] guids;
							}
						}


						nv_input_format = (chroma == WEBM_444) ? (bit_depth == 10 ? NV_ENC_BUFFER_FORMAT_YUV444_10BIT : NV_ENC_BUFFER_FORMAT_YUV444) :
																	(bit_depth == 10 ? NV_ENC_BUFFER_FORMAT_YUV420_10BIT : NV_ENC_BUFFER_FORMAT_IYUV);

						bool have_input_format = false;

						if(have_codec && have_profile)
						{
							uint32_t format_count = 0;

							nv_err = nvenc.nvEncGetInputFormatCount(nv_encoder, codecGUID, &format_count);

							if(nv_err == NV_ENC_SUCCESS && format_count > 0)
							{
								bool have_nv12 = false;
								bool have_yv12 = false;
								bool have_iyuv = false;
								bool have_yuv444 = false;
								bool have_yuv420_10bit = false;
								bool have_yuv444_10bit = false;
								bool have_argb = false;
								bool have_argb10 = false;
								bool have_ayuv = false;
								bool have_abgr = false;
								bool have_abgr10 = false;
								bool have_u8 = false;

								NV_ENC_BUFFER_FORMAT* formats = new NV_ENC_BUFFER_FORMAT[format_count];

								uint32_t format_count_again = 0;

								nv_err = nvenc.nvEncGetInputFormats(nv_encoder, codecGUID, formats, format_count, &format_count_again);

								assert(format_count_again == format_count);

								for(int i=0; i < format_count && nv_err == NV_ENC_SUCCESS; i++)
								{
									const NV_ENC_BUFFER_FORMAT& format = formats[i];

									if(format == NV_ENC_BUFFER_FORMAT_NV12)
										have_nv12 = true;
									else if(format == NV_ENC_BUFFER_FORMAT_YV12)
										have_yv12 = true;
									else if(format == NV_ENC_BUFFER_FORMAT_IYUV)
										have_iyuv = true;
									else if(format == NV_ENC_BUFFER_FORMAT_YUV444)
										have_yuv444 = true;
									else if(format == NV_ENC_BUFFER_FORMAT_YUV420_10BIT)
										have_yuv420_10bit = true;
									else if(format == NV_ENC_BUFFER_FORMAT_YUV444_10BIT)
										have_yuv444_10bit = true;
									else if(format == NV_ENC_BUFFER_FORMAT_ARGB)
										have_argb = true;
									else if(format == NV_ENC_BUFFER_FORMAT_ARGB10)
										have_argb10 = true;
									else if(format == NV_ENC_BUFFER_FORMAT_AYUV)
										have_ayuv = true;
									else if(format == NV_ENC_BUFFER_FORMAT_ABGR)
										have_abgr = true;
									else if(format == NV_ENC_BUFFER_FORMAT_ABGR10)
										have_abgr10 = true;
									else if(format == NV_ENC_BUFFER_FORMAT_U8)
										have_u8 = true;
								}

								delete[] formats;

								if(nv_input_format == NV_ENC_BUFFER_FORMAT_IYUV)
									have_input_format = have_iyuv;
								else if(nv_input_format == NV_ENC_BUFFER_FORMAT_YUV444)
									have_input_format = have_yuv444;
								else if(nv_input_format == NV_ENC_BUFFER_FORMAT_YUV420_10BIT)
									have_input_format = have_yuv420_10bit;
								else if(nv_input_format == NV_ENC_BUFFER_FORMAT_YUV444_10BIT)
									have_input_format = have_yuv444_10bit;
							}
						}


						const GUID presetGUID = NV_ENC_PRESET_P7_GUID;

						bool have_preset = false;

						if(have_codec && have_profile && have_input_format)
						{
							uint32_t preset_count = 0;

							nv_err = nvenc.nvEncGetEncodePresetCount(nv_encoder, codecGUID, &preset_count);

							if (nv_err == NV_ENC_SUCCESS && preset_count > 0)
							{
								GUID *guids = new GUID[preset_count];

								uint32_t preset_count_again = 0;

								nv_err = nvenc.nvEncGetEncodeProfileGUIDs(nv_encoder, codecGUID, guids, preset_count, &preset_count_again);

								//assert(preset_count_again == preset_count); // ??

								for(int i=0; i < preset_count_again && nv_err == NV_ENC_SUCCESS && !have_preset; i++)
								{
									if(guids[i] == presetGUID)
										have_preset = true;
								}

								delete[] guids;
							}
						}

						assert(!have_preset); // not sure what's going on here, using NV_ENC_PRESET_P7_GUID anyway


						bool have_capabilities = true;

						if(bit_depth == 10)
						{
							int can10bit = 0;

							NV_ENC_CAPS_PARAM caps = { 0 };
							caps.version = NV_ENC_CAPS_PARAM_VER;
							caps.capsToQuery = NV_ENC_CAPS_SUPPORT_10BIT_ENCODE;

							nvenc.nvEncGetEncodeCaps(nv_encoder, codecGUID, &caps, &can10bit);

							if(!can10bit)
								have_capabilities = false;
						}

						if(chroma == WEBM_444)
						{
							int can4444 = 0;

							NV_ENC_CAPS_PARAM caps = { 0 };
							caps.version = NV_ENC_CAPS_PARAM_VER;
							caps.capsToQuery = NV_ENC_CAPS_SUPPORT_YUV444_ENCODE;

							nvenc.nvEncGetEncodeCaps(nv_encoder, codecGUID, &caps, &can4444);

							if (!can4444)
								have_capabilities = false;
						}
						else if(chroma == WEBM_422)
						{
							have_capabilities = false;
						}


						if(nv_err == NV_ENC_SUCCESS && have_codec && have_profile && have_input_format && have_capabilities)
						{
							const NV_ENC_TUNING_INFO tuningInfo = NV_ENC_TUNING_INFO_HIGH_QUALITY;

							NV_ENC_PRESET_CONFIG presetConfig = { 0 };
							presetConfig.version = NV_ENC_PRESET_CONFIG_VER;
							presetConfig.presetCfg.version = NV_ENC_CONFIG_VER;

							nv_err = nvenc.nvEncGetEncodePresetConfigEx(nv_encoder, codecGUID, presetGUID, tuningInfo, &presetConfig);

							if(nv_err == NV_ENC_SUCCESS)
							{
								NV_ENC_CONFIG &config = presetConfig.presetCfg;

								NV_ENC_RC_PARAMS &rcParams = config.rcParams;

								if(method == WEBM_METHOD_CONSTANT_QUALITY || method == WEBM_METHOD_CONSTRAINED_QUALITY)
								{
									rcParams.rateControlMode = NV_ENC_PARAMS_RC_CONSTQP;

									rcParams.constQP.qpIntra = rcParams.constQP.qpInterP = rcParams.constQP.qpInterB = (100 - videoQualityP.value.intValue);
								}
								else
								{
									if(method == WEBM_METHOD_VBR)
									{
										rcParams.rateControlMode = NV_ENC_PARAMS_RC_VBR;
									}
									else if(method == WEBM_METHOD_BITRATE)
									{
										rcParams.rateControlMode = NV_ENC_PARAMS_RC_CBR;
									}
									else
										assert(false);

									rcParams.averageBitRate = bitrateP.value.intValue * 1000;
									rcParams.maxBitRate = rcParams.averageBitRate * 120 / 100;
								}

								assert(rcParams.multiPass == NV_ENC_MULTI_PASS_DISABLED);
								rcParams.multiPass = (twoPassP.value.intValue ? NV_ENC_TWO_PASS_FULL_RESOLUTION : NV_ENC_MULTI_PASS_DISABLED);

								NV_ENC_CONFIG_AV1 &av1config = config.encodeCodecConfig.av1Config;

								assert(av1config.chromaFormatIDC == 1); // 4:2:0, 4:4:4 currently not supported
								av1config.inputBitDepth = (bit_depth == 10 ? NV_ENC_BIT_DEPTH_10 : NV_ENC_BIT_DEPTH_8);
								av1config.outputBitDepth = av1config.inputBitDepth;

								ConfigureNVENCEncoder(config, customArgs);

								NV_ENC_INITIALIZE_PARAMS params = { 0 };

								params.version = NV_ENC_INITIALIZE_PARAMS_VER;
								params.encodeGUID = codecGUID;
								params.presetGUID = presetGUID;
								params.encodeWidth = renderParms.inWidth;
								params.encodeHeight = renderParms.inHeight;
								params.darWidth = renderParms.inWidth * renderParms.inPixelAspectRatioNumerator;
								params.darHeight = renderParms.inHeight * renderParms.inPixelAspectRatioDenominator;
								params.frameRateNum = fps.numerator;
								params.frameRateDen = fps.denominator;
								params.enableEncodeAsync = FALSE;
								params.enablePTD = TRUE;
								params.reportSliceOffsets = FALSE;
								params.enableSubFrameWrite = FALSE;
								params.enableExternalMEHints = FALSE;
								params.enableMEOnlyMode = FALSE;
								params.enableWeightedPrediction = FALSE;
								params.splitEncodeMode = FALSE;
								params.enableOutputInVidmem = FALSE;
								params.enableReconFrameOutput = FALSE;
								params.enableOutputStats = FALSE;
								params.enableUniDirectionalB = FALSE;
								params.privDataSize = 0;
								params.reserved = 0;
								params.privData = NULL;
								params.encodeConfig = &config;
								params.maxEncodeWidth = 0;
								params.maxEncodeHeight = 0;
								params.tuningInfo = tuningInfo;
								params.bufferFormat = NV_ENC_BUFFER_FORMAT_UNDEFINED; // only for DX12
								params.outputStatsLevel = NV_ENC_OUTPUT_STATS_NONE;


								NV_ENC_CONFIG alpha_config;
								NV_ENC_INITIALIZE_PARAMS alpha_params;

								if(use_alpha)
								{
									alpha_config = config;

									alpha_config.monoChromeEncoding = TRUE;

									if(method == WEBM_METHOD_BITRATE || method == WEBM_METHOD_VBR)
									{
										config.rcParams.averageBitRate = (config.rcParams.averageBitRate * 3 / 4);
										config.rcParams.maxBitRate = (config.rcParams.maxBitRate * 3 / 4);

										alpha_config.rcParams.averageBitRate = (config.rcParams.averageBitRate / 3);
										alpha_config.rcParams.maxBitRate = (config.rcParams.maxBitRate / 3);
									}

									alpha_params = params;

									alpha_params.enablePTD = params.enablePTD = FALSE;

									nv_alpha_input_format = nv_input_format;
								}


								nv_err = nvenc.nvEncInitializeEncoder(nv_encoder, &params);

								const int num_buffers = std::min(64, keyframeMaxDistanceP.value.intValue * 4);

								if(nv_err == NV_ENC_SUCCESS)
								{
									for(int i=0; i < num_buffers && nv_err == NV_ENC_SUCCESS; i++)
									{
										NV_ENC_CREATE_INPUT_BUFFER input_params = { 0 };

										input_params.version = NV_ENC_CREATE_INPUT_BUFFER_VER;
										input_params.width = params.encodeWidth;
										input_params.height = params.encodeHeight;
										input_params.bufferFmt = nv_input_format;
										input_params.inputBuffer = NULL;
										input_params.pSysMemBuffer = NULL;

										nv_err = nvenc.nvEncCreateInputBuffer(nv_encoder, &input_params);

										if(nv_err == NV_ENC_SUCCESS)
										{
											nv_input_buffers.push_back(input_params.inputBuffer);

											NV_ENC_CREATE_BITSTREAM_BUFFER output_params = { 0 };

											output_params.version = NV_ENC_CREATE_BITSTREAM_BUFFER_VER;
											output_params.reserved = 0;
											output_params.bitstreamBuffer = NULL;

											nv_err = nvenc.nvEncCreateBitstreamBuffer(nv_encoder, &output_params);

											if(nv_err == NV_ENC_SUCCESS)
											{
												nv_output_buffers.push_back(output_params.bitstreamBuffer);
											}
										}
									}
								}


								if(use_alpha && nv_err == NV_ENC_SUCCESS)
								{
									nv_err = nvenc.nvEncInitializeEncoder(nv_alpha_encoder, &alpha_params);

									if(nv_err == NV_ENC_SUCCESS)
									{
										for(int i=0; i < num_buffers && nv_err == NV_ENC_SUCCESS; i++)
										{
											NV_ENC_CREATE_INPUT_BUFFER input_params = { 0 };

											input_params.version = NV_ENC_CREATE_INPUT_BUFFER_VER;
											input_params.width = alpha_params.encodeWidth;
											input_params.height = alpha_params.encodeHeight;
											input_params.bufferFmt = nv_alpha_input_format;
											input_params.inputBuffer = NULL;
											input_params.pSysMemBuffer = NULL;

											nv_err = nvenc.nvEncCreateInputBuffer(nv_alpha_encoder, &input_params);

											if(nv_err == NV_ENC_SUCCESS)
											{
												nv_alpha_input_buffers.push_back(input_params.inputBuffer);

												NV_ENC_CREATE_BITSTREAM_BUFFER output_params = { 0 };

												output_params.version = NV_ENC_CREATE_BITSTREAM_BUFFER_VER;
												output_params.reserved = 0;
												output_params.bitstreamBuffer = NULL;

												nv_err = nvenc.nvEncCreateBitstreamBuffer(nv_alpha_encoder, &output_params);

												if(nv_err == NV_ENC_SUCCESS)
												{
													nv_alpha_output_buffers.push_back(output_params.bitstreamBuffer);
												}
											}
										}
									}
								}
							}
						}
						else
						{
							// Maybe you have an Nvidia card but not a 4090...

							nvenc.nvEncDestroyEncoder(nv_encoder);

							nv_encoder = NULL;

							if(use_alpha)
							{
								nvenc.nvEncDestroyEncoder(nv_alpha_encoder);

								nv_alpha_encoder = NULL;
							}

							CUresult cuErr = cuCtxDestroy(cudaContext);

							cudaContext = NULL;

							if(av1_auto)
								av1_codec = fallback_codec;
							else
								result = exportReturn_InternalError;
						}
					}

					if(nv_err == NV_ENC_SUCCESS)
					{
						multipass = false; // We set the multipass flag, but does it happen internally? Don't see how to actually run two passes.
					}
					else
					{
						codecMessage = "Failed to initialize NVENC encoder";

						if(cudaContext != NULL)
						{
							CUresult cuErr = cuCtxDestroy(cudaContext);

							cudaContext = NULL;
						}

						if(av1_auto)
							av1_codec = fallback_codec;
						else
							result = exportReturn_InternalError;
					}
				}
				else
				{
					codecMessage = "Failed to create CUDA context";

					if(av1_auto)
						av1_codec = fallback_codec;
					else
						result = exportReturn_InternalError;
				}
			}
			else
			{
				codecMessage = "NVENC codec not available";

				if(av1_auto)
					av1_codec = fallback_codec;
				else
					result = exportReturn_InternalError;
			}
		#else
			codecMessage = "NVENC codec not available";

			if(av1_auto)
				av1_codec = fallback_codec;
			else
				result = exportReturn_InternalError;
		#endif // WEBM_HAVE_NVENC
		}
	}


	const int passes = (multipass ? 2 : 1);
	
	for(int pass = 0; pass < passes && result == malNoError; pass++)
	{
		const bool vbr_pass = (passes > 1 && pass == 0);
		
		vpx_codec_err_t vpx_codec_err = VPX_CODEC_OK;
		
		vpx_codec_ctx_t vpx_encoder;
		vpx_codec_iter_t vpx_encoder_iter = NULL;
		std::queue<vpx_codec_cx_pkt_t> vpx_encoder_queue;
		
		vpx_codec_ctx_t vpx_alpha_encoder;
		vpx_codec_iter_t vpx_alpha_encoder_iter = NULL;
		std::queue<vpx_codec_cx_pkt_t> vpx_alpha_encoder_queue;
		
		
		aom_codec_err_t aom_codec_err = AOM_CODEC_OK;
		
		aom_codec_ctx_t aom_encoder;
		aom_codec_iter_t aom_encoder_iter = NULL;
		std::queue<aom_codec_cx_pkt_t> aom_encoder_queue;
		
		aom_codec_ctx_t aom_alpha_encoder;
		aom_codec_iter_t aom_alpha_encoder_iter = NULL;
		std::queue<aom_codec_cx_pkt_t> aom_alpha_encoder_queue;
		
		
		const uint64_t alpha_id = 1;
		
		const bool copy_buffers = use_alpha;
		
		unsigned long vpx_deadline = VPX_DL_GOOD_QUALITY;

												
		PrTime videoEncoderTime = exportInfoP->startTime;
		
		if(exportInfoP->exportVideo && result == malNoError)
		{
			if(video_codec == WEBM_CODEC_VP8 || video_codec == WEBM_CODEC_VP9)
			{
				vpx_codec_iface_t *iface = (video_codec == WEBM_CODEC_VP9) ? vpx_codec_vp9_cx() : vpx_codec_vp8_cx();
				
				vpx_codec_enc_cfg_t config;
				vpx_codec_enc_config_default(iface, &config, 0);
				
				config.g_w = renderParms.inWidth;
				config.g_h = renderParms.inHeight;
				
				// (only applies to VP9)
				// Profile 0 is 4:2:0 only
				// Profile 1 can do 4:4:4 and 4:2:2
				// Profile 2 can do 10- and 12-bit, 4:2:0 only
				// Profile 3 can do 10- and 12-bit, 4:4:4 and 4:2:2
				config.g_profile = (chroma > WEBM_420 ?
										(bit_depth > 8 ? 3 : 1) :
										(bit_depth > 8 ? 2 : 0) );
				
				config.g_bit_depth = (bit_depth == 12 ? VPX_BITS_12 :
										bit_depth == 10 ? VPX_BITS_10 :
										VPX_BITS_8);
				
				config.g_input_bit_depth = config.g_bit_depth;
				
				
				if(method == WEBM_METHOD_CONSTANT_QUALITY || method == WEBM_METHOD_CONSTRAINED_QUALITY)
				{
					config.rc_end_usage = (method == WEBM_METHOD_CONSTANT_QUALITY ? VPX_Q : VPX_CQ);
					config.g_pass = VPX_RC_ONE_PASS;
					
					const int min_q = config.rc_min_quantizer + 1;
					const int max_q = config.rc_max_quantizer;
					
					// our 0...100 slider will be used to bring max_q down to min_q
					config.rc_max_quantizer = min_q + ((((float)(100 - videoQualityP.value.intValue) / 100.f) * (max_q - min_q)) + 0.5f);
				}
				else
				{
					if(method == WEBM_METHOD_VBR)
					{
						config.rc_end_usage = VPX_VBR;
					}
					else if(method == WEBM_METHOD_BITRATE)
					{
						config.rc_end_usage = VPX_CBR;
						config.g_pass = VPX_RC_ONE_PASS;
					}
					else
						assert(false);
				}
				
				if(passes == 2)
				{
					if(vbr_pass)
					{
						config.g_pass = VPX_RC_FIRST_PASS;
					}
					else
					{
						config.g_pass = VPX_RC_LAST_PASS;
						
						config.rc_twopass_stats_in.buf = vbr_buffer;
						config.rc_twopass_stats_in.sz = vbr_buffer_size;
					}
				}
				else
					config.g_pass = VPX_RC_ONE_PASS;
					
				
				config.rc_target_bitrate = bitrateP.value.intValue;
				
				
				config.g_threads = g_num_cpus;
				
				config.g_timebase.num = fps.denominator;
				config.g_timebase.den = fps.numerator;
				
				config.kf_max_dist = keyframeMaxDistanceP.value.intValue;
				
				
				ConfigureVPXEncoderPre(config, vpx_deadline, customArgs);
				
				assert(config.kf_max_dist >= config.kf_min_dist);
				
				
				vpx_codec_enc_cfg_t alpha_config = config;
				
				if(use_alpha)
				{
					alpha_config.kf_min_dist = alpha_config.kf_max_dist = config.kf_min_dist = config.kf_max_dist;
					
					alpha_config.rc_target_bitrate = config.rc_target_bitrate / 3;
					
					config.rc_target_bitrate = config.rc_target_bitrate * 2 / 3;
					
					if(passes == 2 && !vbr_pass)
					{
						config.rc_twopass_stats_in.buf = alpha_vbr_buffer;
						config.rc_twopass_stats_in.sz = alpha_vbr_buffer_size;
					}
				}
				
				
				const vpx_codec_flags_t flags = (config.g_bit_depth == VPX_BITS_8 ? 0 : VPX_CODEC_USE_HIGHBITDEPTH);
				
				vpx_codec_err = vpx_codec_enc_init(&vpx_encoder, iface, &config, flags);
				
				if(use_alpha && vpx_codec_err == VPX_CODEC_OK)
				{
					vpx_codec_err = vpx_codec_enc_init(&vpx_alpha_encoder, iface, &alpha_config, flags);
				}
				
				
				if(vpx_codec_err == VPX_CODEC_OK)
				{
					if(method == WEBM_METHOD_CONSTANT_QUALITY || method == WEBM_METHOD_CONSTRAINED_QUALITY)
					{
						const int min_q = config.rc_min_quantizer;
						const int max_q = config.rc_max_quantizer;
						
						// CQ Level should be between min_q and max_q
						const int cq_level = (min_q + max_q) / 2;
					
						vpx_codec_control(&vpx_encoder, VP8E_SET_CQ_LEVEL, cq_level);
						
						if(use_alpha)
							vpx_codec_control(&vpx_alpha_encoder, VP8E_SET_CQ_LEVEL, cq_level);
					}
					
					if(video_codec == WEBM_CODEC_VP9)
					{
						vpx_codec_control(&vpx_encoder, VP8E_SET_CPUUSED, 2); // much faster if we do this
						
						vpx_codec_control(&vpx_encoder, VP9E_SET_TILE_COLUMNS, mylog2(g_num_cpus)); // this gives us some multithreading
						vpx_codec_control(&vpx_encoder, VP9E_SET_FRAME_PARALLEL_DECODING, 1);
						
						if(use_alpha)
						{
							vpx_codec_control(&vpx_alpha_encoder, VP8E_SET_CPUUSED, 2);
							
							vpx_codec_control(&vpx_alpha_encoder, VP9E_SET_TILE_COLUMNS, mylog2(g_num_cpus));
							vpx_codec_control(&vpx_alpha_encoder, VP9E_SET_FRAME_PARALLEL_DECODING, 1);
						}
					}
				
					ConfigureVPXEncoderPost(&vpx_encoder, customArgs);
					
					if(use_alpha)
						ConfigureVPXEncoderPost(&vpx_alpha_encoder, customArgs);
				}
			}
			else
			{
				assert(video_codec == WEBM_CODEC_AV1);
				
				if(av1_codec == AV1_CODEC_AOM && result == malNoError)
				{
					aom_codec_iface_t* iface = aom_codec_av1_cx();

					aom_codec_enc_cfg_t config;
					aom_codec_enc_config_default(iface, &config, AOM_USAGE_GOOD_QUALITY);

					config.g_w = renderParms.inWidth;
					config.g_h = renderParms.inHeight;

					// Profile 0 (Main) can do 8- and 10-bit 4:2:0 and 4:0:0 (monochrome)
					// Profile 1 (High) adds 4:4:4
					// Profile 2 (Professional) adds 12-bit and 4:2:2
					config.g_profile = (bit_depth == 12 || chroma == WEBM_422) ? 2 :
										chroma == WEBM_444 ? 1 : 0;

					config.g_bit_depth = (bit_depth == 12 ? AOM_BITS_12 :
											bit_depth == 10 ? AOM_BITS_10 :
											AOM_BITS_8);

					config.g_input_bit_depth = config.g_bit_depth;


					if(method == WEBM_METHOD_CONSTANT_QUALITY || method == WEBM_METHOD_CONSTRAINED_QUALITY)
					{
						config.rc_end_usage = (method == WEBM_METHOD_CONSTANT_QUALITY ? AOM_Q : AOM_CQ);
						config.g_pass = AOM_RC_ONE_PASS;

						const int min_q = config.rc_min_quantizer + 1;
						const int max_q = config.rc_max_quantizer;

						// our 0...100 slider will be used to bring max_q down to min_q
						config.rc_max_quantizer = min_q + ((((float)(100 - videoQualityP.value.intValue) / 100.f) * (max_q - min_q)) + 0.5f);
					}
					else
					{
						if(method == WEBM_METHOD_VBR)
						{
							config.rc_end_usage = AOM_VBR;
						}
						else if(method == WEBM_METHOD_BITRATE)
						{
							config.rc_end_usage = AOM_CBR;
							config.g_pass = AOM_RC_ONE_PASS;
						}
						else
							assert(false);
					}

					if(passes == 2)
					{
						if(vbr_pass)
						{
							config.g_pass = AOM_RC_FIRST_PASS;
						}
						else
						{
							config.g_pass = AOM_RC_LAST_PASS;

							config.rc_twopass_stats_in.buf = vbr_buffer;
							config.rc_twopass_stats_in.sz = vbr_buffer_size;
						}
					}
					else
						config.g_pass = AOM_RC_ONE_PASS;


					config.rc_target_bitrate = bitrateP.value.intValue;


					config.g_threads = g_num_cpus;

					config.g_timebase.num = fps.denominator;
					config.g_timebase.den = fps.numerator;

					config.kf_max_dist = keyframeMaxDistanceP.value.intValue;


					ConfigureAOMEncoderPre(config, customArgs);

					assert(config.kf_max_dist >= config.kf_min_dist);


					aom_codec_enc_cfg_t alpha_config = config;

					if(use_alpha)
					{
						alpha_config.monochrome = 1;

						alpha_config.g_profile = (bit_depth == 12 ? 2 : 0);

						alpha_config.kf_min_dist = alpha_config.kf_max_dist = config.kf_min_dist = config.kf_max_dist;

						alpha_config.rc_target_bitrate = config.rc_target_bitrate / 3;

						config.rc_target_bitrate = config.rc_target_bitrate * 2 / 3;

						if(passes == 2 && !vbr_pass)
						{
							config.rc_twopass_stats_in.buf = alpha_vbr_buffer;
							config.rc_twopass_stats_in.sz = alpha_vbr_buffer_size;
						}
					}


					const aom_codec_flags_t flags = (config.g_bit_depth == AOM_BITS_8 ? 0 : AOM_CODEC_USE_HIGHBITDEPTH);

					aom_codec_err = aom_codec_enc_init(&aom_encoder, iface, &config, flags);

					if(use_alpha && aom_codec_err == AOM_CODEC_OK)
					{
						aom_codec_err = aom_codec_enc_init(&aom_alpha_encoder, iface, &alpha_config, flags);
					}


					if(aom_codec_err == AOM_CODEC_OK)
					{
						if(method == WEBM_METHOD_CONSTANT_QUALITY || method == WEBM_METHOD_CONSTRAINED_QUALITY)
						{
							const int min_q = config.rc_min_quantizer;
							const int max_q = config.rc_max_quantizer;

							// CQ Level should be between min_q and max_q
							const int cq_level = (min_q + max_q) / 2;

							aom_codec_control(&aom_encoder, AOME_SET_CQ_LEVEL, cq_level);

							if(use_alpha)
								aom_codec_control(&aom_alpha_encoder, AOME_SET_CQ_LEVEL, cq_level);
						}

						aom_codec_control(&aom_encoder, AOME_SET_CPUUSED, 2); // much faster if we do this?

						aom_codec_control(&aom_encoder, AV1E_SET_TILE_COLUMNS, mylog2(g_num_cpus)); // this gives us some multithreading
						aom_codec_control(&aom_encoder, AV1E_SET_FRAME_PARALLEL_DECODING, 1);

						if(use_alpha)
						{
							aom_codec_control(&aom_alpha_encoder, AOME_SET_CPUUSED, 2);

							aom_codec_control(&aom_alpha_encoder, AV1E_SET_TILE_COLUMNS, mylog2(g_num_cpus));
							aom_codec_control(&aom_alpha_encoder, AV1E_SET_FRAME_PARALLEL_DECODING, 1);
						}

						ConfigureAOMEncoderPost(&aom_encoder, customArgs);

						if(use_alpha)
							ConfigureAOMEncoderPost(&aom_alpha_encoder, customArgs);
					}
				}
				else
					assert(av1_codec == AV1_CODEC_NVENC);
			}
		}


		if(passes > 1 || !codecMessage.empty())
		{
			std::string msg = (vbr_pass ? "Analyzing video" : "Encoding WebM movie");

			if (!codecMessage.empty())
				msg += ", " + codecMessage;

			prUTF16Char utf_str[256];

			utf16ncpy(utf_str, codecMessage.c_str(), 255);

			mySettings->exportProgressSuite->SetProgressString(exID, utf_str);
		}

	
	#define OV_OK 0
	
		int v_err = OV_OK;
	
		vorbis_info vi;
		vorbis_comment vc;
		vorbis_dsp_state vd;
		vorbis_block vb;
		ogg_packet op;
		
		bool packet_waiting = false;
		op.granulepos = 0;
		op.packet = NULL;
		op.bytes = 0;
		
		OpusMSEncoder *opus = NULL;
		float *opus_buffer = NULL;
		unsigned char *opus_compressed_buffer = NULL;
		opus_int32 opus_compressed_buffer_size = 0;
		int opus_pre_skip = 0;
										
		int opus_frame_size = 960;
		float *pr_audio_buffer[6] = {NULL, NULL, NULL, NULL, NULL, NULL};
		
		size_t private_size = 0;
		void *private_data = NULL;
		
		csSDK_int32 maxBlip = 100;
		
		if(exportInfoP->exportAudio && !vbr_pass && result == malNoError)
		{
			mySettings->sequenceAudioSuite->GetMaxBlip(audioRenderID, frameRateP.value.timeValue, &maxBlip);
			
			if(audio_codec == WEBM_CODEC_OPUS)
			{
				const int sample_rate = 48000;
				
				const int mapping_family = (audioChannels > 2 ? 1 : 0);
				
				const int streams = (audioChannels > 2 ? 4 : 1);
				
				const int coupled_streams = (audioChannels > 2 ? 2 :
												audioChannels == 2 ? 1:
												0);
				
				const unsigned char surround_mapping[6] = {0, 4, 1, 2, 3, 5};
				const unsigned char stereo_mapping[6] = {0, 1, 0, 1, 0, 1};
				
				const unsigned char *mapping = (audioChannels > 2 ? surround_mapping : stereo_mapping);
				
				int err = -1;
				
				opus = opus_multistream_encoder_create(sample_rate, audioChannels,
														streams, coupled_streams, mapping,
														OPUS_APPLICATION_AUDIO, &err);
				
				if(opus != NULL && err == OPUS_OK)
				{
					if(!autoBitrateP.value.intValue) // OPUS_AUTO is the default
						opus_multistream_encoder_ctl(opus, OPUS_SET_BITRATE(opusBitrateP.value.intValue * 1000));
						
				
					// build Opus headers
					// http://wiki.xiph.org/OggOpus
					// http://tools.ietf.org/html/draft-terriberry-oggopus-01
					// http://wiki.xiph.org/MatroskaOpus
					
					// ID header
					unsigned char id_head[28];
					memset(id_head, 0, 28);
					size_t id_header_size = 0;
					
					strcpy((char *)id_head, "OpusHead");
					id_head[8] = 1; // version
					id_head[9] = audioChannels;
					
					
					// pre-skip
					opus_int32 skip = 0;
					opus_multistream_encoder_ctl(opus, OPUS_GET_LOOKAHEAD(&skip));
					opus_pre_skip = skip;
					
					const unsigned short skip_us = skip;
					id_head[10] = skip_us & 0xff;
					id_head[11] = skip_us >> 8;
					
					
					// sample rate
					const unsigned int sample_rate_ui = sample_rate;
					id_head[12] = sample_rate_ui & 0xff;
					id_head[13] = (sample_rate_ui & 0xff00) >> 8;
					id_head[14] = (sample_rate_ui & 0xff0000) >> 16;
					id_head[15] = (sample_rate_ui & 0xff000000) >> 24;
					
					
					// output gain (set to 0)
					id_head[16] = id_head[17] = 0;
					
					
					// channel mapping
					id_head[18] = mapping_family;
					
					if(mapping_family == 1)
					{
						assert(audioChannels == 6);
					
						id_head[19] = streams;
						id_head[20] = coupled_streams;
						memcpy(&id_head[21], mapping, 6);
						
						id_header_size = 27;
					}
					else
					{
						id_header_size = 19;
					}
					
					private_size = id_header_size;
					
					private_data = malloc(private_size);
					
					if(private_data == NULL)
						throw exportReturn_ErrMemory;
					
					memcpy(private_data, id_head, private_size);
					
					
					// figure out the frame size to use
					opus_frame_size = sample_rate / 400;
					
					const int samples_per_frame = sample_rate * fps.denominator / fps.numerator;
					
					while(opus_frame_size * 2 < samples_per_frame && opus_frame_size * 2 < maxBlip)
					{
						opus_frame_size *= 2;
					}
					
					opus_buffer = (float *)malloc(sizeof(float) * audioChannels * opus_frame_size);
					
					if(opus_buffer == NULL)
						throw exportReturn_ErrMemory;
					
					opus_compressed_buffer_size = sizeof(float) * audioChannels * opus_frame_size * 2; // why not?
					
					opus_compressed_buffer = (unsigned char *)malloc(opus_compressed_buffer_size);
					
					if(opus_compressed_buffer == NULL)
						throw exportReturn_ErrMemory;
				}
				else
					v_err = (err != 0 ? err : -1);
			}
			else
			{
				vorbis_info_init(&vi);
				
				
				long min_bitrate = -1, max_bitrate = -1;
				
				vorbis_get_limits(audioChannels, sampleRateP.value.floatValue,
									min_bitrate, max_bitrate);
				
				const bool qualityOnly = (min_bitrate < 0 || max_bitrate < 0); // user should have use Quality
				
									
				if(audioMethodP.value.intValue == OGG_BITRATE && !qualityOnly)
				{
					long bitrate = audioBitrateP.value.intValue * 1000;
					
					if(bitrate < min_bitrate)
						bitrate = min_bitrate;
					else if(bitrate > max_bitrate)
						bitrate = max_bitrate;
					
				
					v_err = vorbis_encode_init(&vi,
												audioChannels,
												sampleRateP.value.floatValue,
												-1,
												bitrate,
												-1);
				}
				else
				{
					v_err = vorbis_encode_init_vbr(&vi,
													audioChannels,
													sampleRateP.value.floatValue,
													audioQualityP.value.floatValue);
				}
				
				if(v_err == OV_OK)
				{
					vorbis_comment_init(&vc);
					vorbis_analysis_init(&vd, &vi);
					vorbis_block_init(&vd, &vb);
					
					
					ogg_packet header;
					ogg_packet header_comm;
					ogg_packet header_code;
					
					vorbis_analysis_headerout(&vd, &vc, &header, &header_comm, &header_code);
					
					private_data = MakePrivateData(header, header_comm, header_code, private_size);
					
					if(private_data == NULL)
						throw exportReturn_ErrMemory;
					
					opus_frame_size = maxBlip;
				}
				else
					exportInfoP->exportAudio = kPrFalse;
			}
			
			for(int i=0; i < audioChannels; i++)
			{
				pr_audio_buffer[i] = (float *)malloc(sizeof(float) * opus_frame_size);
				
				if(pr_audio_buffer[i] == NULL)
					throw exportReturn_ErrMemory;
			}
		}
		
		
		if(vpx_codec_err == VPX_CODEC_OK && aom_codec_err == AOM_CODEC_OK && v_err == OV_OK && result == malNoError)
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
					
					if(video_codec == WEBM_CODEC_AV1)
					{
						if(av1_codec == AV1_CODEC_AOM)
						{
							aom_fixed_buf_t *privateH = aom_codec_get_global_headers(&aom_encoder);

							if(privateH != NULL)
								video->SetCodecPrivate((const uint8_t *)privateH->buf, privateH->sz);
						}
					#ifdef WEBM_HAVE_NVENC
						else if(av1_codec == AV1_CODEC_NVENC)
						{
							void *privateP = malloc(NV_MAX_SEQ_HDR_LEN);

							if(privateP != NULL)
							{
								uint32_t payloadSize = 0;

								NV_ENC_SEQUENCE_PARAM_PAYLOAD payload = { 0 };

								payload.version = NV_ENC_SEQUENCE_PARAM_PAYLOAD_VER;
								payload.inBufferSize = NV_MAX_SEQ_HDR_LEN;
								payload.spsId = 0;
								payload.ppsId = 0;
								payload.spsppsBuffer = privateP;
								payload.outSPSPPSPayloadSize = &payloadSize;

								NVENCSTATUS err = nvenc.nvEncGetSequenceParams(nv_encoder, &payload);

								if(err == NV_ENC_SUCCESS)
									video->SetCodecPrivate((const uint8_t *)privateP, payloadSize);

								free(privateP);
							}
						}
					#endif // WEBM_HAVE_NVENC
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
					
					const uint64_t horizontal_subsampling = (chroma == WEBM_444 ? 0 : 1);
					const uint64_t vertical_subsampling = (chroma == WEBM_420 ? 1 : 0);
					
					color.set_chroma_subsampling_horz(horizontal_subsampling);
					color.set_chroma_subsampling_vert(vertical_subsampling);
					
					// don't want to presume
					//color.set_matrix_coefficients(mkvmuxer::Colour::kBt709);
					//color.set_range(mkvmuxer::Colour::kBroadcastRange);
					//color.set_transfer_characteristics(mkvmuxer::Colour::kIturBt709Tc);
					//color.set_primaries(mkvmuxer::Colour::kIturBt709P);
					
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
					
					mkvmuxer::AudioTrack* const audio = static_cast<mkvmuxer::AudioTrack *>(muxer_segment->GetTrackByNumber(audio_track));
					
					audio->set_codec_id(audio_codec == WEBM_CODEC_OPUS ? mkvmuxer::Tracks::kOpusCodecId :
										mkvmuxer::Tracks::kVorbisCodecId);
					
					if(audio_codec == WEBM_CODEC_OPUS)
					{
						// http://wiki.xiph.org/MatroskaOpus
						
						audio->set_seek_pre_roll(80000000);
						
						audio->set_codec_delay((PrAudioSample)opus_pre_skip * S2NS / (PrAudioSample)sampleRateP.value.floatValue);
					}

					if(private_data)
					{
						bool copied = audio->SetCodecPrivate((const uint8_t *)private_data, private_size);
						
						assert(copied);
						
						free(private_data);
					}

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
			const PrAudioSample endAudioSample = (exportInfoP->endTime - exportInfoP->startTime) /
													(ticksPerSecond / (PrAudioSample)sampleRateP.value.floatValue);
													
			assert(ticksPerSecond % (PrAudioSample)sampleRateP.value.floatValue == 0);
			
			
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
							
					if(audio_codec == WEBM_CODEC_OPUS)
					{
						assert(opus != NULL);
						
						uint64_t opus_timeStamp = currentAudioSample * S2NS / (uint64_t)sampleRateP.value.floatValue;
						
						while(((opus_timeStamp <= timeStamp) || last_frame) && currentAudioSample < (endAudioSample + opus_pre_skip) && result == malNoError)
						{
							const int samples = opus_frame_size;
							
							result = audioSuite->GetAudio(audioRenderID, samples, pr_audio_buffer, false);
							
							if(result == malNoError)
							{
								for(int i=0; i < samples; i++)
								{
									for(int c=0; c < audioChannels; c++)
									{
										opus_buffer[(i * audioChannels) + c] = pr_audio_buffer[c][i];
									}
								}
								
								int len = opus_multistream_encode_float(opus, opus_buffer, opus_frame_size,
																			opus_compressed_buffer, opus_compressed_buffer_size);
								
								if(len > 0)
								{
									bool added = false;
									
									if((currentAudioSample + samples) > (endAudioSample + opus_pre_skip))
									{
										const int64_t discardPaddingSamples = (currentAudioSample + samples) - (endAudioSample + opus_pre_skip);
										const int64_t discardPadding = discardPaddingSamples * S2NS / (int64_t)sampleRateP.value.floatValue;
										
										added = muxer_segment->AddFrameWithDiscardPadding(opus_compressed_buffer, len,
																		discardPadding, audio_track, opus_timeStamp, true);
									}
									else
									{
										added = muxer_segment->AddFrame(opus_compressed_buffer, len,
																			audio_track, opus_timeStamp, true);
									}
																			
									if(!added)
										result = exportReturn_InternalError;
								}
								else if(len < 0)
									result = exportReturn_InternalError;
								
								
								currentAudioSample += samples;
								
								opus_timeStamp = currentAudioSample * S2NS / (uint64_t)sampleRateP.value.floatValue;
							}
						}
					}
					else
					{
						uint64_t op_timeStamp = op.granulepos * S2NS / (uint64_t)sampleRateP.value.floatValue;
					
						while(op_timeStamp <= timeStamp && op.granulepos < endAudioSample && result == malNoError)
						{	
							// We don't know what samples are in the packet until we get it,
							// but by then it's too late to decide if we don't want it for this frame.
							// So we'll hold on to that packet and use it next frame.
							if(packet_waiting && op.packet != NULL && op.bytes > 0)
							{
								bool added = muxer_segment->AddFrame(op.packet, op.bytes,
																	audio_track, op_timeStamp, true);
																		
								if(added)
									packet_waiting = false;
								else
									result = exportReturn_InternalError;
							}
							
							
							// push out packets
							while(vorbis_analysis_blockout(&vd, &vb) == 1 && result == malNoError)
							{
								vorbis_analysis(&vb, NULL);
								vorbis_bitrate_addblock(&vb);
								
								while(vorbis_bitrate_flushpacket(&vd, &op) && result == malNoError)
								{
									assert(!packet_waiting);
									
									op_timeStamp = op.granulepos * S2NS / (uint64_t)sampleRateP.value.floatValue;
									
									if(op_timeStamp <= timeStamp || last_frame)
									{
										bool added = muxer_segment->AddFrame(op.packet, op.bytes,
																			audio_track, op_timeStamp, true);
																				
										if(!added)
											result = exportReturn_InternalError;
									}
									else
										packet_waiting = true;
								}
								
								if(packet_waiting)
									break;
							}
							
							
							if(packet_waiting)
								break;
							
							
							// make new packets
							if(op_timeStamp <= timeStamp && op.granulepos < endAudioSample && result == malNoError)
							{
								int samples = opus_frame_size; // opus_frame_size is also the size of our buffer in samples
								
								assert(samples == maxBlip);
								
								assert(currentAudioSample <= endAudioSample); // so samples won't be negative
								
								if(samples > (endAudioSample - currentAudioSample))
									samples = (endAudioSample - currentAudioSample);
									
								if(samples > 0)
								{
									float **buffer = vorbis_analysis_buffer(&vd, samples);
									
									result = audioSuite->GetAudio(audioRenderID, samples, pr_audio_buffer, false);
									
									for(int c=0; c < audioChannels; c++)
									{
										for(int i=0; i < samples; i++)
										{
											buffer[c][i] = pr_audio_buffer[c][i];
										}
									}
								}
								
								currentAudioSample += samples;
								
								
								if(result == malNoError)
								{
									vorbis_analysis_wrote(&vd, samples);
									
									if(currentAudioSample >= endAudioSample)
										vorbis_analysis_wrote(&vd, NULL); // we have sent everything in
								}
							}
						}
					}
				}
				
				
				if(exportInfoP->exportVideo && (videoTime < exportInfoP->endTime) && result == malNoError) // there will some audio after the last video frame
				{
					bool made_frame = false;
					
					while(!made_frame && result == suiteError_NoError)
					{
						if(video_codec == WEBM_CODEC_VP8 || video_codec == WEBM_CODEC_VP9)
						{
							while(const vpx_codec_cx_pkt_t *pkt = vpx_codec_get_cx_data(&vpx_encoder, &vpx_encoder_iter))
							{
								if(vbr_pass)
								{
									if(pkt->kind == VPX_CODEC_STATS_PKT)
									{
										vpx_codec_cx_pkt_t q_pkt = *pkt;

										if(copy_buffers)
										{
											q_pkt.data.twopass_stats.buf = malloc(q_pkt.data.twopass_stats.sz);
											if(q_pkt.data.twopass_stats.buf == NULL)
												throw exportReturn_ErrMemory;
											memcpy(q_pkt.data.twopass_stats.buf, pkt->data.twopass_stats.buf, q_pkt.data.twopass_stats.sz);
										}

										vpx_encoder_queue.push(q_pkt);
									}
								}
								else
								{
									if(pkt->kind == VPX_CODEC_CX_FRAME_PKT)
									{
										vpx_codec_cx_pkt_t q_pkt = *pkt;

										if(copy_buffers)
										{
											q_pkt.data.frame.buf = malloc(q_pkt.data.frame.sz);
											if(q_pkt.data.frame.buf == NULL)
												throw exportReturn_ErrMemory;
											memcpy(q_pkt.data.frame.buf, pkt->data.frame.buf, q_pkt.data.frame.sz);
										}

										vpx_encoder_queue.push(q_pkt);
									}
								}
								
								assert(pkt->kind != VPX_CODEC_FPMB_STATS_PKT); // don't know what to do with this

								if(!copy_buffers)
									break;
							}
						}
						else
						{
							assert(video_codec == WEBM_CODEC_AV1);
							
							if(av1_codec == AV1_CODEC_AOM)
							{
								while(const aom_codec_cx_pkt_t* pkt = aom_codec_get_cx_data(&aom_encoder, &aom_encoder_iter))
								{
									if(vbr_pass)
									{
										if(pkt->kind == AOM_CODEC_STATS_PKT)
										{
											aom_codec_cx_pkt_t q_pkt = *pkt;

											if(copy_buffers)
											{
												q_pkt.data.twopass_stats.buf = malloc(q_pkt.data.twopass_stats.sz);
												if(q_pkt.data.twopass_stats.buf == NULL)
													throw exportReturn_ErrMemory;
												memcpy(q_pkt.data.twopass_stats.buf, pkt->data.twopass_stats.buf, q_pkt.data.twopass_stats.sz);
											}

											aom_encoder_queue.push(q_pkt);
										}
									}
									else
									{
										if(pkt->kind == AOM_CODEC_CX_FRAME_PKT)
										{
											aom_codec_cx_pkt_t q_pkt = *pkt;

											if(copy_buffers)
											{
												q_pkt.data.frame.buf = malloc(q_pkt.data.frame.sz);
												if(q_pkt.data.frame.buf == NULL)
													throw exportReturn_ErrMemory;
												memcpy(q_pkt.data.frame.buf, pkt->data.frame.buf, q_pkt.data.frame.sz);
											}

											aom_encoder_queue.push(q_pkt);
										}
									}

									assert(pkt->kind != VPX_CODEC_FPMB_STATS_PKT); // don't know what to do with this

									if(!copy_buffers)
										break;
								}
							}
						#ifdef WEBM_HAVE_NVENC
							else if(av1_codec == AV1_CODEC_NVENC)
							{
								while(nv_output_available)
								{
									assert(nv_output_buffer_idx < nv_input_buffer_idx);

									if(vbr_pass)
									{
										assert(false);
									}
									else
									{
										NV_ENC_LOCK_BITSTREAM lock = { 0 };

										lock.version = NV_ENC_LOCK_BITSTREAM_VER;
										lock.outputBitstream = nv_output_buffers[nv_output_buffer_idx];

										nv_err = nvenc.nvEncLockBitstream(nv_encoder, &lock);

										if(nv_err == NV_ENC_SUCCESS)
										{
											NV_ENC_LOCK_BITSTREAM q_lock = lock;
											q_lock.bitstreamBufferPtr = malloc(q_lock.bitstreamSizeInBytes);
											if (q_lock.bitstreamBufferPtr == NULL)
												throw exportReturn_ErrMemory;
											memcpy(q_lock.bitstreamBufferPtr, lock.bitstreamBufferPtr, q_lock.bitstreamSizeInBytes);
											nv_encoder_queue.push(q_lock);

											nvenc.nvEncUnlockBitstream(nv_encoder, nv_output_buffers[nv_output_buffer_idx]);

											nv_output_buffer_idx++;

											if(nv_output_buffer_idx == nv_input_buffer_idx)
											{
												nv_output_buffer_idx = nv_input_buffer_idx = 0;

												nv_output_available = false;
											}
										}
										else if(nv_err == NV_ENC_ERR_INVALID_PARAM)
										{
											// Huh? I guess the next buffer isn't ready?

											nv_output_available = false;
										}
										else
											result = exportReturn_InternalError;
									}
								}
							}
						#endif // WEBM_HAVE_NVENC
							else
								assert(false);
						}
						
						if(use_alpha)
						{
							assert(copy_buffers);
							
							if(video_codec == WEBM_CODEC_VP8 || video_codec == WEBM_CODEC_VP9)
							{
								while(const vpx_codec_cx_pkt_t *pkt = vpx_codec_get_cx_data(&vpx_alpha_encoder, &vpx_alpha_encoder_iter))
								{
									if(vbr_pass)
									{
										if(pkt->kind == VPX_CODEC_STATS_PKT)
										{
											vpx_codec_cx_pkt_t q_pkt = *pkt;
											q_pkt.data.twopass_stats.buf = malloc(q_pkt.data.twopass_stats.sz);
											if(q_pkt.data.twopass_stats.buf == NULL)
												throw exportReturn_ErrMemory;
											memcpy(q_pkt.data.twopass_stats.buf, pkt->data.twopass_stats.buf, q_pkt.data.twopass_stats.sz);
											vpx_alpha_encoder_queue.push(q_pkt);
										}
									}
									else
									{
										if(pkt->kind == VPX_CODEC_CX_FRAME_PKT)
										{
											vpx_codec_cx_pkt_t q_pkt = *pkt;
											q_pkt.data.frame.buf = malloc(q_pkt.data.frame.sz);
											if(q_pkt.data.frame.buf == NULL)
												throw exportReturn_ErrMemory;
											memcpy(q_pkt.data.frame.buf, pkt->data.frame.buf, q_pkt.data.frame.sz);
											vpx_alpha_encoder_queue.push(q_pkt);
										}
									}
									
									assert(pkt->kind != VPX_CODEC_FPMB_STATS_PKT); // don't know what to do with this
								}
							}
							else
							{
								assert(video_codec == WEBM_CODEC_AV1);
								
								if(av1_codec == AV1_CODEC_AOM)
								{
									while(const aom_codec_cx_pkt_t* pkt = aom_codec_get_cx_data(&aom_alpha_encoder, &aom_alpha_encoder_iter))
									{
										if(vbr_pass)
										{
											if(pkt->kind == AOM_CODEC_STATS_PKT)
											{
												aom_codec_cx_pkt_t q_pkt = *pkt;
												q_pkt.data.twopass_stats.buf = malloc(q_pkt.data.twopass_stats.sz);
												if(q_pkt.data.twopass_stats.buf == NULL)
													throw exportReturn_ErrMemory;
												memcpy(q_pkt.data.twopass_stats.buf, pkt->data.twopass_stats.buf, q_pkt.data.twopass_stats.sz);
												aom_alpha_encoder_queue.push(q_pkt);
											}
										}
										else
										{
											if(pkt->kind == AOM_CODEC_CX_FRAME_PKT)
											{
												aom_codec_cx_pkt_t q_pkt = *pkt;
												q_pkt.data.frame.buf = malloc(q_pkt.data.frame.sz);
												if(q_pkt.data.frame.buf == NULL)
													throw exportReturn_ErrMemory;
												memcpy(q_pkt.data.frame.buf, pkt->data.frame.buf, q_pkt.data.frame.sz);
												aom_alpha_encoder_queue.push(q_pkt);
											}
										}

										assert(pkt->kind != AOM_CODEC_FPMB_STATS_PKT); // don't know what to do with this
									}
								}
							#ifdef WEBM_HAVE_NVENC
								else if(av1_codec == AV1_CODEC_NVENC)
								{
									while(nv_alpha_output_available)
									{
										assert(nv_alpha_output_buffer_idx < nv_alpha_input_buffer_idx);

										if(vbr_pass)
										{
											assert(false);
										}
										else
										{
											NV_ENC_LOCK_BITSTREAM lock = { 0 };

											lock.version = NV_ENC_LOCK_BITSTREAM_VER;
											lock.outputBitstream = nv_alpha_output_buffers[nv_alpha_output_buffer_idx];

											nv_err = nvenc.nvEncLockBitstream(nv_alpha_encoder, &lock);

											if(nv_err == NV_ENC_SUCCESS)
											{
												NV_ENC_LOCK_BITSTREAM q_lock = lock;
												q_lock.bitstreamBufferPtr = malloc(q_lock.bitstreamSizeInBytes);
												if(q_lock.bitstreamBufferPtr == NULL)
													throw exportReturn_ErrMemory;
												memcpy(q_lock.bitstreamBufferPtr, lock.bitstreamBufferPtr, q_lock.bitstreamSizeInBytes);
												nv_alpha_encoder_queue.push(q_lock);

												nvenc.nvEncUnlockBitstream(nv_alpha_encoder, nv_alpha_output_buffers[nv_alpha_output_buffer_idx]);

												nv_alpha_output_buffer_idx++;

												if(nv_alpha_output_buffer_idx == nv_alpha_input_buffer_idx)
												{
													nv_alpha_output_buffer_idx = nv_alpha_input_buffer_idx = 0;

													nv_alpha_output_available = false;
												}
											}
											else if(nv_err == NV_ENC_ERR_INVALID_PARAM)
											{
												nv_alpha_output_available = false;
											}
											else
												result = exportReturn_InternalError;
										}
									}
								}
							#endif // WEBM_HAVE_NVENC
								else
									assert(false);
							}
						}
						
						if(video_codec == WEBM_CODEC_VP8 || video_codec == WEBM_CODEC_VP9)
						{
							if(!vpx_encoder_queue.empty() && (!use_alpha || !vpx_alpha_encoder_queue.empty()))
							{
								vpx_codec_cx_pkt_t *pkt = NULL;
								vpx_codec_cx_pkt_t *alpha_pkt = NULL;
							
								vpx_codec_cx_pkt_t pkt_data;
								vpx_codec_cx_pkt_t alpha_pkt_data;
								
								pkt_data = vpx_encoder_queue.front();
								pkt = &pkt_data;
								
								if(use_alpha)
								{
									alpha_pkt_data = vpx_alpha_encoder_queue.front();
									alpha_pkt = &alpha_pkt_data;
								}
							
								if(pkt->kind == VPX_CODEC_STATS_PKT)
								{
									assert(vbr_pass);
								
									if(vbr_buffer_size == 0)
										vbr_buffer = memorySuite->NewPtr(pkt->data.twopass_stats.sz);
									else
										memorySuite->SetPtrSize(&vbr_buffer, vbr_buffer_size + pkt->data.twopass_stats.sz);
									
									memcpy(&vbr_buffer[vbr_buffer_size], pkt->data.twopass_stats.buf, pkt->data.twopass_stats.sz);
									
									vbr_buffer_size += pkt->data.twopass_stats.sz;
									
									made_frame = true;
									
									if(use_alpha)
									{
										assert(alpha_pkt->kind == VPX_CODEC_STATS_PKT);
										
										if(alpha_vbr_buffer_size == 0)
											alpha_vbr_buffer = memorySuite->NewPtr(alpha_pkt->data.twopass_stats.sz);
										else
											memorySuite->SetPtrSize(&alpha_vbr_buffer, alpha_vbr_buffer_size + alpha_pkt->data.twopass_stats.sz);
										
										memcpy(&alpha_vbr_buffer[alpha_vbr_buffer_size], alpha_pkt->data.twopass_stats.buf, alpha_pkt->data.twopass_stats.sz);
										
										alpha_vbr_buffer_size += alpha_pkt->data.twopass_stats.sz;
									}
								}
								else if(pkt->kind == VPX_CODEC_CX_FRAME_PKT)
								{
									assert( !vbr_pass );
									assert( !(pkt->data.frame.flags & VPX_FRAME_IS_INVISIBLE) ); // libwebm not handling these now
									assert( !(pkt->data.frame.flags & VPX_FRAME_IS_FRAGMENT) );
									assert( pkt->data.frame.pts == (videoTime - exportInfoP->startTime) * fps.numerator / (ticksPerSecond * fps.denominator) );
									assert( pkt->data.frame.duration == 1 ); // because of how we did the timescale
								
									if(use_alpha)
									{
										assert( !(alpha_pkt->data.frame.flags & VPX_FRAME_IS_INVISIBLE) );
										assert( !(alpha_pkt->data.frame.flags & VPX_FRAME_IS_FRAGMENT) );
										assert( alpha_pkt->data.frame.pts == (videoTime - exportInfoP->startTime) * fps.numerator / (ticksPerSecond * fps.denominator) );
										assert( alpha_pkt->data.frame.duration == 1 );
										
										assert(alpha_pkt->kind == VPX_CODEC_CX_FRAME_PKT);
										
										if(pkt->data.frame.flags & VPX_FRAME_IS_KEY)
											assert(alpha_pkt->data.frame.flags & VPX_FRAME_IS_KEY);
										
										bool added = muxer_segment->AddFrameWithAdditional((const uint8_t *)pkt->data.frame.buf, pkt->data.frame.sz,
																							(const uint8_t *)alpha_pkt->data.frame.buf, alpha_pkt->data.frame.sz, alpha_id,
																							vid_track, timeStamp,
																							pkt->data.frame.flags & VPX_FRAME_IS_KEY);
																			
										if( !(pkt->data.frame.flags & VPX_FRAME_IS_INVISIBLE) )
											made_frame = true;
										
										if(!added)
											result = exportReturn_InternalError;
									}
									else
									{
										bool added = muxer_segment->AddFrame((const uint8_t *)pkt->data.frame.buf, pkt->data.frame.sz,
																			vid_track, timeStamp,
																			pkt->data.frame.flags & VPX_FRAME_IS_KEY);
																			
										if( !(pkt->data.frame.flags & VPX_FRAME_IS_INVISIBLE) )
											made_frame = true;
										
										if(!added)
											result = exportReturn_InternalError;
									}
								}
								
								if(copy_buffers)
									free(vbr_pass ? pkt_data.data.twopass_stats.buf : pkt_data.data.frame.buf);

								vpx_encoder_queue.pop();
								
								if(use_alpha)
								{
									assert(copy_buffers);
									free(vbr_pass ? alpha_pkt_data.data.twopass_stats.buf : alpha_pkt_data.data.frame.buf);
									vpx_alpha_encoder_queue.pop();
								}
							}
						}
						else
						{
							assert(video_codec == WEBM_CODEC_AV1);
							
							if(av1_codec == AV1_CODEC_AOM)
							{
								if(!aom_encoder_queue.empty() && (!use_alpha || !aom_alpha_encoder_queue.empty()))
								{
									aom_codec_cx_pkt_t* pkt = NULL;
									aom_codec_cx_pkt_t* alpha_pkt = NULL;

									aom_codec_cx_pkt_t pkt_data;
									aom_codec_cx_pkt_t alpha_pkt_data;

									pkt_data = aom_encoder_queue.front();
									pkt = &pkt_data;

									if(use_alpha)
									{
										alpha_pkt_data = aom_alpha_encoder_queue.front();
										alpha_pkt = &alpha_pkt_data;
									}

									if(pkt->kind == AOM_CODEC_STATS_PKT)
									{
										assert(vbr_pass);

										if(vbr_buffer_size == 0)
											vbr_buffer = memorySuite->NewPtr(pkt->data.twopass_stats.sz);
										else
											memorySuite->SetPtrSize(&vbr_buffer, vbr_buffer_size + pkt->data.twopass_stats.sz);

										memcpy(&vbr_buffer[vbr_buffer_size], pkt->data.twopass_stats.buf, pkt->data.twopass_stats.sz);

										vbr_buffer_size += pkt->data.twopass_stats.sz;

										made_frame = true;

										if(use_alpha)
										{
											assert(alpha_pkt->kind == AOM_CODEC_STATS_PKT);

											if(alpha_vbr_buffer_size == 0)
												alpha_vbr_buffer = memorySuite->NewPtr(alpha_pkt->data.twopass_stats.sz);
											else
												memorySuite->SetPtrSize(&alpha_vbr_buffer, alpha_vbr_buffer_size + alpha_pkt->data.twopass_stats.sz);

											memcpy(&alpha_vbr_buffer[alpha_vbr_buffer_size], alpha_pkt->data.twopass_stats.buf, alpha_pkt->data.twopass_stats.sz);

											alpha_vbr_buffer_size += alpha_pkt->data.twopass_stats.sz;
										}
									}
									else if(pkt->kind == AOM_CODEC_CX_FRAME_PKT)
									{
										assert(!vbr_pass);
										assert(pkt->data.frame.pts == (videoTime - exportInfoP->startTime) * fps.numerator / (ticksPerSecond * fps.denominator));
										assert(pkt->data.frame.duration == 1); // because of how we did the timescale

										if(use_alpha)
										{
											assert(alpha_pkt->data.frame.pts == (videoTime - exportInfoP->startTime) * fps.numerator / (ticksPerSecond * fps.denominator));
											assert(alpha_pkt->data.frame.duration == 1);

											assert(alpha_pkt->kind == AOM_CODEC_CX_FRAME_PKT);

											if(pkt->data.frame.flags & AOM_FRAME_IS_KEY)
												assert(alpha_pkt->data.frame.flags & AOM_FRAME_IS_KEY);

											bool added = muxer_segment->AddFrameWithAdditional((const uint8_t*)pkt->data.frame.buf, pkt->data.frame.sz,
																								(const uint8_t*)alpha_pkt->data.frame.buf, alpha_pkt->data.frame.sz, alpha_id,
																								vid_track, timeStamp,
																								pkt->data.frame.flags & AOM_FRAME_IS_KEY);

											made_frame = true;

											if(!added)
												result = exportReturn_InternalError;
										}
										else
										{
											bool added = muxer_segment->AddFrame((const uint8_t*)pkt->data.frame.buf, pkt->data.frame.sz,
																					vid_track, timeStamp,
																					pkt->data.frame.flags & AOM_FRAME_IS_KEY);

											made_frame = true;

											if(!added)
												result = exportReturn_InternalError;
										}
									}

									if(copy_buffers)
										free(vbr_pass ? pkt_data.data.twopass_stats.buf : pkt_data.data.frame.buf);

									aom_encoder_queue.pop();

									if(use_alpha)
									{
										assert(copy_buffers);
										free(vbr_pass ? alpha_pkt_data.data.twopass_stats.buf : alpha_pkt_data.data.frame.buf);
										aom_alpha_encoder_queue.pop();
									}
								}
							}
						#ifdef WEBM_HAVE_NVENC
							else if(av1_codec == AV1_CODEC_NVENC)
							{
								if(!nv_encoder_queue.empty() && (!use_alpha || !nv_alpha_encoder_queue.empty()))
								{
									NV_ENC_LOCK_BITSTREAM &lock = nv_encoder_queue.front();

									assert(lock.outputTimeStamp == (videoTime - exportInfoP->startTime) * fps.numerator / (ticksPerSecond * fps.denominator));
									assert(lock.outputDuration == 1);
									assert(lock.pictureStruct == NV_ENC_PIC_STRUCT_FRAME);
									assert(lock.pictureType != NV_ENC_PIC_TYPE_I); // not I, IDR!

									if(use_alpha)
									{
										NV_ENC_LOCK_BITSTREAM &alpha_lock = nv_alpha_encoder_queue.front();

										assert(alpha_lock.outputTimeStamp == (videoTime - exportInfoP->startTime) * fps.numerator / (ticksPerSecond * fps.denominator));
										assert(alpha_lock.outputDuration == 1);
										assert(alpha_lock.pictureStruct == NV_ENC_PIC_STRUCT_FRAME);
										assert(alpha_lock.pictureType != NV_ENC_PIC_TYPE_I);

										if(lock.pictureType == NV_ENC_PIC_TYPE_IDR)
											assert(alpha_lock.pictureType == NV_ENC_PIC_TYPE_IDR);

										bool added = muxer_segment->AddFrameWithAdditional((const uint8_t*)lock.bitstreamBufferPtr, lock.bitstreamSizeInBytes,
																							(const uint8_t*)alpha_lock.bitstreamBufferPtr, alpha_lock.bitstreamSizeInBytes, alpha_id,
																							vid_track, timeStamp,
																							lock.pictureType == NV_ENC_PIC_TYPE_IDR);

										made_frame = true;

										if(!added)
											result = exportReturn_InternalError;

										free(alpha_lock.bitstreamBufferPtr);

										nv_alpha_encoder_queue.pop();
									}
									else
									{
										bool added = muxer_segment->AddFrame((const uint8_t*)lock.bitstreamBufferPtr, lock.bitstreamSizeInBytes,
																			vid_track, timeStamp,
																			lock.pictureType == NV_ENC_PIC_TYPE_IDR);

										made_frame = true;

										if(!added)
											result = exportReturn_InternalError;
									}
									
									free(lock.bitstreamBufferPtr);

									nv_encoder_queue.pop();
								}
							}
						#endif
							else
								assert(false);
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
								
								if(video_codec == WEBM_CODEC_VP8 || video_codec == WEBM_CODEC_VP9)
								{
									assert(vpx_encoder_queue.empty());
									assert(vpx_alpha_encoder_queue.empty());
								}
								else
								{
									assert(video_codec == WEBM_CODEC_AV1);
									
									if(av1_codec == AV1_CODEC_AOM)
									{
										assert(aom_encoder_queue.empty());
										assert(aom_alpha_encoder_queue.empty());
									}
								#ifdef WEBM_HAVE_NVENC
									else if(av1_codec == AV1_CODEC_NVENC)
									{
										assert(false); // not doing NVENC 2-pass
										assert(nv_encoder_queue.empty());
										assert(nv_alpha_encoder_queue.empty());
									}
								#endif // WEBM_HAVE_NVENC
									else
										assert(false);
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
							
							const vpx_codec_pts_t encoder_timeStamp = encoder_fileTime * fps.numerator / (ticksPerSecond * fps.denominator);
							const vpx_codec_pts_t encoder_nextTimeStamp = encoder_nextFileTime * fps.numerator / (ticksPerSecond * fps.denominator);
							const unsigned long encoder_duration = encoder_nextTimeStamp - encoder_timeStamp;
							
							// BUT, if we're setting timebase to 1/fps, then timestamp is just frame number.
							// And since frame number isn't going to overflow at big times the way encoder_timeStamp is,
							// let's just use that.
							const vpx_codec_pts_t encoder_FrameNumber = encoder_fileTime / frameRateP.value.timeValue;
							const unsigned long encoder_FrameDuration = 1;
							
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
									
									if(video_codec == WEBM_CODEC_VP8 || video_codec == WEBM_CODEC_VP9)
									{
										// see validate_img() and validate_config() in vp8_cx_iface.c and vp9_cx_iface.c
										const vpx_img_fmt_t imgfmt8 = chroma == WEBM_444 ? VPX_IMG_FMT_I444 :
																		chroma == WEBM_422 ? VPX_IMG_FMT_I422 :
																		VPX_IMG_FMT_I420;
																		
										const vpx_img_fmt_t imgfmt16 = chroma == WEBM_444 ? VPX_IMG_FMT_I44416 :
																		chroma == WEBM_422 ? VPX_IMG_FMT_I42216 :
																		VPX_IMG_FMT_I42016;
																		
										const vpx_img_fmt_t imgfmt = (bit_depth > 8 ? imgfmt16 : imgfmt8);
										
												
										vpx_image_t img_data;
										vpx_image_t *img = vpx_img_alloc(&img_data, imgfmt, width, height, 32);
										
										vpx_image_t alpha_img_data;
										vpx_image_t *alpha_img = NULL;
										
										if(use_alpha)
											alpha_img = vpx_img_alloc(&alpha_img_data, imgfmt, width, height, 32);
										
										
										if(img && (!use_alpha || alpha_img))
										{
											if(bit_depth > 8)
											{
												img->bit_depth = bit_depth;
												img->bps = img->bps * bit_depth / 16;
												
												if(use_alpha)
												{
													alpha_img->bit_depth = bit_depth;
													alpha_img->bps = alpha_img->bps * bit_depth / 16;
												}
											}
										
											CopyPixToVPXImg(img, alpha_img, renderResult.outFrame, pixSuite, pix2Suite);
											
											
											vpx_codec_err_t encode_err = vpx_codec_encode(&vpx_encoder, img, encoder_FrameNumber, encoder_FrameDuration, 0, vpx_deadline);
											
											if(encode_err == VPX_CODEC_OK)
											{
												videoEncoderTime += frameRateP.value.timeValue;
												
												vpx_encoder_iter = NULL;
											}
											else
												result = exportReturn_InternalError;
											
											vpx_img_free(img);
											
											
											if(use_alpha)
											{
												vpx_codec_err_t alpha_encode_err = vpx_codec_encode(&vpx_alpha_encoder, alpha_img, encoder_FrameNumber, encoder_FrameDuration, 0, vpx_deadline);
												
												if(alpha_encode_err == VPX_CODEC_OK)
												{
													vpx_alpha_encoder_iter = NULL;
												}
												else
													result = exportReturn_InternalError;
												
												vpx_img_free(alpha_img);
											}
										}
										else
											result = exportReturn_ErrMemory;
									}
									else
									{
										assert(video_codec == WEBM_CODEC_AV1);
										
										if(av1_codec == AV1_CODEC_AOM)
										{
											const aom_img_fmt_t imgfmt8 = chroma == WEBM_444 ? AOM_IMG_FMT_I444 :
												chroma == WEBM_422 ? AOM_IMG_FMT_I422 :
												AOM_IMG_FMT_I420;

											const aom_img_fmt_t imgfmt16 = chroma == WEBM_444 ? AOM_IMG_FMT_I44416 :
												chroma == WEBM_422 ? AOM_IMG_FMT_I42216 :
												AOM_IMG_FMT_I42016;

											const aom_img_fmt_t imgfmt = (bit_depth > 8 ? imgfmt16 : imgfmt8);


											aom_image_t img_data;
											aom_image_t* img = aom_img_alloc(&img_data, imgfmt, width, height, 32);

											aom_image_t alpha_img_data;
											aom_image_t* alpha_img = NULL;

											if(use_alpha)
												alpha_img = aom_img_alloc(&alpha_img_data, imgfmt, width, height, 32);


											if(img && (!use_alpha || alpha_img))
											{
												if(bit_depth > 8)
												{
													img->bit_depth = bit_depth;
													img->bps = img->bps * bit_depth / 16;

													if(use_alpha)
													{
														alpha_img->bit_depth = bit_depth;
														alpha_img->bps = alpha_img->bps * bit_depth / 16;
													}
												}

												CopyPixToAOMImg(img, alpha_img, renderResult.outFrame, pixSuite, pix2Suite);


												aom_codec_err_t encode_err = aom_codec_encode(&aom_encoder, img, encoder_FrameNumber, encoder_FrameDuration, 0);

												if(encode_err == AOM_CODEC_OK)
												{
													videoEncoderTime += frameRateP.value.timeValue;

													aom_encoder_iter = NULL;
												}
												else
													result = exportReturn_InternalError;

												aom_img_free(img);


												if(use_alpha)
												{
													aom_codec_err_t alpha_encode_err = aom_codec_encode(&aom_alpha_encoder, alpha_img, encoder_FrameNumber, encoder_FrameDuration, 0);

													if(alpha_encode_err == AOM_CODEC_OK)
													{
														aom_alpha_encoder_iter = NULL;
													}
													else
														result = exportReturn_InternalError;

													aom_img_free(alpha_img);
												}
											}
											else
												result = exportReturn_ErrMemory;
										}
									#ifdef WEBM_HAVE_NVENC
										else if(av1_codec == AV1_CODEC_NVENC)
										{
											if(nv_input_buffer_idx < nv_input_buffers.size() && (!use_alpha || nv_alpha_input_buffer_idx < nv_alpha_input_buffers.size()))
											{
												NV_ENC_LOCK_INPUT_BUFFER lockParams = { 0 };

												lockParams.version = NV_ENC_LOCK_INPUT_BUFFER_VER;
												lockParams.doNotWait = FALSE;
												lockParams.inputBuffer = nv_input_buffers[nv_input_buffer_idx];
												lockParams.bufferDataPtr = NULL;
												lockParams.pitch = 0;

												NV_ENC_LOCK_INPUT_BUFFER alphaLockParams = lockParams;

												nv_err = nvenc.nvEncLockInputBuffer(nv_encoder, &lockParams);

												if(use_alpha && nv_err == NV_ENC_SUCCESS)
												{
													alphaLockParams.inputBuffer = nv_alpha_input_buffers[nv_alpha_input_buffer_idx];

													nv_err = nvenc.nvEncLockInputBuffer(nv_alpha_encoder, &alphaLockParams);
												}

												if(nv_err == NV_ENC_SUCCESS)
												{
													CopyPixToNVENCBuf(lockParams.bufferDataPtr, lockParams.pitch, nv_input_format,
																		alphaLockParams.bufferDataPtr, alphaLockParams.pitch, nv_alpha_input_format,
																		width, height,
																		renderResult.outFrame, pixSuite, pix2Suite);

													nv_err = nvenc.nvEncUnlockInputBuffer(nv_encoder, nv_input_buffers[nv_input_buffer_idx]);

													if(use_alpha && nv_err == NV_ENC_SUCCESS)
														nv_err = nvenc.nvEncUnlockInputBuffer(nv_alpha_encoder, nv_alpha_input_buffers[nv_alpha_input_buffer_idx]);

													if(nv_err == NV_ENC_SUCCESS)
													{
														NV_ENC_PIC_PARAMS params = { 0 };

														params.version = NV_ENC_PIC_PARAMS_VER;
														params.inputWidth = width;
														params.inputHeight = height;
														params.inputPitch = lockParams.pitch;
														params.encodePicFlags = 0;
														params.frameIdx = encoder_FrameNumber;
														params.inputTimeStamp = encoder_timeStamp;
														params.inputDuration = encoder_duration;
														params.inputBuffer = nv_input_buffers[nv_input_buffer_idx];
														params.outputBitstream = nv_output_buffers[nv_input_buffer_idx];
														params.completionEvent = NULL;
														params.bufferFmt = nv_input_format;
														params.pictureStruct = NV_ENC_PIC_STRUCT_FRAME;
														params.pictureType = (use_alpha && (encoder_FrameNumber % (keyframeMaxDistanceP.value.intValue - 1) == 0)) ? NV_ENC_PIC_TYPE_IDR : NV_ENC_PIC_TYPE_P;

														nv_err = nvenc.nvEncEncodePicture(nv_encoder, &params);

														videoEncoderTime += frameRateP.value.timeValue;

														nv_input_buffer_idx++;

														assert(nv_err != NV_ENC_ERR_ENCODER_BUSY);

														if(nv_err == NV_ENC_SUCCESS)
														{
															nv_output_available = true;
														}
														else if(nv_err == NV_ENC_ERR_NEED_MORE_INPUT)
														{
															nv_output_available = false;

															nv_err = NV_ENC_SUCCESS;
														}
														else
															result = exportReturn_InternalError;
													}

													if(use_alpha && nv_err == NV_ENC_SUCCESS)
													{
														NV_ENC_PIC_PARAMS params = { 0 };

														params.version = NV_ENC_PIC_PARAMS_VER;
														params.inputWidth = width;
														params.inputHeight = height;
														params.inputPitch = alphaLockParams.pitch;
														params.encodePicFlags = 0;
														params.frameIdx = encoder_FrameNumber;
														params.inputTimeStamp = encoder_timeStamp;
														params.inputDuration = encoder_duration;
														params.inputBuffer = nv_alpha_input_buffers[nv_alpha_input_buffer_idx];
														params.outputBitstream = nv_alpha_output_buffers[nv_alpha_input_buffer_idx];
														params.completionEvent = NULL;
														params.bufferFmt = nv_alpha_input_format;
														params.pictureStruct = NV_ENC_PIC_STRUCT_FRAME;
														params.pictureType = (encoder_FrameNumber % (keyframeMaxDistanceP.value.intValue - 1) == 0) ? NV_ENC_PIC_TYPE_IDR : NV_ENC_PIC_TYPE_P;

														nv_err = nvenc.nvEncEncodePicture(nv_alpha_encoder, &params);

														nv_alpha_input_buffer_idx++;

														assert(nv_err != NV_ENC_ERR_ENCODER_BUSY);

														if(nv_err == NV_ENC_SUCCESS)
														{
															nv_alpha_output_available = true;
														}
														else if(nv_err == NV_ENC_ERR_NEED_MORE_INPUT)
														{
															nv_alpha_output_available = false;

															nv_err = NV_ENC_SUCCESS;
														}
														else
															result = exportReturn_InternalError;
													}
												}
											}
											else
											{
												assert(false); // seems we never have to do this

												if(nv_input_buffer_idx >= nv_input_buffers.size())
												{
													// flush the encoder
													NV_ENC_PIC_PARAMS params = { 0 };

													params.version = NV_ENC_PIC_PARAMS_VER;
													params.encodePicFlags = NV_ENC_PIC_FLAG_EOS;
													params.inputBuffer = NULL;
													params.outputBitstream = NULL;
													params.completionEvent = NULL;

													nv_err = nvenc.nvEncEncodePicture(nv_encoder, &params);

													if(nv_err == NV_ENC_SUCCESS)
													{
														nv_output_available = true;
													}
													else if(nv_err == NV_ENC_ERR_NEED_MORE_INPUT)
													{
														nv_output_available = false;
														assert(false);
													}
													else
														result = exportReturn_InternalError;
												}

												if(use_alpha && nv_alpha_input_buffer_idx >= nv_alpha_input_buffers.size())
												{
													NV_ENC_PIC_PARAMS params = { 0 };

													params.version = NV_ENC_PIC_PARAMS_VER;
													params.encodePicFlags = NV_ENC_PIC_FLAG_EOS;
													params.inputBuffer = NULL;
													params.outputBitstream = NULL;
													params.completionEvent = NULL;

													nv_err = nvenc.nvEncEncodePicture(nv_alpha_encoder, &params);

													if(nv_err == NV_ENC_SUCCESS)
													{
														nv_alpha_output_available = true;
													}
													else if(nv_err == NV_ENC_ERR_NEED_MORE_INPUT)
													{
														nv_alpha_output_available = false;
														assert(false);
													}
													else
														result = exportReturn_InternalError;
												}
											}
										}
									#endif // WEBM_HAVE_NVENC
										else
											assert(false);
									}
									
									
									pixSuite->Dispose(renderResult.outFrame);
								}
							}
							else
							{
								// squeeze the last bit out of the encoder
								if(video_codec == WEBM_CODEC_VP8 || video_codec == WEBM_CODEC_VP9)
								{
									vpx_codec_err_t encode_err = vpx_codec_encode(&vpx_encoder, NULL, encoder_FrameNumber, encoder_FrameDuration, 0, vpx_deadline);
									
									if(encode_err == VPX_CODEC_OK)
									{
										videoEncoderTime = LONG_LONG_MAX;
										
										vpx_encoder_iter = NULL;
									}
									else
										result = exportReturn_InternalError;
									
									
									if(use_alpha)
									{
										vpx_codec_err_t alpha_encode_err = vpx_codec_encode(&vpx_alpha_encoder, NULL, encoder_FrameNumber, encoder_FrameDuration, 0, vpx_deadline);
										
										if(alpha_encode_err == VPX_CODEC_OK)
										{
											vpx_alpha_encoder_iter = NULL;
										}
										else
											result = exportReturn_InternalError;
									}
								}
								else
								{
									assert(video_codec == WEBM_CODEC_AV1);
									
									if(av1_codec == AV1_CODEC_AOM)
									{
										aom_codec_err_t encode_err = aom_codec_encode(&aom_encoder, NULL, encoder_FrameNumber, encoder_FrameDuration, 0);

										if(encode_err == AOM_CODEC_OK)
										{
											videoEncoderTime = LONG_LONG_MAX;

											aom_encoder_iter = NULL;
										}
										else
											result = exportReturn_InternalError;


										if(use_alpha)
										{
											aom_codec_err_t alpha_encode_err = aom_codec_encode(&aom_alpha_encoder, NULL, encoder_FrameNumber, encoder_FrameDuration, 0);

											if(alpha_encode_err == AOM_CODEC_OK)
											{
												aom_alpha_encoder_iter = NULL;
											}
											else
												result = exportReturn_InternalError;
										}
									}
								#ifdef WEBM_HAVE_NVENC
									else if(av1_codec == AV1_CODEC_NVENC)
									{
										NV_ENC_PIC_PARAMS params = { 0 };

										params.version = NV_ENC_PIC_PARAMS_VER;
										params.encodePicFlags = NV_ENC_PIC_FLAG_EOS;
										params.inputBuffer = NULL;
										params.outputBitstream = NULL;
										params.completionEvent = NULL;

										nv_err = nvenc.nvEncEncodePicture(nv_encoder, &params);

										videoEncoderTime = LONG_LONG_MAX;

										if(nv_err == NV_ENC_SUCCESS)
										{
											nv_output_available = true;
										}
										else if(nv_err == NV_ENC_ERR_NEED_MORE_INPUT)
										{
											nv_output_available = false;
											assert(false);
										}
										else
											result = exportReturn_InternalError;

										if(use_alpha && nv_err == NV_ENC_SUCCESS)
										{
											nv_err = nvenc.nvEncEncodePicture(nv_alpha_encoder, &params);

											if(nv_err == NV_ENC_SUCCESS)
											{
												nv_alpha_output_available = true;
											}
											else if(nv_err == NV_ENC_ERR_NEED_MORE_INPUT)
											{
												nv_alpha_output_available = false;
												assert(false);
											}
											else
												result = exportReturn_InternalError;
										}
									}
								#endif // WEBM_HAVE_NVENC
									else
										assert(false);
								}
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
				if(audio_codec == WEBM_CODEC_OPUS)
					assert(currentAudioSample >= (endAudioSample + opus_pre_skip));
				else
					assert(op.granulepos == endAudioSample);
			}
		}
		else
			result = exportReturn_InternalError;
		


		if(exportInfoP->exportVideo)
		{
			if(video_codec == WEBM_CODEC_VP8 || video_codec == WEBM_CODEC_VP9)
			{
				if(result == malNoError)
					assert(NULL == vpx_codec_get_cx_data(&vpx_encoder, &vpx_encoder_iter) && vpx_encoder_queue.empty());
			
				vpx_codec_err_t destroy_err = vpx_codec_destroy(&vpx_encoder);
				assert(destroy_err == VPX_CODEC_OK);
				
				if(use_alpha)
				{
					if(result == malNoError)
						assert(NULL == vpx_codec_get_cx_data(&vpx_alpha_encoder, &vpx_alpha_encoder_iter) && vpx_alpha_encoder_queue.empty());
					
					vpx_codec_err_t alpha_destroy_err = vpx_codec_destroy(&vpx_alpha_encoder);
					assert(alpha_destroy_err == VPX_CODEC_OK);
				}
			}
			else
			{
				assert(video_codec == WEBM_CODEC_AV1);
				
				if(av1_codec == AV1_CODEC_AOM)
				{
					if(result == malNoError)
						assert(NULL == aom_codec_get_cx_data(&aom_encoder, &aom_encoder_iter) && aom_encoder_queue.empty());

					aom_codec_err_t destroy_err = aom_codec_destroy(&aom_encoder);
					assert(destroy_err == AOM_CODEC_OK);

					if(use_alpha)
					{
						if(result == malNoError)
							assert(NULL == aom_codec_get_cx_data(&aom_alpha_encoder, &aom_alpha_encoder_iter) && aom_alpha_encoder_queue.empty());

						aom_codec_err_t alpha_destroy_err = aom_codec_destroy(&aom_alpha_encoder);
						assert(alpha_destroy_err == AOM_CODEC_OK);
					}
				}
			#ifdef WEBM_HAVE_NVENC
				else if(av1_codec == AV1_CODEC_NVENC)
				{
					assert(nv_input_buffer_idx == 0);
					assert(!nv_output_available);
					assert(nv_output_buffer_idx == 0);
					assert(nv_encoder_queue.empty());

					for(int i=0; i < nv_input_buffers.size(); i++)
						nvenc.nvEncDestroyInputBuffer(nv_encoder, nv_input_buffers[i]);

					for(int i=0; i < nv_output_buffers.size(); i++)
						nvenc.nvEncDestroyBitstreamBuffer(nv_encoder, nv_output_buffers[i]);

					nv_err = nvenc.nvEncDestroyEncoder(nv_encoder);

					assert(nv_err == NV_ENC_SUCCESS);

					nv_encoder = NULL;

					if(use_alpha)
					{
						assert(nv_alpha_input_buffer_idx == 0);
						assert(!nv_alpha_output_available);
						assert(nv_alpha_output_buffer_idx == 0);
						assert(nv_alpha_encoder_queue.empty());

						for (int i=0; i < nv_alpha_input_buffers.size(); i++)
							nvenc.nvEncDestroyInputBuffer(nv_alpha_encoder, nv_alpha_input_buffers[i]);

						for (int i=0; i < nv_alpha_output_buffers.size(); i++)
							nvenc.nvEncDestroyBitstreamBuffer(nv_alpha_encoder, nv_alpha_output_buffers[i]);

						nv_err = nvenc.nvEncDestroyEncoder(nv_alpha_encoder);

						assert(nv_err == NV_ENC_SUCCESS);

						nv_alpha_encoder = NULL;
					}

					CUresult cuErr = cuCtxDestroy(cudaContext);

					assert(cuErr == CUDA_SUCCESS);

					cudaContext = NULL;
				}
			#endif // WEBM_HAVE_NVENC
				else
					assert(false);
			}
		}
			
		if(exportInfoP->exportAudio && !vbr_pass)
		{
			if(audio_codec == WEBM_CODEC_OPUS)
			{
				if(opus)
					opus_multistream_encoder_destroy(opus);
				
				if(opus_buffer)
					free(opus_buffer);
				
				if(opus_compressed_buffer)
					free(opus_compressed_buffer);
			}
			else
			{
				if(result == malNoError)
					assert(vorbis_analysis_blockout(&vd, &vb) == 0);
			
				vorbis_block_clear(&vb);
				vorbis_dsp_clear(&vd);
				vorbis_comment_clear(&vc);
				vorbis_info_clear(&vi);
			}
			
			for(int i=0; i < audioChannels; i++)
			{
				if(pr_audio_buffer[i] != NULL)
					free(pr_audio_buffer[i]);
			}
		}
	}

#ifdef WEBM_HAVE_NVENC
	assert(nv_encoder == NULL);
	assert(nv_alpha_encoder == NULL);
	assert(cudaContext == NULL);
#endif
	
	
	if(muxer_segment != NULL)
	{
		bool final = muxer_segment->Finalize();
		
		if(!final)
			result = exportReturn_InternalError;
	}
	
	
	}catch(...) { result = exportReturn_InternalError; }
	
	
	delete muxer_segment;
	
	delete writer;
	
	
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
		#ifdef WEBM_HAVE_NVENC
			const bool haveNVENC = (nvenc.version > 0);
		#else
			const bool haveNVENC = false;
		#endif
			result = exSDKPostProcessParams(stdParmsP,
											reinterpret_cast<exPostProcessParamsRec*>(param1),
											haveNVENC);
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
