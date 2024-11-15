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

#include "WebM_Premiere_VPL.h"

#ifndef WEBM_HAVE_VPL
#error "Looks like you shouldn't be compiling this"
#endif

#ifndef VPLVERSION
#define VPLVERSION(major, minor) (major << 16 | minor)
#endif

#define MAJOR_API_VERSION_REQUIRED 2
#define MINOR_API_VERSION_REQUIRED 2

#ifndef ALIGN16
#define ALIGN16(value)           (((value + 15) >> 4) << 4)
#endif


#include <sstream>

#include <assert.h>


IntelVPLEncoder::IntelVPLEncoder(int width, int height, const exRatioValue &pixelAspect,
									const exRatioValue &fps,
									Codec codec,
									WebM_Video_Method method, int quality, int bitrate,
									bool twoPass, bool vbrPass, void *vbrBuffer, size_t vbrBufferSize,
									int keyframeMaxDistance, bool forceKeyframes,
									WebM_Chroma_Sampling sampling, int bitDepth,
									WebM_ColorSpace colorSpace, const std::string &custom,
									PrSDKPPixSuite *pixSuite, PrSDKPPix2Suite *pix2Suite, bool alpha) :
	VideoEncoder(pixSuite, pix2Suite, alpha),
	_loader(NULL),
	_session(NULL),
	_privateData(NULL),
	_privateSize(0),
	_frameFormat(0),
	_outputBuffer({}),
	_vbrPass(twoPass && vbrPass),
	_keyframeMaxDistance(keyframeMaxDistance),
	_forceKeyframes(forceKeyframes),
	_sampling(sampling),
	_bitDepth(bitDepth),
	_colorSpace(colorSpace)
{
	if (twoPass)
		throw exportReturn_InternalError;

	_loader = MFXLoad();

	if(_loader == NULL)
		throw exportReturn_InternalError;
	
	const mfxU32 codecCode = (codec == AV1 ? MFX_CODEC_AV1 : MFX_CODEC_VP9);

	configU32("mfxImplDescription.Impl", MFX_IMPL_TYPE_HARDWARE);
	configU32("mfxImplDescription.mfxEncoderDescription.encoder.CodecID", codecCode);
	configU32("mfxImplDescription.ApiVersion.Version", VPLVERSION(MAJOR_API_VERSION_REQUIRED, MINOR_API_VERSION_REQUIRED));

	mfxStatus err = MFXCreateSession(_loader, 0, &_session);

	if(err != MFX_ERR_NONE)
		throw exportReturn_InternalError;


	if(sampling == WEBM_422 || bitDepth > 10)
		throw exportReturn_InternalError;

	_frameFormat = (sampling == WEBM_444 ? (bitDepth > 8 ? MFX_FOURCC_Y410 : MFX_FOURCC_AYUV) :
											(bitDepth > 8 ? MFX_FOURCC_P010 : MFX_FOURCC_NV12));

	mfxU16 chromaFormat = (sampling == WEBM_444 ? MFX_CHROMAFORMAT_YUV444 : MFX_CHROMAFORMAT_YUV420);


	mfxVideoParam encodeParams = {};

	encodeParams.mfx.CodecId = codecCode;
	encodeParams.mfx.TargetUsage = MFX_TARGETUSAGE_BEST_QUALITY;

	encodeParams.mfx.GopRefDist = keyframeMaxDistance;
	encodeParams.mfx.EncodedOrder = forceKeyframes;

	if(method == WEBM_METHOD_CONSTANT_QUALITY || method == WEBM_METHOD_CONSTRAINED_QUALITY)
	{
		encodeParams.mfx.RateControlMethod = MFX_RATECONTROL_CQP;

		const int min_q = 1;
		const int max_q = 255;

		// our 0...100 slider will be used to bring max_q down to min_q
		encodeParams.mfx.QPI = encodeParams.mfx.QPP = encodeParams.mfx.QPB = min_q + ((((float)(100 - quality) / 100.f) * (max_q - min_q)) + 0.5f);
	}
	else
	{
		if(method == WEBM_METHOD_VBR)
		{
			encodeParams.mfx.RateControlMethod = MFX_RATECONTROL_VBR;
		}
		else if(method == WEBM_METHOD_BITRATE)
		{
			encodeParams.mfx.RateControlMethod = MFX_RATECONTROL_CBR;
		}
		else
			assert(false);

		encodeParams.mfx.TargetKbps = bitrate;
		encodeParams.mfx.MaxKbps = bitrate * 120 / 100;
	}


	if(codec == AV1)
	{
		encodeParams.mfx.CodecProfile = (bitDepth == 12 || sampling == WEBM_422) ? MFX_PROFILE_AV1_PRO :
											sampling == WEBM_444 ? MFX_PROFILE_AV1_HIGH : MFX_PROFILE_AV1_MAIN;
	}
	else
	{
		assert(codec == VP9);

		encodeParams.mfx.CodecProfile = (sampling > WEBM_420 ?
											(bitDepth > 8 ? MFX_PROFILE_VP9_3 : MFX_PROFILE_VP9_1) :
											(bitDepth > 8 ? MFX_PROFILE_VP9_2 : MFX_PROFILE_VP9_0));
	}


	encodeParams.mfx.FrameInfo.FrameRateExtN = fps.numerator;
	encodeParams.mfx.FrameInfo.FrameRateExtD = fps.denominator;
	encodeParams.mfx.FrameInfo.FourCC = _frameFormat;
	encodeParams.mfx.FrameInfo.ChromaFormat = chromaFormat;
	encodeParams.mfx.FrameInfo.CropW = width;
	encodeParams.mfx.FrameInfo.CropH = height;
	encodeParams.mfx.FrameInfo.Width = ALIGN16(width);
	encodeParams.mfx.FrameInfo.Height = ALIGN16(height);
	encodeParams.mfx.FrameInfo.BitDepthChroma = encodeParams.mfx.FrameInfo.BitDepthLuma = bitDepth;

	const size_t bufferSize = (width * height * 4 * (bitDepth > 8 ? 2 : 1));
	encodeParams.mfx.BufferSizeInKB = bufferSize / 1024;

	encodeParams.IOPattern = MFX_IOPATTERN_IN_SYSTEM_MEMORY;

	mfxExtVP9Param vp9params = {};
	//mfxExtAV1BitstreamParam av1BitstreamParam = {}; // not needed because WriteIVFHeaders defaults to off for AV1

	mfxExtBuffer *extBuffers[1];

	if(codec == VP9)
	{
		vp9params.Header.BufferId = MFX_EXTBUFF_VP9_PARAM;
		vp9params.Header.BufferSz = sizeof(mfxExtVP9Param);
		vp9params.FrameWidth = width;
		vp9params.FrameHeight = height;
		vp9params.WriteIVFHeaders = MFX_CODINGOPTION_OFF; // Really threw me for a loop when I saw "DKIF" signatures at the beginning of packets

		extBuffers[0] = &vp9params.Header;

		encodeParams.NumExtParam = 1;
		encodeParams.ExtParam = extBuffers;
	}

	applyCustom(encodeParams, custom);

	err = MFXVideoENCODE_Query(_session, &encodeParams, &encodeParams);

	if(err == MFX_WRN_INCOMPATIBLE_VIDEO_PARAM)
	{
		if(_frameFormat != encodeParams.mfx.FrameInfo.FourCC)
		{
			assert(false);
			_frameFormat = encodeParams.mfx.FrameInfo.FourCC;
		}

		assert(encodeParams.mfx.FrameInfo.ChromaFormat == chromaFormat);
		assert(encodeParams.mfx.BufferSizeInKB == 0); // doesn't like me using this
	}
	else if(err == MFX_WRN_PARTIAL_ACCELERATION)
	{
		assert(false);
	}
	else if(err != MFX_ERR_NONE)
		throw exportReturn_InternalError;
	
	err = MFXVideoENCODE_Init(_session, &encodeParams);

	if(err != MFX_ERR_NONE)
		throw exportReturn_InternalError;


	mfxExtCodingOptionSPSPPS extSPSPPS = {};

	extSPSPPS.Header.BufferId = MFX_EXTBUFF_CODING_OPTION_SPSPPS;
	extSPSPPS.Header.BufferSz = sizeof(mfxExtCodingOptionSPSPPS);

	mfxU8 spsData[1024] = {};
	extSPSPPS.SPSBuffer = spsData;
	extSPSPPS.SPSBufSize = sizeof(spsData);

	mfxU8 ppsData[1024] = {};
	extSPSPPS.PPSBuffer = ppsData;
	extSPSPPS.PPSBufSize = sizeof(ppsData);

	mfxExtBuffer *extBuf = (mfxExtBuffer *)&extSPSPPS;

	mfxVideoParam privateDataParam = {};

	privateDataParam.NumExtParam = 1;
	privateDataParam.ExtParam = &extBuf;

	err = MFXVideoENCODE_GetVideoParam(_session, &privateDataParam);

	if(err == MFX_ERR_NONE)
	{
		_privateSize = extSPSPPS.SPSBufSize;

		_privateData = malloc(_privateSize);

		if(_privateData != NULL)
			memcpy(_privateData, extSPSPPS.SPSBuffer, _privateSize);
		else
			throw exportReturn_ErrMemory;
	}
	else
		assert(err == MFX_ERR_UNSUPPORTED && codec == VP9); // unsupported in VP9


	_outputBuffer.MaxLength = (encodeParams.mfx.BufferSizeInKB > 0 ? encodeParams.mfx.BufferSizeInKB * 1024 : bufferSize);
	_outputBuffer.Data = (mfxU8 *)malloc(_outputBuffer.MaxLength);

	if(_outputBuffer.Data == NULL)
		throw exportReturn_ErrMemory;
}

IntelVPLEncoder::~IntelVPLEncoder()
{
	if(_session != NULL)
	{
		MFXVideoENCODE_Close(_session);

		MFXClose(_session);
	}

	if(_outputBuffer.Data != NULL)
		free(_outputBuffer.Data);

	if(_loader != NULL)
		MFXUnload(_loader);

	if (_privateData != NULL)
		free(_privateData);
}

void *
IntelVPLEncoder::getPrivateData(size_t &size)
{
	size = _privateSize;
	
	return _privateData;
}

bool
IntelVPLEncoder::compressFrame(const PPixHand &frame, PrTime time, PrTime duration)
{
	prRect bounds;
	_pixSuite->GetBounds(frame, &bounds);

	const csSDK_int32 width = (bounds.right - bounds.left);
	const csSDK_int32 height = (bounds.bottom - bounds.top);

	mfxFrameSurface1 *surface = NULL;

	mfxStatus err = MFXMemory_GetSurfaceForEncode(_session, &surface);

	if(err != MFX_ERR_NONE || surface == NULL)
		return false;

	err = surface->FrameInterface->Map(surface, MFX_MAP_WRITE);

	if(err != MFX_ERR_NONE)
		return false;

	assert(surface->Info.CropW == width);
	assert(surface->Info.CropH == height);
	assert(surface->Info.FourCC == _frameFormat);
	assert(surface->Info.BitDepthLuma == _bitDepth);

	if(surface->Info.FourCC == MFX_FOURCC_NV12 || surface->Info.FourCC == MFX_FOURCC_P010)
	{
		assert((surface->Info.FourCC == MFX_FOURCC_NV12 && _bitDepth == 8) ||
				(surface->Info.FourCC == MFX_FOURCC_P010 && _bitDepth == 10));
		assert(_sampling == WEBM_420);

		NV12Buffer yuv;

		yuv.width = width;
		yuv.height = height;
		yuv.sampling = _sampling;
		yuv.bitDepth = (_bitDepth > 8 ? 16 : 8);
		yuv.colorSpace = _colorSpace;
		yuv.fullRange = _alpha;
		yuv.y = surface->Data.Y;
		yuv.uv = surface->Data.UV;
		yuv.yRowbytes = surface->Data.Pitch;
		yuv.uvRowbytes = surface->Data.Pitch;
		yuv.uvReversed = false;

		CopyPixToBuffer(yuv, frame);
	}
	else
	{
		assert(surface->Info.FourCC == MFX_FOURCC_AYUV || surface->Info.FourCC == MFX_FOURCC_Y410);
		assert((surface->Info.FourCC == MFX_FOURCC_AYUV && _bitDepth == 8) ||
				(surface->Info.FourCC == MFX_FOURCC_Y410 && _bitDepth == 10));
		assert(_sampling == WEBM_444);

		const size_t planarRowbytes = (width * (_bitDepth > 8 ? 2 : 1));

		uint8_t *planarYUV = (uint8_t*)malloc(height * 3 * planarRowbytes);

		if (planarYUV == NULL)
			throw exportReturn_ErrMemory;
		
		YUVBuffer yuv;

		yuv.width = width;
		yuv.height = height;
		yuv.sampling = _sampling;
		yuv.bitDepth = _bitDepth;
		yuv.colorSpace = _colorSpace;
		yuv.fullRange = _alpha;
		yuv.y = planarYUV;
		yuv.u = (yuv.y + (height * planarRowbytes));
		yuv.v = (yuv.u + (height * planarRowbytes));
		yuv.yRowbytes = planarRowbytes;
		yuv.uRowbytes = planarRowbytes;
		yuv.vRowbytes = planarRowbytes;

		CopyPixToBuffer(yuv, frame);

		if(surface->Info.FourCC == MFX_FOURCC_AYUV)
		{
			for(int y = 0; y < height; y++)
			{
				uint8_t *ayuvPix = surface->Data.Y + (y * surface->Data.Pitch);
				uint8_t *yPix = yuv.y + (y * yuv.yRowbytes);
				uint8_t *uPix = yuv.u + (y * yuv.uRowbytes);
				uint8_t *vPix = yuv.v + (y * yuv.vRowbytes);

				for (int x = 0; x < width; x++)
				{
					*ayuvPix++ = *yPix++;
					*ayuvPix++ = 255;
					*ayuvPix++ = *vPix++;
					*ayuvPix++ = *uPix++;
				}
			}
		}
		else
		{
			assert(surface->Info.FourCC == MFX_FOURCC_Y410);
			assert(_bitDepth == 10);

			for(int y=0; y < height; y++)
			{
				mfxY410 *yuvPix = surface->Data.Y410 + (y * surface->Data.Pitch / sizeof(mfxY410));
				uint16_t *yPix = (uint16_t *)(yuv.y + (y * yuv.yRowbytes));
				uint16_t *uPix = (uint16_t *)(yuv.u + (y * yuv.uRowbytes));
				uint16_t *vPix = (uint16_t *)(yuv.v + (y * yuv.vRowbytes));

				for(int x=0; x < width; x++)
				{
					yuvPix->Y = *yPix++;
					yuvPix->U = *uPix++;
					yuvPix->V = *vPix++;
					yuvPix->A = 4;

					yuvPix++;
				}
			}
		}
	}

	err = surface->FrameInterface->Unmap(surface);

	assert(err == MFX_ERR_NONE);

	surface->Data.TimeStamp = time;


	mfxEncodeCtrl ctrl = {};
	ctrl.FrameType = (time % _keyframeMaxDistance == 0 ? MFX_FRAMETYPE_I : MFX_FRAMETYPE_P);

	mfxSyncPoint syncp = NULL;

	err = MFXVideoENCODE_EncodeFrameAsync(_session, (_forceKeyframes ? &ctrl : NULL), surface, &_outputBuffer, &syncp);

	if(err == MFX_ERR_NONE)
	{
		if(syncp != NULL)
		{
			do{
				err = MFXVideoCORE_SyncOperation(_session, syncp, 100);

				if(err == MFX_ERR_NONE)
				{
					Packet *packet = new Packet;

					packet->size = _outputBuffer.DataLength;
					packet->data = malloc(packet->size);
					if(packet->data == NULL)
						throw exportReturn_ErrMemory;
					assert(_outputBuffer.DataOffset == 0);
					memcpy(packet->data, _outputBuffer.Data + _outputBuffer.DataOffset, packet->size);
					_outputBuffer.DataLength = 0;
					packet->time = _outputBuffer.TimeStamp;
					packet->duration = 1;
					packet->keyframe = (_outputBuffer.FrameType & MFX_FRAMETYPE_I);
					assert(_outputBuffer.PicStruct == MFX_PICSTRUCT_PROGRESSIVE);

					_queue.push(packet);
				}

			}while(err == MFX_WRN_IN_EXECUTION);
		}
	}
	else if(err != MFX_ERR_MORE_DATA)
		return false;
				
	return true;
}

bool
IntelVPLEncoder::endOfStream()
{
	mfxSyncPoint syncp = NULL;

	mfxStatus err = MFXVideoENCODE_EncodeFrameAsync(_session, NULL, NULL, &_outputBuffer, &syncp);

	if(err == MFX_ERR_NONE)
	{
		if(syncp != NULL)
		{
			do{
				err = MFXVideoCORE_SyncOperation(_session, syncp, 100);

				if(err == MFX_ERR_NONE)
				{
					Packet *packet = new Packet;

					packet->size = _outputBuffer.DataLength;
					packet->data = malloc(packet->size);
					if(packet->data == NULL)
						throw exportReturn_ErrMemory;
					assert(_outputBuffer.DataOffset == 0);
					memcpy(packet->data, _outputBuffer.Data + _outputBuffer.DataOffset, packet->size);
					_outputBuffer.DataLength = 0;
					packet->time = _outputBuffer.TimeStamp;
					packet->duration = 1;
					packet->keyframe = (_outputBuffer.FrameType & MFX_FRAMETYPE_I);
					assert(_outputBuffer.PicStruct == MFX_PICSTRUCT_PROGRESSIVE);

					_queue.push(packet);
				}

			}while(err == MFX_WRN_IN_EXECUTION);
		}
	}
	else if (err != MFX_ERR_MORE_DATA)
		return false;

	return true;
}

void
IntelVPLEncoder::configU32(const char *prop, mfxU32 val)
{
	mfxConfig config = MFXCreateConfig(_loader);

	if(config == NULL)
		throw exportReturn_InternalError;
	
	mfxVariant variant;
	variant.Type = MFX_VARIANT_TYPE_U32;
	variant.Data.U32 = val;

	mfxStatus err = MFXSetConfigFilterProperty(config, (mfxU8 *)prop, variant);

	if(err != MFX_ERR_NONE)
		throw exportReturn_InternalError;
}

template <typename T>
static void SetValue(T &v, const std::string &s)
{
	std::stringstream ss;

	ss << s;

	ss >> v;
}

void
IntelVPLEncoder::applyCustom(mfxVideoParam &param, const std::string &custom)
{
	std::vector<std::string> args;

	if(quotedTokenize(custom, args, " =\t\r\n") && args.size() > 0)
	{
		const int num_args = args.size();

		args.push_back(""); // so there's always an i+1

		int i = 0;

		while(i < num_args)
		{
			const std::string& arg = args[i];
			const std::string& val = args[i + 1];

			if(arg == "--TargetUsage")
			{	SetValue(param.mfx.TargetUsage, val); i++;	}

			else if (arg == "--TargetKbps")
			{	SetValue(param.mfx.TargetKbps, val); i++;	}

			else if (arg == "--MaxKbps")
			{	SetValue(param.mfx.MaxKbps, val); i++;	}

			else if (arg == "--QPI")
			{	SetValue(param.mfx.QPI, val); i++;	}

			else if (arg == "--QPP")
			{	SetValue(param.mfx.QPP, val); i++;	}

			else if (arg == "--QPB")
			{	SetValue(param.mfx.QPB, val); i++;	}
		}
	}
}

static bool hasVP9 = false;
static bool hasAV1 = false;

void
IntelVPLEncoder::initialize()
{
	mfxLoader loader = MFXLoad();

	if(loader != NULL)
	{
		int i = 0;

		mfxHDL descH = NULL;

		while(MFX_ERR_NONE == MFXEnumImplementations(loader, i++, MFX_IMPLCAPS_IMPLDESCSTRUCTURE, &descH) && descH != NULL)
		{
			mfxImplDescription *desc = (mfxImplDescription *)descH;

			assert(desc->Impl == MFX_IMPL_TYPE_HARDWARE);

			if(desc->ApiVersion.Version >= VPLVERSION(MAJOR_API_VERSION_REQUIRED, MINOR_API_VERSION_REQUIRED))
			{
				for(int j=0; j < desc->Enc.NumCodecs; j++)
				{
					bool haveNV12 = false;
					bool haveP010 = false;
					bool haveAYUV = false;
					bool haveY410 = false;
					
					for(int a=0; a < desc->Enc.Codecs[j].NumProfiles; a++)
					{
						for(int b=0; b < desc->Enc.Codecs[j].Profiles[a].NumMemTypes; b++)
						{
							for(int c=0; c < desc->Enc.Codecs[j].Profiles[a].MemDesc[b].NumColorFormats; c++)
							{
								const mfxU32 &colorFormat = desc->Enc.Codecs[j].Profiles[a].MemDesc[b].ColorFormats[c];

								if(colorFormat == MFX_FOURCC_NV12)
									haveNV12 = true;
								else if(colorFormat == MFX_FOURCC_P010)
									haveP010 = true;
								else if(colorFormat == MFX_FOURCC_AYUV)
									haveAYUV = true;
								else if(colorFormat == MFX_FOURCC_Y410)
									haveY410 = true;
							}
						}
					}

					if(haveNV12 && haveP010 && haveAYUV && haveY410)
					{
						const mfxU32 &codecID = desc->Enc.Codecs[j].CodecID;

						if(codecID == MFX_CODEC_VP9)
							hasVP9 = true;
						else if(codecID == MFX_CODEC_AV1)
							hasAV1 = true;
					}
				}
			}

			mfxStatus err = MFXDispReleaseImplDescription(loader, descH);

			assert(err == MFX_ERR_NONE);
		}

		MFXUnload(loader);
	}
}

bool
IntelVPLEncoder::available(Codec codec)
{
	if(codec == VP9)
		return hasVP9;
	else
		return hasAV1;
}