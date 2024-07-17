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

#include "WebM_Premiere_NVENC.h"

#include <sstream>

#include <assert.h>


#ifndef WEBM_HAVE_NVENC
#error "Looks like you shouldn't be compiling this"
#endif


#include <cuda_runtime.h>
static CUdevice cudaDevice;

static bool haveAV1 = false;
static NV_ENCODE_API_FUNCTION_LIST nvenc = { 0 };



NVENCEncoder::NVENCEncoder(int width, int height, const exRatioValue &pixelAspect,
							const exRatioValue &fps,
							WebM_Video_Method method, int quality, int bitrate,
							bool twoPass, bool vbrPass, void *vbrBuffer, size_t vbrBufferSize,
							int keyframeMaxDistance, bool forceKeyframes,
							WebM_Chroma_Sampling sampling, int bitDepth,
							WebM_ColorSpace colorSpace, const std::string &custom,
							PrSDKPPixSuite *pixSuite, PrSDKPPix2Suite* pix2Suite, bool alpha) :
	VideoEncoder(pixSuite, pix2Suite, alpha),
	_cudaContext(NULL),
	_encoder(NULL),
	_format(NV_ENC_BUFFER_FORMAT_UNDEFINED),
	_input_buffer_idx(0),
	_output_available(false),
	_output_buffer_idx(0),
	_privateData(NULL),
	_privateSize(0),
	_vbrPass(twoPass && vbrPass),
	_keyframeMaxDistance(keyframeMaxDistance),
	_forceKeyframes(forceKeyframes),
	_sampling(sampling),
	_bitDepth(bitDepth),
	_colorSpace(colorSpace)
{
	if(nvenc.version == 0)
		throw exportReturn_InternalError;

	CUresult cuErr = cuCtxCreate(&_cudaContext, CU_CTX_SCHED_AUTO, cudaDevice);

	if(cuErr != CUDA_SUCCESS)
		throw exportReturn_InternalError;


	assert(_cudaContext != NULL);

	NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS sessionParams = { 0 };

	sessionParams.version = NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS_VER;
	sessionParams.deviceType = NV_ENC_DEVICE_TYPE_CUDA;
	sessionParams.device = _cudaContext;
	sessionParams.apiVersion = NVENCAPI_VERSION;

	NVENCSTATUS err = nvenc.nvEncOpenEncodeSessionEx(&sessionParams, &_encoder);

	if(err != NV_ENC_SUCCESS || _encoder == NULL)
		throw exportReturn_InternalError;


	const GUID codecGUID = NV_ENC_CODEC_AV1_GUID;

	bool have_codec = false;

	uint32_t codec_count = 0;
	err = nvenc.nvEncGetEncodeGUIDCount(_encoder, &codec_count);

	if(err == NV_ENC_SUCCESS && codec_count > 0)
	{
		GUID *guids = new GUID[codec_count];

		uint32_t codec_count_again = 0;

		err = nvenc.nvEncGetEncodeGUIDs(_encoder, guids, codec_count, &codec_count_again);

		assert(codec_count_again == codec_count);

		for(int i=0; i < codec_count && err == NV_ENC_SUCCESS && !have_codec; i++)
		{
			if(guids[i] == codecGUID)
				have_codec = true;
		}

		delete[] guids;
	}


	const GUID profileGUID = NV_ENC_AV1_PROFILE_MAIN_GUID;

	bool have_profile = false;

	if(have_codec && err == NV_ENC_SUCCESS)
	{
		uint32_t profile_count = 0;

		err = nvenc.nvEncGetEncodeProfileGUIDCount(_encoder, codecGUID, &profile_count);

		if(err == NV_ENC_SUCCESS && profile_count > 0)
		{
			GUID *guids = new GUID[profile_count];

			uint32_t profile_count_again = 0;

			err = nvenc.nvEncGetEncodeProfileGUIDs(_encoder, codecGUID, guids, profile_count, &profile_count_again);

			assert(profile_count_again == profile_count);

			for(int i=0; i < profile_count && err == NV_ENC_SUCCESS && !have_profile; i++)
			{
				if(guids[i] == profileGUID)
					have_profile = true;
			}

			delete[] guids;
		}
	}


	_format = (sampling == WEBM_444) ? (bitDepth == 10 ? NV_ENC_BUFFER_FORMAT_YUV444_10BIT : NV_ENC_BUFFER_FORMAT_YUV444) :
										(bitDepth == 10 ? NV_ENC_BUFFER_FORMAT_YUV420_10BIT : NV_ENC_BUFFER_FORMAT_IYUV);

	bool have_input_format = false;

	if(have_profile && err == NV_ENC_SUCCESS)
	{
		uint32_t format_count = 0;

		err = nvenc.nvEncGetInputFormatCount(_encoder, codecGUID, &format_count);

		if(err == NV_ENC_SUCCESS && format_count > 0)
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

			err = nvenc.nvEncGetInputFormats(_encoder, codecGUID, formats, format_count, &format_count_again);

			assert(format_count_again == format_count);

			for(int i=0; i < format_count && err == NV_ENC_SUCCESS; i++)
			{
				const NV_ENC_BUFFER_FORMAT &format = formats[i];

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

			if(_format == NV_ENC_BUFFER_FORMAT_IYUV)
				have_input_format = have_iyuv;
			else if(_format == NV_ENC_BUFFER_FORMAT_YUV444)
				have_input_format = have_yuv444;
			else if(_format == NV_ENC_BUFFER_FORMAT_YUV420_10BIT)
				have_input_format = have_yuv420_10bit;
			else if(_format == NV_ENC_BUFFER_FORMAT_YUV444_10BIT)
				have_input_format = have_yuv444_10bit;
		}
	}


	const GUID presetGUID = NV_ENC_PRESET_P7_GUID;

	bool have_preset = false;

	if(have_input_format && err == NV_ENC_SUCCESS)
	{
		uint32_t preset_count = 0;

		err = nvenc.nvEncGetEncodePresetCount(_encoder, codecGUID, &preset_count);

		if(err == NV_ENC_SUCCESS && preset_count > 0)
		{
			GUID *guids = new GUID[preset_count];

			uint32_t preset_count_again = 0;

			err = nvenc.nvEncGetEncodeProfileGUIDs(_encoder, codecGUID, guids, preset_count, &preset_count_again);

			//assert(preset_count_again == preset_count); // ??

			for(int i=0; i < preset_count_again && err == NV_ENC_SUCCESS && !have_preset; i++)
			{
				if(guids[i] == presetGUID)
					have_preset = true;
			}

			delete[] guids;
		}
	}

	assert(!have_preset); // not sure what's going on here, using NV_ENC_PRESET_P7_GUID anyway


	bool have_capabilities = true;

	if(bitDepth == 10)
	{
		int can10bit = 0;

		NV_ENC_CAPS_PARAM caps = { 0 };
		caps.version = NV_ENC_CAPS_PARAM_VER;
		caps.capsToQuery = NV_ENC_CAPS_SUPPORT_10BIT_ENCODE;

		nvenc.nvEncGetEncodeCaps(_encoder, codecGUID, &caps, &can10bit);

		if(!can10bit)
			have_capabilities = false;
	}

	if(sampling == WEBM_444)
	{
		int can4444 = 0;

		NV_ENC_CAPS_PARAM caps = { 0 };
		caps.version = NV_ENC_CAPS_PARAM_VER;
		caps.capsToQuery = NV_ENC_CAPS_SUPPORT_YUV444_ENCODE;

		nvenc.nvEncGetEncodeCaps(_encoder, codecGUID, &caps, &can4444);

		if(!can4444)
			have_capabilities = false;
	}
	else if(sampling == WEBM_422)
	{
		have_capabilities = false;
	}


	if(err != NV_ENC_SUCCESS || !have_codec || !have_profile || !have_input_format || !have_capabilities)
	{
		nvenc.nvEncDestroyEncoder(_encoder);

		_encoder = NULL;

		cuCtxDestroy(_cudaContext);

		_cudaContext = NULL;

		throw exportReturn_InternalError;
	}


	const NV_ENC_TUNING_INFO tuningInfo = NV_ENC_TUNING_INFO_HIGH_QUALITY;

	NV_ENC_PRESET_CONFIG presetConfig = { 0 };
	presetConfig.version = NV_ENC_PRESET_CONFIG_VER;
	presetConfig.presetCfg.version = NV_ENC_CONFIG_VER;

	err = nvenc.nvEncGetEncodePresetConfigEx(_encoder, codecGUID, presetGUID, tuningInfo, &presetConfig);

	if(err != NV_ENC_SUCCESS)
		throw exportReturn_InternalError;


	NV_ENC_CONFIG &config = presetConfig.presetCfg;

	NV_ENC_RC_PARAMS &rcParams = config.rcParams;

	if(method == WEBM_METHOD_CONSTANT_QUALITY || method == WEBM_METHOD_CONSTRAINED_QUALITY)
	{
		rcParams.rateControlMode = NV_ENC_PARAMS_RC_CONSTQP;

		rcParams.constQP.qpIntra = rcParams.constQP.qpInterP = rcParams.constQP.qpInterB = (100 - quality);
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

		rcParams.averageBitRate = bitrate * 1000;
		rcParams.maxBitRate = rcParams.averageBitRate * 120 / 100;
	}

	assert(rcParams.multiPass == NV_ENC_MULTI_PASS_DISABLED);
	rcParams.multiPass = (twoPass ? NV_ENC_TWO_PASS_FULL_RESOLUTION : NV_ENC_MULTI_PASS_DISABLED); // does this actually do anything?

	NV_ENC_CONFIG_AV1 &av1config = config.encodeCodecConfig.av1Config;

	assert(av1config.chromaFormatIDC == 1); // 4:2:0, 4:4:4 currently not supported
	av1config.inputBitDepth = (bitDepth == 10 ? NV_ENC_BIT_DEPTH_10 : NV_ENC_BIT_DEPTH_8);
	av1config.outputBitDepth = av1config.inputBitDepth;
	av1config.colorPrimaries = NV_ENC_VUI_COLOR_PRIMARIES_BT709;
	av1config.transferCharacteristics = NV_ENC_VUI_TRANSFER_CHARACTERISTIC_BT709;
	av1config.matrixCoefficients = (colorSpace == WEBM_REC709 ? NV_ENC_VUI_MATRIX_COEFFS_BT709 : NV_ENC_VUI_MATRIX_COEFFS_BT470BG);
	av1config.colorRange = 0;


	applyCustom(config, custom);


	NV_ENC_INITIALIZE_PARAMS params = { 0 };

	params.version = NV_ENC_INITIALIZE_PARAMS_VER;
	params.encodeGUID = codecGUID;
	params.presetGUID = presetGUID;
	params.encodeWidth = width;
	params.encodeHeight = height;
	params.darWidth = width * pixelAspect.numerator;
	params.darHeight = height * pixelAspect.denominator;
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


	err = nvenc.nvEncInitializeEncoder(_encoder, &params);

	if (err != NV_ENC_SUCCESS)
	{
		nvenc.nvEncDestroyEncoder(_encoder);

		_encoder = NULL;

		cuCtxDestroy(_cudaContext);

		_cudaContext = NULL;

		throw exportReturn_InternalError;
	}


	const int num_buffers = std::min(64, keyframeMaxDistance * 4);

	for(int i=0; i < num_buffers && err == NV_ENC_SUCCESS; i++)
	{
		NV_ENC_CREATE_INPUT_BUFFER input_params = { 0 };

		input_params.version = NV_ENC_CREATE_INPUT_BUFFER_VER;
		input_params.width = params.encodeWidth;
		input_params.height = params.encodeHeight;
		input_params.bufferFmt = _format;
		input_params.inputBuffer = NULL;
		input_params.pSysMemBuffer = NULL;

		err = nvenc.nvEncCreateInputBuffer(_encoder, &input_params);

		if(err == NV_ENC_SUCCESS)
		{
			_input_buffers.push_back(input_params.inputBuffer);

			NV_ENC_CREATE_BITSTREAM_BUFFER output_params = { 0 };

			output_params.version = NV_ENC_CREATE_BITSTREAM_BUFFER_VER;
			output_params.reserved = 0;
			output_params.bitstreamBuffer = NULL;

			err = nvenc.nvEncCreateBitstreamBuffer(_encoder, &output_params);

			if(err == NV_ENC_SUCCESS)
			{
				_output_buffers.push_back(output_params.bitstreamBuffer);
			}
		}
	}

	assert(err == NV_ENC_SUCCESS);


	_privateData = malloc(NV_MAX_SEQ_HDR_LEN);

	if(_privateData != NULL)
	{
		uint32_t payloadSize = 0;

		NV_ENC_SEQUENCE_PARAM_PAYLOAD payload = { 0 };

		payload.version = NV_ENC_SEQUENCE_PARAM_PAYLOAD_VER;
		payload.inBufferSize = NV_MAX_SEQ_HDR_LEN;
		payload.spsId = 0;
		payload.ppsId = 0;
		payload.spsppsBuffer = _privateData;
		payload.outSPSPPSPayloadSize = &payloadSize;

		err = nvenc.nvEncGetSequenceParams(_encoder, &payload);

		if(err == NV_ENC_SUCCESS)
			_privateSize = payloadSize;
		else
			assert(false);
	}
	else
		throw exportReturn_ErrMemory;
}

NVENCEncoder::~NVENCEncoder()
{
	if(_encoder != NULL)
	{
		assert(nvenc.version != 0);

		assert(_input_buffer_idx == 0);
		assert(!_output_available);
		assert(_output_buffer_idx == 0);

		for(int i=0; i < _input_buffers.size(); i++)
			nvenc.nvEncDestroyInputBuffer(_encoder, _input_buffers[i]);

		for(int i=0; i < _output_buffers.size(); i++)
			nvenc.nvEncDestroyBitstreamBuffer(_encoder, _output_buffers[i]);

		NVENCSTATUS err = nvenc.nvEncDestroyEncoder(_encoder);

		assert(err == NV_ENC_SUCCESS);
	}

	if(_cudaContext != NULL)
	{
		CUresult err = cuCtxDestroy(_cudaContext);

		assert(err == CUDA_SUCCESS);
	}

	if(_privateData != NULL)
		free(_privateData);
}

void *
NVENCEncoder::getPrivateData(size_t &size)
{
	size = _privateSize;
	
	return _privateData;
}

bool
NVENCEncoder::compressFrame(const PPixHand &frame, PrTime time, PrTime duration)
{
	bool result = true;

	if(_input_buffer_idx >= _input_buffers.size())
	{
		assert(false); // seems we never have to do this

		// flush the encoder
		NV_ENC_PIC_PARAMS params = { 0 };

		params.version = NV_ENC_PIC_PARAMS_VER;
		params.encodePicFlags = NV_ENC_PIC_FLAG_EOS;
		params.inputBuffer = NULL;
		params.outputBitstream = NULL;
		params.completionEvent = NULL;

		NVENCSTATUS err = nvenc.nvEncEncodePicture(_encoder, &params);

		if(err == NV_ENC_SUCCESS)
		{
			_output_available = true;
		}
		else if(err == NV_ENC_ERR_NEED_MORE_INPUT)
		{
			_output_available = false;
			assert(false);
		}
		else
			assert(false);

		checkForPackets();
	}


	prRect bounds;
	_pixSuite->GetBounds(frame, &bounds);

	const csSDK_int32 width = (bounds.right - bounds.left);
	const csSDK_int32 height = (bounds.bottom - bounds.top);

	if(_input_buffer_idx < _input_buffers.size())
	{
		NV_ENC_LOCK_INPUT_BUFFER lockParams = { 0 };

		lockParams.version = NV_ENC_LOCK_INPUT_BUFFER_VER;
		lockParams.doNotWait = FALSE;
		lockParams.inputBuffer = _input_buffers[_input_buffer_idx];
		lockParams.bufferDataPtr = NULL;
		lockParams.pitch = 0;

		NV_ENC_LOCK_INPUT_BUFFER alphaLockParams = lockParams;

		NVENCSTATUS err = nvenc.nvEncLockInputBuffer(_encoder, &lockParams);

		if(err != NV_ENC_SUCCESS)
			return false;


		assert(_format != NV_ENC_BUFFER_FORMAT_NV12);

		const bool subsampled = !(_format == NV_ENC_BUFFER_FORMAT_YUV444 || _format == NV_ENC_BUFFER_FORMAT_YUV444_10BIT);

		if(_format == NV_ENC_BUFFER_FORMAT_YUV420_10BIT)
		{
			assert(subsampled);

			NV12Buffer yuv;

			yuv.width = width;
			yuv.height = height;
			yuv.sampling = _sampling;
			yuv.bitDepth = (_format == NV_ENC_BUFFER_FORMAT_YUV420_10BIT || _format == NV_ENC_BUFFER_FORMAT_YUV444_10BIT) ? 16 : 8; // for 10-bit, data is in the "high bits", so full 16-bit
			yuv.colorSpace = _colorSpace;
			yuv.fullRange = _alpha;
			yuv.yRowbytes = lockParams.pitch; // pitch == rowbytes
			yuv.uvRowbytes = (2 * lockParams.pitch / (subsampled ? 2 : 1));
			yuv.y = (uint8_t *)lockParams.bufferDataPtr;
			yuv.uv = yuv.y + (yuv.yRowbytes * yuv.height);
			yuv.uvReversed = false;

			CopyPixToBuffer(yuv, frame);
		}
		else
		{
			YUVBuffer yuv;

			yuv.width = width;
			yuv.height = height;
			yuv.sampling = _sampling;
			yuv.bitDepth = (_format == NV_ENC_BUFFER_FORMAT_YUV420_10BIT || _format == NV_ENC_BUFFER_FORMAT_YUV444_10BIT) ? 16 : 8; // for 10-bit, data is in the "high bits", so full 16-bit
			yuv.colorSpace = _colorSpace;
			yuv.fullRange = _alpha;
			yuv.yRowbytes = lockParams.pitch; // pitch == rowbytes
			yuv.uRowbytes = (lockParams.pitch / (subsampled ? 2 : 1));
			yuv.vRowbytes = yuv.uRowbytes;
			yuv.y = (uint8_t*)lockParams.bufferDataPtr;
			yuv.u = yuv.y + (height * yuv.yRowbytes);
			yuv.v = yuv.u + (height * yuv.uRowbytes / (subsampled ? 2 : 1));

			CopyPixToBuffer(yuv, frame);
		}


		err = nvenc.nvEncUnlockInputBuffer(_encoder, _input_buffers[_input_buffer_idx]);

		assert(err == NV_ENC_SUCCESS);


		NV_ENC_PIC_PARAMS params = { 0 };

		params.version = NV_ENC_PIC_PARAMS_VER;
		params.inputWidth = width;
		params.inputHeight = height;
		params.inputPitch = lockParams.pitch;
		params.encodePicFlags = 0;
		params.frameIdx = time;
		params.inputTimeStamp = time;
		params.inputDuration = duration;
		params.inputBuffer = _input_buffers[_input_buffer_idx];
		params.outputBitstream = _output_buffers[_input_buffer_idx];
		params.completionEvent = NULL;
		params.bufferFmt = _format;
		params.pictureStruct = NV_ENC_PIC_STRUCT_FRAME;
		params.pictureType = (_forceKeyframes && (time % (_keyframeMaxDistance - 1) == 0)) ? NV_ENC_PIC_TYPE_IDR : NV_ENC_PIC_TYPE_P;

		err = nvenc.nvEncEncodePicture(_encoder, &params);

		_input_buffer_idx++;

		assert(err != NV_ENC_ERR_ENCODER_BUSY);

		if(err == NV_ENC_SUCCESS)
		{
			_output_available = true;

			checkForPackets();
		}
		else if(err == NV_ENC_ERR_NEED_MORE_INPUT)
		{
			_output_available = false;

			err = NV_ENC_SUCCESS;
		}
		else
			result = false;
	}
	else
		result = false;

	return result;
}

bool
NVENCEncoder::endOfStream()
{
	bool result = true;

	NV_ENC_PIC_PARAMS params = { 0 };

	params.version = NV_ENC_PIC_PARAMS_VER;
	params.encodePicFlags = NV_ENC_PIC_FLAG_EOS;
	params.inputBuffer = NULL;
	params.outputBitstream = NULL;
	params.completionEvent = NULL;

	NVENCSTATUS err = nvenc.nvEncEncodePicture(_encoder, &params);

	if(err == NV_ENC_SUCCESS)
	{
		_output_available = true;

		checkForPackets();
	}
	else if(err == NV_ENC_ERR_NEED_MORE_INPUT)
	{
		_output_available = false;
		assert(false);
	}
	else
		result = false;

	return result;
}

void
NVENCEncoder::checkForPackets()
{
	while(_output_available)
	{
		assert(_output_buffer_idx < _input_buffer_idx);

		if(_vbrPass)
		{
			assert(false);
		}
		else
		{
			NV_ENC_LOCK_BITSTREAM lock = { 0 };

			lock.version = NV_ENC_LOCK_BITSTREAM_VER;
			lock.outputBitstream = _output_buffers[_output_buffer_idx];

			NVENCSTATUS err = nvenc.nvEncLockBitstream(_encoder, &lock);

			if(err == NV_ENC_SUCCESS)
			{
				Packet* packet = new Packet;

				packet->size = lock.bitstreamSizeInBytes;
				packet->data = malloc(packet->size);
				if (packet->data == NULL)
					throw exportReturn_ErrMemory;
				memcpy(packet->data, lock.bitstreamBufferPtr, packet->size);
				packet->time = lock.outputTimeStamp;
				packet->duration = lock.outputDuration;
				packet->keyframe = (lock.pictureType == NV_ENC_PIC_TYPE_IDR);

				assert(lock.pictureStruct == NV_ENC_PIC_STRUCT_FRAME);
				assert(lock.pictureType != NV_ENC_PIC_TYPE_I); // not I, IDR!

				_queue.push(packet);


				err = nvenc.nvEncUnlockBitstream(_encoder, _output_buffers[_output_buffer_idx]);

				assert(err == NV_ENC_SUCCESS);

				_output_buffer_idx++;

				if(_output_buffer_idx == _input_buffer_idx)
				{
					_output_buffer_idx = _input_buffer_idx = 0;

					_output_available = false;
				}
			}
			else if(err == NV_ENC_ERR_INVALID_PARAM)
			{
				// Huh? I guess the next buffer isn't ready?

				_output_available = false;
			}
			else
				assert(false);
		}
	}
}

template <typename T>
static void SetValue(T &v, const std::string &s)
{
	std::stringstream ss;
	
	ss << s;
	
	ss >> v;
}

void
NVENCEncoder::applyCustom(NV_ENC_CONFIG &config, const std::string &custom)
{
	std::vector<std::string> args;
	
	if(quotedTokenize(custom, args, " =\t\r\n") && args.size() > 0)
	{
		const int num_args = args.size();
	
		args.push_back(""); // so there's always an i+1
		
		int i = 0;
		
		while(i < num_args)
		{
			const std::string &arg = args[i];
			const std::string &val = args[i + 1];
			
			if(arg == "--gopLength")
			{	SetValue(config.gopLength, val); i++;	}

			else if(arg == "--frameIntervalP")
			{	SetValue(config.frameIntervalP, val); i++;	}

			else if(arg == "--averageBitRate")
			{	SetValue(config.rcParams.averageBitRate, val); i++;	}

			else if(arg == "--maxBitRate")
			{	SetValue(config.rcParams.maxBitRate, val); i++;	}

			else if(arg == "--constQP")
			{
				int tmp;
				SetValue(tmp, val);
				config.rcParams.constQP.qpIntra = config.rcParams.constQP.qpInterP = config.rcParams.constQP.qpInterB = tmp;
				i++;
			}

			else if(arg == "--minQP")
			{
				int tmp;
				SetValue(tmp, val);
				config.rcParams.minQP.qpIntra = config.rcParams.minQP.qpInterP = config.rcParams.minQP.qpInterB = tmp;
				config.rcParams.enableMinQP = 1;
				i++;
			}

			else if(arg == "--maxQP")
			{
				int tmp;
				SetValue(tmp, val);
				config.rcParams.maxQP.qpIntra = config.rcParams.maxQP.qpInterP = config.rcParams.maxQP.qpInterB = tmp;
				config.rcParams.enableMaxQP = 1;
				i++;
			}

			else if(arg == "--initialRCQP")
			{
				int tmp;
				SetValue(tmp, val);
				config.rcParams.initialRCQP.qpIntra = config.rcParams.initialRCQP.qpInterP = config.rcParams.initialRCQP.qpInterB = tmp;
				config.rcParams.enableInitialRCQP = 1;
				i++;
			}

			else if(arg == "--enableAQ")
			{	config.rcParams.enableAQ = 1;	}

			else if(arg == "--strictGOPTarget")
			{	config.rcParams.strictGOPTarget = 1;	}

			else if(arg == "--aqStrength")
			{
				unsigned int tmp;
				SetValue(tmp, val);
				config.rcParams.aqStrength = tmp; i++;
			}

			else if(arg == "--targetQuality")
			{	SetValue(config.rcParams.targetQuality, val); i++;	}

			else if(arg == "--outputAnnexBFormat")
			{	config.encodeCodecConfig.av1Config.outputAnnexBFormat = 1;	}

			i++;
		}
	}
}

void
NVENCEncoder::initialize()
{
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

	if(nvenc.version != 0)
	{
		CUcontext cudaContext = NULL;

		CUresult cuErr = cuCtxCreate(&cudaContext, CU_CTX_SCHED_AUTO, cudaDevice);

		if(cuErr == CUDA_SUCCESS)
		{
			assert(cudaContext != NULL);

			void *encoder = NULL;

			NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS sessionParams = { 0 };

			sessionParams.version = NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS_VER;
			sessionParams.deviceType = NV_ENC_DEVICE_TYPE_CUDA;
			sessionParams.device = cudaContext;
			sessionParams.apiVersion = NVENCAPI_VERSION;

			NVENCSTATUS err = nvenc.nvEncOpenEncodeSessionEx(&sessionParams, &encoder);

			if(err == NV_ENC_SUCCESS && encoder != NULL)
			{
				const GUID codecGUID = NV_ENC_CODEC_AV1_GUID;

				uint32_t codec_count = 0;
				err = nvenc.nvEncGetEncodeGUIDCount(encoder, &codec_count);

				if(err == NV_ENC_SUCCESS && codec_count > 0)
				{
					GUID *guids = new GUID[codec_count];

					uint32_t codec_count_again = 0;

					err = nvenc.nvEncGetEncodeGUIDs(encoder, guids, codec_count, &codec_count_again);

					assert(codec_count_again == codec_count);

					for(int i=0; i < codec_count && err == NV_ENC_SUCCESS; i++)
					{
						if(guids[i] == codecGUID)
							haveAV1 = true;
					}

					delete[] guids;
				}

				nvenc.nvEncDestroyEncoder(encoder);
			}

			cuCtxDestroy(cudaContext);
		}
	}
}

bool
NVENCEncoder::available()
{
	return haveAV1;
}
