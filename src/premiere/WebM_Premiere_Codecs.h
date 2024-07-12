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

#ifndef WEBM_PREMIERE_CODECS_H
#define WEBM_PREMIERE_CODECS_H

#include "PrSDKPPix2Suite.h"
#include "PrSDKExportParamSuite.h"

#include "WebM_Premiere_Export_Params.h"

#include <queue>
#include <vector>
#include <string>


class VideoEncoder
{
  public:
	VideoEncoder(PrSDKPPixSuite *pixSuite, PrSDKPPix2Suite *pix2Suite, bool alpha);
	virtual ~VideoEncoder();
	
	virtual void * getPrivateData(size_t &size) = 0;
	
	virtual bool compressFrame(const PPixHand &frame, PrTime time, PrTime duration) = 0;
	virtual bool endOfStream() = 0;
	
	typedef struct Packet
	{
		void *data;
		size_t size;
		PrTime time;
		PrTime duration;
		bool keyframe;
	} Packet;
	
	bool packetReady() const { return !_queue.empty(); }
	const Packet * getPacket();
	void returnPacket(const Packet *packet);

  protected:
	PrSDKPPixSuite *_pixSuite;
	PrSDKPPix2Suite *_pix2Suite;
	const bool _alpha;
	
	std::queue<Packet *> _queue;
	
  protected:
	typedef struct YUVBuffer
	{
		uint32_t width;
		uint32_t height;
	
		WebM_Chroma_Sampling sampling;
		uint8_t bitDepth;
		WebM_ColorSpace colorSpace;
		bool fullRange;
		
		uint8_t *y;
		uint8_t *u;
		uint8_t *v;
		
		ptrdiff_t yRowbytes;
		ptrdiff_t uRowbytes;
		ptrdiff_t vRowbytes;
	} YUVBuffer;
	
	void CopyPixToBuffer(const YUVBuffer &buf, const PPixHand &pix);
	
	template <typename VUYA_PIX, typename BUF_PIX>
	void CopyVUYAToBuffer(const YUVBuffer &buf, const uint8_t *frameBufferP, ptrdiff_t rowbytes);
	
	template <typename BGRA_PIX, typename BUF_PIX, bool isARGB>
	void CopyBGRAToBuffer(const YUVBuffer &buf, const uint8_t *frameBufferP, ptrdiff_t rowbytes);
	
	static bool quotedTokenize(const std::string &str, std::vector<std::string> &tokens, const std::string &delimiters);

  public:
	static void initialize();
	
	static bool haveCodec(AV1_Codec av1Codec);

	static bool twoPassCapable(WebM_Video_Codec codec, AV1_Codec av1Codec, WebM_Video_Method method, WebM_Chroma_Sampling sampling, int bitDepth, uint32_t width, uint32_t height, bool alpha);
	
	static VideoEncoder * makeEncoder(int width, int height, const exRatioValue &pixelAspect,
										const exRatioValue &fps,
										WebM_Video_Codec codec, AV1_Codec av1Codec,
										WebM_Video_Method method, int quality, int bitrate,
										bool twoPass, bool vbrPass, void *vbrBuffer, size_t vbrBufferSize,
										int keyframeMaxDistance, bool forceKeyframes,
										WebM_Chroma_Sampling sampling, int bitDepth,
										WebM_ColorSpace colorSpace, const std::string &custom,
										PrSDKPPixSuite *pixSuite, PrSDKPPix2Suite *pix2Suite, bool alpha);
};


class AudioEncoder
{
  public:
	AudioEncoder() {}
	virtual ~AudioEncoder() {}
	
	virtual void * getPrivateData(size_t &size) = 0;
	virtual uint64_t getSeekPreRoll() = 0; // in nanoseconds
	virtual uint64_t getCodecDelay() = 0; // in nanoseconds
	
	virtual bool compressSamples(float *sampleBuffers[], csSDK_uint32 samples, csSDK_uint32 discardSamples) = 0;
	virtual bool endOfStream() = 0;
	
	typedef struct Packet
	{
		void *data;
		size_t size;
		PrAudioSample sampleIndex;
		csSDK_uint32 discardSamples;
	} Packet;
	
	const Packet * getPacket();
	void returnPacket(const Packet *packet);

  protected:
	std::queue<Packet *> _queue;

  public:
	static PrAudioChannelLabel * channelOrder(WebM_Audio_Codec codec, PrAudioChannelType channelType);
  
	static AudioEncoder * makeEncoder(int channels, float sampleRate,
										WebM_Audio_Codec codec,
										Ogg_Method method, float quality, int bitrate, bool autoBitrate);
};

#endif // WEBM_PREMIERE_CODECS_H
