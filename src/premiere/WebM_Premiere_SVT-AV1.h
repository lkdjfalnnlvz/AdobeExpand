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

#ifndef WEBM_PREMIERE_SVT_AV1_H
#define WEBM_PREMIERE_SVT_AV1_H

#include "WebM_Premiere_Codecs.h"

#include "EbSvtAv1Enc.h"

// from EbSequenceControlSet.h
enum {
    SVT_SINGLE_PASS, //single pass mode
    SVT_FIRST_PASS, // first pass of two pass mode
    SVT_SECOND_PASS, // Second pass of two pass mode
    SVT_MAX_ENCODE_PASS = 2,
};


class SVTAV1Encoder : public VideoEncoder
{
  public:
	SVTAV1Encoder(int width, int height, const exRatioValue &pixelAspect,
					const exRatioValue &fps,
					WebM_Video_Method method, int quality, int bitrate,
					bool twoPass, bool vbrPass, void *vbrBuffer, size_t vbrBufferSize,
					int keyframeMaxDistance, bool forceKeyframes,
					WebM_Chroma_Sampling sampling, int bitDepth,
					WebM_ColorSpace colorSpace, const std::string &custom,
					PrSDKPPixSuite *pixSuite, PrSDKPPix2Suite *pix2Suite, bool alpha);
	virtual ~SVTAV1Encoder();
	
	virtual void * getPrivateData(size_t &size);
	
	virtual bool compressFrame(const PPixHand &frame, PrTime time, PrTime duration);
	virtual bool endOfStream();

  private:
	void checkForPackets();
	void applyCustom(EbSvtAv1EncConfiguration &config, const std::string &custom);

  private:
	EbComponentType *_encoder;
	bool _eos;
	
	EbBufferHeaderType _header;
	EbSvtIOFormat _image;
	
	void *_privateData;
	size_t _privateSize;
	
	const bool _vbrPass;
	const bool _forceKeyframes;
	const int _keyframeMaxDistance;
	const WebM_Chroma_Sampling _sampling;
	const int _bitDepth;
	const WebM_ColorSpace _colorSpace;
};

#endif // WEBM_PREMIERE_SVT_AV1_H
