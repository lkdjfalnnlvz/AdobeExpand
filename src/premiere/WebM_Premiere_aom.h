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

#ifndef WEBM_PREMIERE_AOM_H
#define WEBM_PREMIERE_AOM_H

#include "WebM_Premiere_Codecs.h"

#include "aom/aom_codec.h"
#include "aom/aom_encoder.h"
#include "aom/aomcx.h"


class AOMEncoder : public VideoEncoder
{
  public:
	AOMEncoder(int width, int height, const exRatioValue &pixelAspect,
					const exRatioValue &fps,
					WebM_Video_Method method, int quality, int bitrate,
					bool twoPass, bool vbrPass, void *vbrBuffer, size_t vbrBufferSize,
					int keyframeMaxDistance, bool forceKeyframes,
					WebM_Chroma_Sampling sampling, int bitDepth,
					WebM_ColorSpace colorSpace, const std::string &custom,
					PrSDKPPixSuite *pixSuite, PrSDKPPix2Suite *pix2Suite, bool alpha);
	virtual ~AOMEncoder();
	
	virtual void * getPrivateData(size_t &size);
	
	virtual bool compressFrame(const PPixHand &frame, PrTime time, PrTime duration);
	virtual bool endOfStream();

  private:
	void checkForPackets();
	void applyCustomPre(aom_codec_enc_cfg_t &config, const std::string &custom);
	void applyCustomPost(const std::string &custom);

  private:
	aom_codec_ctx_t _encoder;
	aom_codec_iter_t _iter;
	
	void *_privateData;
	size_t _privateSize;
	
	const bool _vbrPass;
	const WebM_Chroma_Sampling _sampling;
	const int _bitDepth;
	const WebM_ColorSpace _colorSpace;
};

#endif // WEBM_PREMIERE_AOM_H
