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

#ifndef WEBM_PREMIERE_OPUS_H
#define WEBM_PREMIERE_OPUS_H

#include "WebM_Premiere_Codecs.h"

#include "opus_multistream.h"


class OpusEncoder : public AudioEncoder
{
  public:
	OpusEncoder(int channels, float sampleRate, Ogg_Method method, float quality, int bitrate, bool autoBitrate);
	virtual ~OpusEncoder();

	virtual void * getPrivateData(size_t &size);
	virtual uint64_t getSeekPreRoll() { return 80000000LL; } // http://wiki.xiph.org/MatroskaOpus
	virtual uint64_t getCodecDelay() { return ((uint64_t)_opus_pre_skip * 1000000000LL / 48000LL); }
	
	virtual bool compressSamples(float *sampleBuffers[], csSDK_uint32 samples, csSDK_uint32 discardSamples);
	virtual bool endOfStream();

  private:
	OpusMSEncoder *_encoder;
	
	void *_privateData;
	size_t _privateSize;
	
	const int _channels;
	
	int _opus_pre_skip;
	
	PrAudioSample _sampleCount;
	
	float *_interleavedBuffer;
	size_t _interleavedBufferSize;
	
	unsigned char *_compressedBuffer;
};

#endif // WEBM_PREMIERE_OPUS_H
