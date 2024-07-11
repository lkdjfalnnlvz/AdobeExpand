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

#ifndef WEBM_PREMIERE_VORBIS_H
#define WEBM_PREMIERE_VORBIS_H

#include "WebM_Premiere_Codecs.h"

#include <vorbis/codec.h>
#include <vorbis/vorbisenc.h>


class VorbisEncoder : public AudioEncoder
{
  public:
	VorbisEncoder(int channels, float sampleRate, Ogg_Method method, float quality, int bitrate);
	virtual ~VorbisEncoder();

	virtual void * getPrivateData(size_t &size);
	virtual uint64_t getSeekPreRoll() { return 0; }
	virtual uint64_t getCodecDelay() { return 0; }
	
	virtual bool compressSamples(float *sampleBuffers[], csSDK_uint32 samples, csSDK_uint32 discardSamples);
	virtual bool endOfStream();

  private:
	void checkForPackets();

  private:
	vorbis_info _vi;
	vorbis_comment _vc;
	vorbis_dsp_state _vd;
	vorbis_block _vb;
	
	void *_privateData;
	size_t _privateSize;
	
	const int _channels;
};

#endif // WEBM_PREMIERE_VORBIS_H
