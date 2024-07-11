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

#include "WebM_Premiere_Opus.h"


//extern int g_num_cpus;


OpusEncoder::OpusEncoder(int channels, float sampleRate, Ogg_Method method, float quality, int bitrate, bool autoBitrate) :
	AudioEncoder(),
	_channels(channels),
	_sampleCount(0),
	_interleavedBuffer(NULL),
	_interleavedBufferSize(0),
	_compressedBuffer(NULL)
{
	assert(sampleRate == 48000.f);
	
	const int sample_rate = 48000;
	
	const int mapping_family = (channels > 2 ? 1 : 0);
	
	const int streams = (channels > 2 ? 4 : 1);
	
	const int coupled_streams = (channels > 2 ? 2 :
									channels == 2 ? 1:
									0);
	
	const unsigned char surround_mapping[6] = {0, 4, 1, 2, 3, 5};
	const unsigned char stereo_mapping[6] = {0, 1, 0, 1, 0, 1};
	
	const unsigned char *mapping = (channels > 2 ? surround_mapping : stereo_mapping);
	
	int err = -1;
	
	_encoder = opus_multistream_encoder_create(sample_rate, channels,
												streams, coupled_streams, mapping,
												OPUS_APPLICATION_AUDIO, &err);
	
	if(_encoder != NULL && err == OPUS_OK)
	{
		if(!autoBitrate)
			opus_multistream_encoder_ctl(_encoder, OPUS_SET_BITRATE(bitrate * 1000));
		
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
		id_head[9] = channels;
		
		
		// pre-skip
		opus_int32 skip = 0;
		opus_multistream_encoder_ctl(_encoder, OPUS_GET_LOOKAHEAD(&skip));
		_opus_pre_skip = skip;
		
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
			assert(channels == 6);
		
			id_head[19] = streams;
			id_head[20] = coupled_streams;
			memcpy(&id_head[21], mapping, 6);
			
			id_header_size = 27;
		}
		else
		{
			id_header_size = 19;
		}
		
		_privateSize = id_header_size;
		
		_privateData = malloc(_privateSize);
		
		if(_privateData == NULL)
			throw exportReturn_ErrMemory;
		
		memcpy(_privateData, id_head, _privateSize);
	}
	else
		throw exportReturn_InternalError;
}

OpusEncoder::~OpusEncoder()
{
	assert(_queue.empty());
	
	opus_multistream_encoder_destroy(_encoder);
	
	if(_privateData != NULL)
		free(_privateData);
	
	if(_interleavedBuffer != NULL)
		free(_interleavedBuffer);
	
	if(_compressedBuffer != NULL)
		free(_compressedBuffer);
}

void *
OpusEncoder::getPrivateData(size_t &size)
{
	size = _privateSize;
	
	return _privateData;
}

bool
OpusEncoder::compressSamples(float *sampleBuffers[], csSDK_uint32 samples, csSDK_uint32 discardSamples)
{
	const size_t bufferSize = (sizeof(float) * _channels * samples);
	
	if(_interleavedBuffer == NULL)
	{
		_interleavedBuffer = (float *)malloc(bufferSize);
		if(_interleavedBuffer == NULL)
			throw exportReturn_ErrMemory;
		_interleavedBufferSize = bufferSize;
		
		assert(_compressedBuffer == NULL);
		_compressedBuffer = (unsigned char *)malloc(bufferSize * 2);
		if(_compressedBuffer == NULL)
			throw exportReturn_ErrMemory;
	}
	else if(_interleavedBufferSize < bufferSize)
	{
		_interleavedBuffer = (float *)realloc(_interleavedBuffer, bufferSize);
		if(_interleavedBuffer == NULL)
			throw exportReturn_ErrMemory;
		_interleavedBufferSize = bufferSize;
		
		assert(_compressedBuffer != NULL);
		_compressedBuffer = (unsigned char *)realloc(_compressedBuffer, bufferSize * 2);
		if(_compressedBuffer == NULL)
			throw exportReturn_ErrMemory;
	}
	
	
	for(csSDK_uint32 i=0; i < samples; i++)
	{
		for(int c=0; c < _channels; c++)
		{
			_interleavedBuffer[(i * _channels) + c] = sampleBuffers[c][i];
		}
	}
	
	int bytes = opus_multistream_encode_float(_encoder, _interleavedBuffer, samples,
												_compressedBuffer, _interleavedBufferSize * 2);
	
	if(bytes > 0)
	{
		Packet *packet = new Packet;
		
		packet->size = bytes;
		packet->data = malloc(packet->size);
		if(packet->data == NULL)
			throw exportReturn_ErrMemory;
		memcpy(packet->data, _compressedBuffer, packet->size);
		packet->sampleIndex = _sampleCount;
		packet->discardSamples = discardSamples;
		
		_queue.push(packet);
		
		_sampleCount += samples;
	}
	else if(bytes < 0)
	{
		return false;
	}
	else
		assert(false);
	
	return true;
}

bool
OpusEncoder::endOfStream()
{
	assert(false);
}
