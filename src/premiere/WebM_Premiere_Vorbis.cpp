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

#include "WebM_Premiere_Vorbis.h"

#include <assert.h>


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

VorbisEncoder::VorbisEncoder(int channels, float sampleRate, Ogg_Method method, float quality, int bitrate) :
	AudioEncoder(),
	_channels(channels)
{
#define OV_OK 0

	int err = OV_OK;

	vorbis_info_init(&_vi);
	
	long min_bitrate = -1, max_bitrate = -1;
	vorbis_get_limits(channels, sampleRate, min_bitrate, max_bitrate);
	
	const bool qualityOnly = (min_bitrate < 0 || max_bitrate < 0); // user should have used Quality
	
	if(method == OGG_BITRATE && !qualityOnly)
	{
		long v_bitrate = bitrate * 1000;
		
		if(v_bitrate < min_bitrate)
			v_bitrate = min_bitrate;
		else if(v_bitrate > max_bitrate)
			v_bitrate = max_bitrate;
		
	
		err = vorbis_encode_init(&_vi,
									channels,
									sampleRate,
									-1,
									v_bitrate,
									-1);
	}
	else
	{
		err = vorbis_encode_init_vbr(&_vi,
										channels,
										sampleRate,
										quality);
	}
	
	if(err != OV_OK)
		throw exportReturn_InternalError;
	
	vorbis_comment_init(&_vc);
	vorbis_analysis_init(&_vd, &_vi);
	vorbis_block_init(&_vd, &_vb);
	
	
	ogg_packet header;
	ogg_packet header_comm;
	ogg_packet header_code;
	
	err = vorbis_analysis_headerout(&_vd, &_vc, &header, &header_comm, &header_code);
	
	if(err != OV_OK)
		throw exportReturn_InternalError;
		
	_privateData = MakePrivateData(header, header_comm, header_code, _privateSize);
	
	if(_privateData == NULL)
		throw exportReturn_ErrMemory;
}

VorbisEncoder::~VorbisEncoder()
{
	assert(_queue.empty());

	if(_privateData != NULL)
		free(_privateData);

	assert(vorbis_analysis_blockout(&_vd, &_vb) == 0);
	
	vorbis_block_clear(&_vb);
	vorbis_dsp_clear(&_vd);
	vorbis_comment_clear(&_vc);
	vorbis_info_clear(&_vi);
}

void *
VorbisEncoder::getPrivateData(size_t &size)
{
	size = _privateSize;
	
	return _privateData;
}

bool
VorbisEncoder::compressSamples(float *sampleBuffers[], csSDK_uint32 samples, csSDK_uint32 discardSamples)
{
	assert(discardSamples == 0);

	float **buffer = vorbis_analysis_buffer(&_vd, samples);
	
	if(buffer == NULL)
		return false;
	
	const size_t bufferSize = (sizeof(float) * samples);
	
	for(int c=0; c < _channels; c++)
	{
		memcpy(buffer[c], sampleBuffers[c], bufferSize);
	}
	
	int err = vorbis_analysis_wrote(&_vd, samples);
	
	checkForPackets();
	
	return (err == OV_OK);
}

bool
VorbisEncoder::endOfStream()
{
	int err = vorbis_analysis_wrote(&_vd, NULL);
	
	checkForPackets();
	
	return (err == OV_OK);
}

void
VorbisEncoder::checkForPackets()
{
	while(vorbis_analysis_blockout(&_vd, &_vb) == 1)
	{
		int err = vorbis_analysis(&_vb, NULL);
		
		assert(err == OV_OK);
		
		err = vorbis_bitrate_addblock(&_vb);
		
		assert(err == OV_OK);
		
		ogg_packet op;
		
		while(vorbis_bitrate_flushpacket(&_vd, &op))
		{
			Packet *packet = new Packet;
			
			packet->size = op.bytes;
			packet->data = malloc(packet->size);
			if(packet->data == NULL)
				throw exportReturn_ErrMemory;
			memcpy(packet->data, op.packet, packet->size);
			packet->sampleIndex = op.granulepos;
			packet->discardSamples = 0;
			
			_queue.push(packet);
		}
	}
}
