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

#include "WebM_Premiere_SVT-AV1.h"

#include <sstream>


extern int g_num_cpus;


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


SVTAV1Encoder::SVTAV1Encoder(int width, int height, const exRatioValue &pixelAspect,
							const exRatioValue &fps,
							WebM_Video_Method method, int quality, int bitrate,
							bool twoPass, bool vbrPass, void *vbrBuffer, size_t vbrBufferSize,
							int keyframeMaxDistance, bool forceKeyframes,
							WebM_Chroma_Sampling sampling, int bitDepth,
							WebM_ColorSpace colorSpace, const std::string &custom,
							PrSDKPPixSuite *pixSuite, PrSDKPPix2Suite *pix2Suite, bool alpha) :
	VideoEncoder(pixSuite, pix2Suite, alpha),
	_encoder(NULL),
	_eos(false),
	_privateData(NULL),
	_privateSize(0),
	_vbrPass(twoPass && vbrPass),
	_forceKeyframes(forceKeyframes),
	_keyframeMaxDistance(keyframeMaxDistance),
	_sampling(sampling),
	_bitDepth(bitDepth),
	_colorSpace(colorSpace)
{
	EbSvtAv1EncConfiguration config;
	
	EbErrorType err = svt_av1_enc_init_handle(&_encoder, NULL, &config);
	
	if(err == EB_ErrorNone)
	{
		config.source_width = width;
		config.source_height = height;
	
		config.profile = (bitDepth == 12 || sampling == WEBM_422) ? PROFESSIONAL_PROFILE :
							sampling == WEBM_444 ? HIGH_PROFILE : MAIN_PROFILE;

		config.encoder_bit_depth = bitDepth;


		if(method == WEBM_METHOD_CONSTANT_QUALITY || method == WEBM_METHOD_CONSTRAINED_QUALITY)
		{
			assert(!twoPass);
		
			config.rate_control_mode = SVT_AV1_RC_MODE_CQP_OR_CRF;
			
			const int min_q = config.min_qp_allowed;
			const int max_q = config.max_qp_allowed;

			// our 0...100 slider will be used to bring max_q down to min_q
			config.qp = min_q + ((((float)(100 - quality) / 100.f) * (max_q - min_q)) + 0.5f);
		}
		else
		{
			if(method == WEBM_METHOD_VBR)
			{
				config.rate_control_mode = SVT_AV1_RC_MODE_VBR;
				
				assert(config.pred_structure == SVT_AV1_PRED_RANDOM_ACCESS);
			}
			else if(method == WEBM_METHOD_BITRATE)
			{
				config.rate_control_mode = SVT_AV1_RC_MODE_CBR;
				
				config.pred_structure = SVT_AV1_PRED_LOW_DELAY_B; // get an error if I don't set this
				
				assert(!twoPass); // "Multi-passes is not support with Low Delay mode"
			}
			else
				assert(false);
				
			config.target_bit_rate = bitrate * 1000;
		}

		if(twoPass)
		{
			if(vbrPass)
			{
				config.pass = SVT_FIRST_PASS;
			}
			else
			{
				config.pass = SVT_SECOND_PASS;

				config.rc_stats_buffer.buf = vbrBuffer;
				config.rc_stats_buffer.sz = vbrBufferSize;
			}
		}
		else
			config.pass = SVT_SINGLE_PASS;


		config.enc_mode = 2; // slow, but not too slow?
		
		config.logical_processors = g_num_cpus;

		config.frame_rate_numerator = fps.numerator;
		config.frame_rate_denominator = fps.denominator;

		//config.intra_period_length = keyframeMaxDistance;
		
		config.force_key_frames = forceKeyframes;
		
		applyCustom(config, custom);
		
		
		if(err == EB_ErrorNone)
			err = svt_av1_enc_set_parameter(_encoder, &config);
		
		if(err == EB_ErrorNone)
			err = svt_av1_enc_init(_encoder);
		
		if(err == EB_ErrorNone)
		{
			EbBufferHeaderType *headerBuf = NULL;
			
			err = svt_av1_enc_stream_header(_encoder, &headerBuf);
			
			if(err == EB_ErrorNone && headerBuf != NULL)
			{
				_privateSize = headerBuf->n_filled_len;
				_privateData = malloc(_privateSize);
				if(_privateData == NULL)
					throw exportReturn_ErrMemory;
				memcpy(_privateData, headerBuf->p_buffer, _privateSize);
				
				svt_av1_enc_stream_header_release(headerBuf);
			}
		}
		
		if(err == EB_ErrorNone)
		{
			_header.size = sizeof(EbBufferHeaderType);
			_header.p_buffer = (uint8_t *)&_image;
			_header.p_app_private = NULL;
			_header.pic_type = EB_AV1_INVALID_PICTURE;
			_header.flags = 0;
			_header.qp = 0;
			_header.luma_sse = _header.cb_sse = _header.cr_sse = 0;
			_header.luma_ssim = _header.cb_ssim = _header.cr_ssim = 0;
			_header.metadata = NULL;

			const size_t y_rowbytes = (bitDepth > 8 ? sizeof(uint16_t) : sizeof(uint8_t)) * width;
			const size_t y_size = (y_rowbytes * height);
			const size_t c_rowbytes = (y_rowbytes / (sampling == WEBM_444 ? 1 : 2));
			const size_t c_size = (c_rowbytes * height / (sampling == WEBM_444 ? 1 : 2));

			_image.luma = (uint8_t *)malloc(y_size);
			if(_image.luma == NULL)
				throw exportReturn_ErrMemory;

			_image.cb = (uint8_t *)malloc(c_size);
			if(_image.cb == NULL)
				throw exportReturn_ErrMemory;
				
			_image.cr = (uint8_t *)malloc(c_size);
			if(_image.cr == NULL)
				throw exportReturn_ErrMemory;
				
			_header.n_alloc_len = _header.n_filled_len = (y_size + c_size + c_size);

			_image.y_stride = y_rowbytes / (bitDepth > 8 ? 2 : 1);
			_image.cb_stride = c_rowbytes / (bitDepth > 8 ? 2 : 1);
			_image.cr_stride = c_rowbytes / (bitDepth > 8 ? 2 : 1);

			_image.width = width;
			_image.height = height;

			_image.org_x = 0;
			_image.org_y = 0;

			_image.color_fmt = (sampling == WEBM_444 ? EB_YUV444 : EB_YUV420);
			_image.bit_depth = (EbBitDepth)bitDepth; // EB_EIGHT_BIT, EB_TEN_BIT, EB_TWELVE_BIT
		}
	}
	
	if(err != EB_ErrorNone)
		throw exportReturn_InternalError;
}

SVTAV1Encoder::~SVTAV1Encoder()
{
	EbBufferHeaderType *packet = NULL;
	assert(EB_NoErrorEmptyQueue == svt_av1_enc_get_packet(_encoder, &packet, TRUE));
	
	EbErrorType err = svt_av1_enc_deinit(_encoder);
	assert(err == EB_ErrorNone);
	
	err = svt_av1_enc_deinit_handle(_encoder);
	assert(err == EB_ErrorNone);
	
	if(_privateData != NULL)
		free(_privateData);
	
	if(_image.luma != NULL)
		free(_image.luma);

	if(_image.cb != NULL)
		free(_image.cb);

	if(_image.cr != NULL)
		free(_image.cr);
}

void *
SVTAV1Encoder::getPrivateData(size_t &size)
{
	size = _privateSize;
	
	return _privateData;
}

bool
SVTAV1Encoder::compressFrame(const PPixHand &frame, PrTime time, PrTime duration)
{
	prRect bounds;
	_pixSuite->GetBounds(frame, &bounds);

	const csSDK_int32 width = (bounds.right - bounds.left);
	const csSDK_int32 height = (bounds.bottom - bounds.top);

	assert(width == _image.width && height == _image.height);

	YUVBuffer yuv;
	
	yuv.width = width;
	yuv.height = height;
	yuv.sampling = _sampling;
	yuv.bitDepth = _bitDepth;
	yuv.colorSpace = _colorSpace;
	yuv.fullRange = false;
	yuv.y = _image.luma;
	yuv.u = _image.cb;
	yuv.v = _image.cr;
	yuv.yRowbytes = _image.y_stride * (_image.bit_depth > 8 ? 2 : 1);
	yuv.uRowbytes = _image.cb_stride * (_image.bit_depth > 8 ? 2 : 1);
	yuv.vRowbytes = _image.cr_stride * (_image.bit_depth > 8 ? 2 : 1);
	
	CopyPixToBuffer(yuv, frame);
	
											
	_header.n_tick_count = _header.dts = _header.pts = time;
	assert(duration == 1);
	_header.flags = 0;
	_header.pic_type = (_forceKeyframes ? (time % _keyframeMaxDistance == 0 ? EB_AV1_KEY_PICTURE : EB_AV1_INTER_PICTURE) : EB_AV1_INVALID_PICTURE);
	
	
	EbErrorType err = svt_av1_enc_send_picture(_encoder, &_header);
	
	if(err == EB_ErrorNone)
		checkForPackets();
	
	return (err == EB_ErrorNone);
}

bool
SVTAV1Encoder::endOfStream()
{
	_header.pts = 0;
	_header.flags = EB_BUFFERFLAG_EOS;

	EbErrorType err = svt_av1_enc_send_picture(_encoder, &_header);
	
	_eos = true;
	
	if(err == EB_ErrorNone)
		checkForPackets();
	
	return (err == EB_ErrorNone);
}

void
SVTAV1Encoder::checkForPackets()
{
	EbErrorType err = EB_ErrorNone;
	
	while(err == EB_ErrorNone)
	{
		EbBufferHeaderType *pkt = NULL;
		
		err = svt_av1_enc_get_packet(_encoder, &pkt, _eos);
		
		if(err == EB_ErrorNone && pkt != NULL)
		{
			if(_vbrPass)
			{
				if(pkt->flags & EB_BUFFERFLAG_EOS)
				{
					assert(pkt->p_buffer == NULL && pkt->n_filled_len == 0);
					
					SvtAv1FixedBuf first_pass_stat;
					
					err = svt_av1_enc_get_stream_info(_encoder, SVT_AV1_STREAM_INFO_FIRST_PASS_STATS_OUT, &first_pass_stat);
					
					if(err == EB_ErrorNone)
					{
						Packet *packet = new Packet;
						
						packet->size = first_pass_stat.sz;
						packet->data = malloc(packet->size);
						if(packet->data == NULL)
							throw exportReturn_ErrMemory;
						memcpy(packet->data, first_pass_stat.buf, packet->size);
						packet->time = 0;
						packet->duration = 0;
						packet->keyframe = false;
						
						_queue.push(packet);
					}
					else
						assert(false);
				}
				else
				{
					assert(pkt->flags & EB_BUFFERFLAG_HAS_TD);
					
					Packet *packet = new Packet;
					
					packet->size = pkt->n_filled_len;
					packet->data = malloc(packet->size);
					if(packet->data == NULL)
						throw exportReturn_ErrMemory;
					memcpy(packet->data, pkt->p_buffer, packet->size);
					packet->time = 0;
					packet->duration = 0;
					packet->keyframe = false;
					
					_queue.push(packet);
				}
			}
			else
			{
				if(!(pkt->flags & EB_BUFFERFLAG_EOS))
				{
					Packet *packet = new Packet;
					
					packet->size = pkt->n_filled_len;
					packet->data = malloc(packet->size);
					if(packet->data == NULL)
						throw exportReturn_ErrMemory;
					memcpy(packet->data, pkt->p_buffer, packet->size);
					packet->time = pkt->pts;
					packet->duration = 1;
					packet->keyframe = (pkt->pic_type == EB_AV1_KEY_PICTURE);
					
					_queue.push(packet);
				}
			}
		}
		else
			assert(err == EB_NoErrorEmptyQueue); // anything else is a real error
	}
}

void
SVTAV1Encoder::applyCustom(EbSvtAv1EncConfiguration &config, const std::string &custom)
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
			
			EbErrorType err = svt_av1_enc_parse_parameter(&config, arg.c_str(), val.c_str());
			
			i += (err == EB_ErrorNone ? 2 : 1);
		}
	}
}
