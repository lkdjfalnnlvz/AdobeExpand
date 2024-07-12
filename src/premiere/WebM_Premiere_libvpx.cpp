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

#include "WebM_Premiere_libvpx.h"

#include <sstream>

#include <assert.h>

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


LibVPXEncoder::LibVPXEncoder(int width, int height, const exRatioValue &pixelAspect,
								const exRatioValue &fps,
								Codec codec,
								WebM_Video_Method method, int quality, int bitrate,
								bool twoPass, bool vbrPass, void *vbrBuffer, size_t vbrBufferSize,
								int keyframeMaxDistance, bool forceKeyframes,
								WebM_Chroma_Sampling sampling, int bitDepth,
								WebM_ColorSpace colorSpace, const std::string &custom,
								PrSDKPPixSuite *pixSuite, PrSDKPPix2Suite *pix2Suite, bool alpha) :
	VideoEncoder(pixSuite, pix2Suite, alpha),
	_iter(NULL),
	_deadline(VPX_DL_GOOD_QUALITY),
	_vbrPass(twoPass && vbrPass),
	_sampling(sampling),
	_bitDepth(bitDepth),
	_colorSpace(colorSpace)
{
	vpx_codec_iface_t *iface = (codec == VP9) ? vpx_codec_vp9_cx() : vpx_codec_vp8_cx();
	
	vpx_codec_enc_cfg_t config;
	vpx_codec_enc_config_default(iface, &config, 0);
	
	config.g_w = width;
	config.g_h = height;
	
	assert(bitDepth == 8 || codec == VP9);
	assert(sampling == WEBM_420 || codec == VP9);
	
	// (only applies to VP9)
	// Profile 0 is 4:2:0 only
	// Profile 1 can do 4:4:4 and 4:2:2
	// Profile 2 can do 10- and 12-bit, 4:2:0 only
	// Profile 3 can do 10- and 12-bit, 4:4:4 and 4:2:2
	config.g_profile = (sampling > WEBM_420 ?
							(bitDepth > 8 ? 3 : 1) :
							(bitDepth > 8 ? 2 : 0) );
	
	config.g_bit_depth = (bitDepth == 12 ? VPX_BITS_12 :
							bitDepth == 10 ? VPX_BITS_10 :
							VPX_BITS_8);
	
	config.g_input_bit_depth = config.g_bit_depth;
	
	
	if(method == WEBM_METHOD_CONSTANT_QUALITY || method == WEBM_METHOD_CONSTRAINED_QUALITY)
	{
		config.rc_end_usage = (method == WEBM_METHOD_CONSTANT_QUALITY ? VPX_Q : VPX_CQ);
		config.g_pass = VPX_RC_ONE_PASS;
		
		const int min_q = config.rc_min_quantizer + 1;
		const int max_q = config.rc_max_quantizer;
		
		// our 0...100 slider will be used to bring max_q down to min_q
		config.rc_max_quantizer = min_q + ((((float)(100 - quality) / 100.f) * (max_q - min_q)) + 0.5f);
	}
	else
	{
		if(method == WEBM_METHOD_VBR)
		{
			config.rc_end_usage = VPX_VBR;
		}
		else if(method == WEBM_METHOD_BITRATE)
		{
			config.rc_end_usage = VPX_CBR;
			config.g_pass = VPX_RC_ONE_PASS;
		}
		else
			assert(false);
	}
	
	if(twoPass)
	{
		if(vbrPass)
		{
			config.g_pass = VPX_RC_FIRST_PASS;
		}
		else
		{
			config.g_pass = VPX_RC_LAST_PASS;
			
			config.rc_twopass_stats_in.buf = vbrBuffer;
			config.rc_twopass_stats_in.sz = vbrBufferSize;
		}
	}
	else
		config.g_pass = VPX_RC_ONE_PASS;
		
	
	config.rc_target_bitrate = bitrate;
	
	
	config.g_threads = g_num_cpus;
	
	config.g_timebase.num = fps.denominator;
	config.g_timebase.den = fps.numerator;
	
	config.kf_max_dist = keyframeMaxDistance;
	
	if(forceKeyframes)
		config.kf_min_dist = config.kf_max_dist;
	
	applyCustomPre(config, _deadline, custom);
	
	assert(config.kf_max_dist >= config.kf_min_dist);
	
		
	const vpx_codec_flags_t flags = (config.g_bit_depth == VPX_BITS_8 ? 0 : VPX_CODEC_USE_HIGHBITDEPTH);
	
	vpx_codec_err_t err = vpx_codec_enc_init(&_encoder, iface, &config, flags);
		
	if(err == VPX_CODEC_OK)
	{
		if(method == WEBM_METHOD_CONSTANT_QUALITY || method == WEBM_METHOD_CONSTRAINED_QUALITY)
		{
			const int min_q = config.rc_min_quantizer;
			const int max_q = config.rc_max_quantizer;
			
			// CQ Level should be between min_q and max_q
			const int cq_level = (min_q + max_q) / 2;
		
			vpx_codec_control(&_encoder, VP8E_SET_CQ_LEVEL, cq_level);
		}
		
		if(codec == VP9)
		{
			vpx_codec_control(&_encoder, VP8E_SET_CPUUSED, 2); // much faster if we do this
			
			vpx_codec_control(&_encoder, VP9E_SET_TILE_COLUMNS, mylog2(g_num_cpus)); // this gives us some multithreading
			vpx_codec_control(&_encoder, VP9E_SET_FRAME_PARALLEL_DECODING, 1);
		}
	
		applyCustomPost(custom);
	}

	if(err != VPX_CODEC_OK)
		throw exportReturn_InternalError;
}

LibVPXEncoder::~LibVPXEncoder()
{
	assert(NULL == vpx_codec_get_cx_data(&_encoder, &_iter));
	
	vpx_codec_err_t err = vpx_codec_destroy(&_encoder);
	
	assert(err == VPX_CODEC_OK);
}

void *
LibVPXEncoder::getPrivateData(size_t &size)
{
	size = 0;
	
	return nullptr;
}

bool
LibVPXEncoder::compressFrame(const PPixHand &frame, PrTime time, PrTime duration)
{
	prRect bounds;
	_pixSuite->GetBounds(frame, &bounds);

	const csSDK_int32 width = (bounds.right - bounds.left);
	const csSDK_int32 height = (bounds.bottom - bounds.top);

	// see validate_img() and validate_config() in vp8_cx_iface.c and vp9_cx_iface.c
	const vpx_img_fmt_t imgfmt8 = _sampling == WEBM_444 ? VPX_IMG_FMT_I444 :
									_sampling == WEBM_422 ? VPX_IMG_FMT_I422 :
									VPX_IMG_FMT_I420;
									
	const vpx_img_fmt_t imgfmt16 = _sampling == WEBM_444 ? VPX_IMG_FMT_I44416 :
									_sampling == WEBM_422 ? VPX_IMG_FMT_I42216 :
									VPX_IMG_FMT_I42016;
									
	const vpx_img_fmt_t imgfmt = (_bitDepth > 8 ? imgfmt16 : imgfmt8);

			
	vpx_image_t img_data;
	vpx_image_t *img = vpx_img_alloc(&img_data, imgfmt, width, height, 32);
	
	vpx_codec_err_t err = VPX_CODEC_OK;
	
	if(img)
	{
		img->cs = (_colorSpace == WEBM_REC709 ? VPX_CS_BT_709 : VPX_CS_BT_601);
		img->range = VPX_CR_STUDIO_RANGE;
		
		if(_bitDepth > 8)
		{
			img->bit_depth = _bitDepth;
			img->bps = img->bps * _bitDepth / 16;
		}
		else
			assert(img->bit_depth == 8);
			
		YUVBuffer yuv;
		
		yuv.width = width;
		yuv.height = height;
		yuv.sampling = _sampling;
		yuv.bitDepth = _bitDepth;
		yuv.colorSpace = _colorSpace;
		yuv.fullRange = false;
		yuv.y = img->planes[VPX_PLANE_Y];
		yuv.u = img->planes[VPX_PLANE_U];
		yuv.v = img->planes[VPX_PLANE_V];
		yuv.yRowbytes = img->stride[VPX_PLANE_Y];
		yuv.uRowbytes = img->stride[VPX_PLANE_U];
		yuv.vRowbytes = img->stride[VPX_PLANE_V];
		
		CopyPixToBuffer(yuv, frame);
	
		err = vpx_codec_encode(&_encoder, img, time, duration, 0, _deadline);
		
		if(err == VPX_CODEC_OK)
			_iter = NULL;
		
		vpx_img_free(img);
	}
	else
		return false;
	
	if(err == VPX_CODEC_OK)
		checkForPackets();
	
	return (err == VPX_CODEC_OK);
}

bool
LibVPXEncoder::endOfStream()
{
	vpx_codec_err_t err = vpx_codec_encode(&_encoder, NULL, 0, 1, 0, _deadline);
	
	if(err == VPX_CODEC_OK)
	{
		_iter = NULL;
		
		checkForPackets();
	}
	
	return (err == VPX_CODEC_OK);
}

void
LibVPXEncoder::checkForPackets()
{
	while(const vpx_codec_cx_pkt_t *pkt = vpx_codec_get_cx_data(&_encoder, &_iter))
	{
		if(_vbrPass)
		{
			if(pkt->kind == VPX_CODEC_STATS_PKT)
			{
				Packet *packet = new Packet;
				
				packet->size = pkt->data.twopass_stats.sz;
				packet->data = malloc(packet->size);
				if(packet->data == NULL)
					throw exportReturn_ErrMemory;
				memcpy(packet->data, pkt->data.twopass_stats.buf, packet->size);
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
			if(pkt->kind == VPX_CODEC_CX_FRAME_PKT)
			{
				assert( !(pkt->data.frame.flags & VPX_FRAME_IS_INVISIBLE) ); // libwebm not handling these now
				assert( !(pkt->data.frame.flags & VPX_FRAME_IS_FRAGMENT) );
			
				Packet *packet = new Packet;
				
				packet->size = pkt->data.frame.sz;
				packet->data = malloc(packet->size);
				if(packet->data == NULL)
					throw exportReturn_ErrMemory;
				memcpy(packet->data, pkt->data.frame.buf, packet->size);
				packet->time = pkt->data.frame.pts;
				packet->duration = pkt->data.frame.duration;
				packet->keyframe = (pkt->data.frame.flags & VPX_FRAME_IS_KEY);
				
				_queue.push(packet);
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
LibVPXEncoder::applyCustomPre(vpx_codec_enc_cfg_t &config, unsigned long &deadline, const std::string &custom)
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
			
			if(arg == "--best")
			{	deadline = VPX_DL_BEST_QUALITY;	}
			
			else if(arg == "--good")
			{	deadline = VPX_DL_GOOD_QUALITY;	}
			
			else if(arg == "--rt")
			{	deadline = VPX_DL_REALTIME;	}
			
			else if(arg == "-d" || arg == "--deadline")
			{	SetValue(deadline, val); i++;	}

			else if(arg == "-t" || arg == "--threads")
			{	SetValue(config.g_threads, val); i++;	}
			
			else if(arg == "--lag-in-frames")
			{	SetValue(config.g_lag_in_frames, val); i++;	}
			
			else if(arg == "--drop-frame")
			{	SetValue(config.rc_dropframe_thresh, val); i++;	}
			
			else if(arg == "--resize-allowed")
			{	SetValue(config.rc_resize_allowed, val); i++;	}
			
			else if(arg == "--resize-width")
			{	SetValue(config.rc_scaled_width, val); i++;	}
			
			else if(arg == "--resize-height")
			{	SetValue(config.rc_scaled_height, val); i++;	}
			
			else if(arg == "--resize-up")
			{	SetValue(config.rc_resize_up_thresh, val); i++;	}
			
			else if(arg == "--resize-down")
			{	SetValue(config.rc_resize_down_thresh, val); i++;	}
			
			else if(arg == "--target-bitrate")
			{	SetValue(config.rc_target_bitrate, val); i++;	}
			
			else if(arg == "--min-q")
			{	SetValue(config.rc_min_quantizer, val); i++;	}
			
			else if(arg == "--max-q")
			{	SetValue(config.rc_max_quantizer, val); i++;	}
			
			else if(arg == "--undershoot-pct")
			{	SetValue(config.rc_undershoot_pct, val); i++;	}
			
			else if(arg == "--overshoot-pct")
			{	SetValue(config.rc_overshoot_pct, val); i++;	}

			else if(arg == "--buf-sz")
			{	SetValue(config.rc_buf_sz, val); i++;	}

			else if(arg == "--buf-initial-sz")
			{	SetValue(config.rc_buf_initial_sz, val); i++;	}

			else if(arg == "--buf-optimal-sz")
			{	SetValue(config.rc_buf_optimal_sz, val); i++;	}

			else if(arg == "--bias-pct")
			{	SetValue(config.rc_2pass_vbr_bias_pct, val); i++;	}

			else if(arg == "--minsection-pct")
			{	SetValue(config.rc_2pass_vbr_minsection_pct, val); i++;	}

			else if(arg == "--maxsection-pct")
			{	SetValue(config.rc_2pass_vbr_maxsection_pct, val); i++;	}

			else if(arg == "--kf-min-dist")
			{	SetValue(config.kf_min_dist, val); i++;	}

			else if(arg == "--kf-max-dist")
			{	SetValue(config.kf_max_dist, val); i++;	}

			else if(arg == "--disable-kf")
			{	config.kf_mode = VPX_KF_DISABLED;	}

			
			i++;
		}
	}
}

#define ConfigureVPXValue(encoder, ctrl_id, s) \
	do{							\
		std::stringstream ss;	\
		ss << s;				\
		int v = 0;				\
		ss >> v;				\
		vpx_codec_err_t err = vpx_codec_control(encoder, ctrl_id, v); \
		if(err != VPX_CODEC_OK)	\
			config_err = err;	\
	}while(0)

void
LibVPXEncoder::applyCustomPost(const std::string &custom)
{
	std::vector<std::string> args;
	
	if(quotedTokenize(custom, args, " =\t\r\n") && args.size() > 0)
	{
		vpx_codec_err_t config_err = VPX_CODEC_OK;
		
		const int num_args = args.size();

		args.push_back(""); // so there's always an i+1
		
		int i = 0;
		
		while(i < num_args)
		{
			const std::string &arg = args[i];
			const std::string &val = args[i + 1];
		
			if(arg == "--noise-sensitivity")
			{	ConfigureVPXValue(&_encoder, VP8E_SET_NOISE_SENSITIVITY, val); i++;	}

			else if(arg == "--sharpness")
			{	ConfigureVPXValue(&_encoder, VP8E_SET_SHARPNESS, val); i++;	}

			else if(arg == "--static-thresh")
			{	ConfigureVPXValue(&_encoder, VP8E_SET_STATIC_THRESHOLD, val); i++;	}

			else if(arg == "--cpu-used")
			{	ConfigureVPXValue(&_encoder, VP8E_SET_CPUUSED, val); i++;	}

			else if(arg == "--token-parts")
			{	ConfigureVPXValue(&_encoder, VP8E_SET_TOKEN_PARTITIONS, val); i++;	}

			else if(arg == "--tile-columns")
			{	ConfigureVPXValue(&_encoder, VP9E_SET_TILE_COLUMNS, val); i++;	}

			else if(arg == "--tile-rows")
			{	ConfigureVPXValue(&_encoder, VP9E_SET_TILE_ROWS, val); i++;	}
			
			else if(arg == "--auto-alt-ref")
			{	ConfigureVPXValue(&_encoder, VP8E_SET_ENABLEAUTOALTREF, val); i++;	}

			else if(arg == "--arnr-maxframes")
			{	ConfigureVPXValue(&_encoder, VP8E_SET_ARNR_MAXFRAMES, val); i++;	}

			else if(arg == "--arnr-strength")
			{	ConfigureVPXValue(&_encoder, VP8E_SET_ARNR_STRENGTH, val); i++;	}

			else if(arg == "--arnr-type")
			{	ConfigureVPXValue(&_encoder, VP8E_SET_ARNR_TYPE, val); i++;	}

			else if(arg == "--tune")
			{
				unsigned int ival = val == "psnr" ? VP8_TUNE_PSNR :
									val == "ssim" ? VP8_TUNE_SSIM :
									VP8_TUNE_PSNR;
			
				ConfigureVPXValue(&_encoder, VP8E_SET_TUNING, ival);
				i++;
			}

			else if(arg == "--cq-level")
			{	ConfigureVPXValue(&_encoder, VP8E_SET_CQ_LEVEL, val); i++;	}
			
			else if(arg == "--max-intra-rate")
			{	ConfigureVPXValue(&_encoder, VP8E_SET_MAX_INTRA_BITRATE_PCT, val); i++;	}

			else if(arg == "--gf-cbr-boost")
			{	ConfigureVPXValue(&_encoder, VP9E_SET_GF_CBR_BOOST_PCT, val); i++;	}

			else if(arg == "--screen-content-mode")
			{	ConfigureVPXValue(&_encoder, VP8E_SET_SCREEN_CONTENT_MODE, val);	i++;	}
			
			else if(arg == "--lossless")
			{	ConfigureVPXValue(&_encoder, VP9E_SET_LOSSLESS, 1);	}
			
			else if(arg == "--frame-parallel")
			{	ConfigureVPXValue(&_encoder, VP9E_SET_FRAME_PARALLEL_DECODING, val);	i++;	}

			else if(arg == "--aq-mode")
			{	ConfigureVPXValue(&_encoder, VP9E_SET_AQ_MODE, val);	i++;	}

			else if(arg == "--frame_boost")
			{	ConfigureVPXValue(&_encoder, VP9E_SET_FRAME_PERIODIC_BOOST, val); i++;	}

			else if(arg == "--noise-sensitivity")
			{	ConfigureVPXValue(&_encoder, VP9E_SET_NOISE_SENSITIVITY, val); i++;	}
			
			else if(arg == "--tune-content")
			{
				unsigned int ival = val == "default" ? VP9E_CONTENT_DEFAULT :
									val == "screen" ? VP9E_CONTENT_SCREEN :
									val == "film" ? VP9E_CONTENT_FILM :
									VP9E_CONTENT_DEFAULT;
			
				ConfigureVPXValue(&_encoder, VP9E_SET_TUNE_CONTENT, ival);
				i++;
			}

			else if(arg == "--color-space")
			{
				unsigned int ival = val == "unknown" ? VPX_CS_UNKNOWN :
									val == "bt601" ? VPX_CS_BT_601 :
									val == "bt709" ? VPX_CS_BT_709 :
									val == "smpte170" ? VPX_CS_SMPTE_170 :
									val == "smpte240" ? VPX_CS_SMPTE_240 :
									val == "bt2020" ? VPX_CS_BT_2020 :
									val == "sRGB" ? VPX_CS_SRGB :
									VPX_CS_UNKNOWN;
			
				ConfigureVPXValue(&_encoder, VP9E_SET_COLOR_SPACE, ival);
				i++;
			}
			
			else if(arg == "--min-gf-interval")
			{	ConfigureVPXValue(&_encoder, VP9E_SET_MIN_GF_INTERVAL, val); i++;	}
			
			else if(arg == "--max-gf-interval")
			{	ConfigureVPXValue(&_encoder, VP9E_SET_MAX_GF_INTERVAL, val); i++;	}

			else if(arg == "--target-level")
			{	ConfigureVPXValue(&_encoder, VP9E_SET_TARGET_LEVEL, val); i++;	}

			else if(arg == "--row-mt")
			{	ConfigureVPXValue(&_encoder, VP9E_SET_ROW_MT, 1);	}

			else if(arg == "--color-range")
			{
				unsigned int ival = val == "studio" ? VPX_CR_STUDIO_RANGE :
									val == "full" ? VPX_CR_FULL_RANGE :
									VPX_CR_STUDIO_RANGE;
			
				ConfigureVPXValue(&_encoder, VP9E_SET_COLOR_RANGE, ival);
				i++;
			}

			i++;
		}
		
		assert(config_err == VPX_CODEC_OK);
	}
}
