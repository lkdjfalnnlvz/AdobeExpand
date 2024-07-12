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

#include "WebM_Premiere_aom.h"

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


AOMEncoder::AOMEncoder(int width, int height, const exRatioValue &pixelAspect,
						const exRatioValue &fps,
						WebM_Video_Method method, int quality, int bitrate,
						bool twoPass, bool vbrPass, void *vbrBuffer, size_t vbrBufferSize,
						int keyframeMaxDistance, bool forceKeyframes,
						WebM_Chroma_Sampling sampling, int bitDepth,
						WebM_ColorSpace colorSpace, const std::string &custom,
						PrSDKPPixSuite *pixSuite, PrSDKPPix2Suite *pix2Suite, bool alpha) :
	VideoEncoder(pixSuite, pix2Suite, alpha),
	_iter(NULL),
	_privateData(NULL),
	_privateSize(0),
	_vbrPass(twoPass && vbrPass),
	_sampling(sampling),
	_bitDepth(bitDepth),
	_colorSpace(colorSpace)
{
	aom_codec_iface_t *iface = aom_codec_av1_cx();
	
	aom_codec_enc_cfg_t config;
	aom_codec_enc_config_default(iface, &config, AOM_USAGE_GOOD_QUALITY);
	
	config.g_w = width;
	config.g_h = height;
	
	// Profile 0 (Main) can do 8- and 10-bit 4:2:0 and 4:0:0 (monochrome)
	// Profile 1 (High) adds 4:4:4
	// Profile 2 (Professional) adds 12-bit and 4:2:2
	config.g_profile = (bitDepth == 12 || sampling == WEBM_422) ? 2 :
						sampling == WEBM_444 ? 1 : 0;

	config.g_bit_depth = (bitDepth == 12 ? AOM_BITS_12 :
							bitDepth == 10 ? AOM_BITS_10 :
							AOM_BITS_8);

	config.g_input_bit_depth = config.g_bit_depth;
	
	
	if(method == WEBM_METHOD_CONSTANT_QUALITY || method == WEBM_METHOD_CONSTRAINED_QUALITY)
	{
		config.rc_end_usage = (method == WEBM_METHOD_CONSTANT_QUALITY ? AOM_Q : AOM_CQ);
		config.g_pass = AOM_RC_ONE_PASS;
		
		const int min_q = config.rc_min_quantizer + 1;
		const int max_q = config.rc_max_quantizer;
		
		// our 0...100 slider will be used to bring max_q down to min_q
		config.rc_max_quantizer = min_q + ((((float)(100 - quality) / 100.f) * (max_q - min_q)) + 0.5f);
	}
	else
	{
		if(method == WEBM_METHOD_VBR)
		{
			config.rc_end_usage = AOM_VBR;
		}
		else if(method == WEBM_METHOD_BITRATE)
		{
			config.rc_end_usage = AOM_CBR;
			config.g_pass = AOM_RC_ONE_PASS;
		}
		else
			assert(false);
	}
	
	if(twoPass)
	{
		if(vbrPass)
		{
			config.g_pass = AOM_RC_FIRST_PASS;
		}
		else
		{
			config.g_pass = AOM_RC_LAST_PASS;
			
			config.rc_twopass_stats_in.buf = vbrBuffer;
			config.rc_twopass_stats_in.sz = vbrBufferSize;
		}
	}
	else
		config.g_pass = AOM_RC_ONE_PASS;
		
	
	config.rc_target_bitrate = bitrate;
	
	
	config.g_threads = g_num_cpus;
	
	config.g_timebase.num = fps.denominator;
	config.g_timebase.den = fps.numerator;
	
	config.kf_max_dist = keyframeMaxDistance;
	
	if(forceKeyframes)
		config.kf_min_dist = config.kf_max_dist;
	
	applyCustomPre(config, custom);
	
	assert(config.kf_max_dist >= config.kf_min_dist);
	
		
	const aom_codec_flags_t flags = (config.g_bit_depth == AOM_BITS_8 ? 0 : AOM_CODEC_USE_HIGHBITDEPTH);
	
	aom_codec_err_t err = aom_codec_enc_init(&_encoder, iface, &config, flags);
		
	if(err == AOM_CODEC_OK)
	{
		if(method == WEBM_METHOD_CONSTANT_QUALITY || method == WEBM_METHOD_CONSTRAINED_QUALITY)
		{
			const int min_q = config.rc_min_quantizer;
			const int max_q = config.rc_max_quantizer;
			
			// CQ Level should be between min_q and max_q
			const int cq_level = (min_q + max_q) / 2;
		
			aom_codec_control(&_encoder, AOME_SET_CQ_LEVEL, cq_level);
		}
		
		aom_codec_control(&_encoder, AOME_SET_CPUUSED, 2); // much faster if we do this
		
		aom_codec_control(&_encoder, AV1E_SET_TILE_COLUMNS, mylog2(g_num_cpus)); // this gives us some multithreading
		aom_codec_control(&_encoder, AV1E_SET_FRAME_PARALLEL_DECODING, 1);
	
		applyCustomPost(custom);
		
		
		aom_fixed_buf_t *privateH = aom_codec_get_global_headers(&_encoder);
		
		if(privateH != NULL && privateH->buf != NULL && privateH->sz > 0)
		{
			_privateSize = privateH->sz;
			_privateData = malloc(_privateSize);
			if(_privateData == NULL)
				throw exportReturn_ErrMemory;
			memcpy(_privateData, privateH->buf, _privateSize);
		}
	}

	if(err != AOM_CODEC_OK)
		throw exportReturn_InternalError;
}

AOMEncoder::~AOMEncoder()
{
	assert(NULL == aom_codec_get_cx_data(&_encoder, &_iter));
	
	aom_codec_err_t err = aom_codec_destroy(&_encoder);
	
	assert(err == AOM_CODEC_OK);
	
	if(_privateData != NULL)
		free(_privateData);
}

void *
AOMEncoder::getPrivateData(size_t &size)
{
	size = _privateSize;
	
	return _privateData;
}

bool
AOMEncoder::compressFrame(const PPixHand &frame, PrTime time, PrTime duration)
{
	prRect bounds;
	_pixSuite->GetBounds(frame, &bounds);

	const csSDK_int32 width = (bounds.right - bounds.left);
	const csSDK_int32 height = (bounds.bottom - bounds.top);

	const aom_img_fmt_t imgfmt8 = _sampling == WEBM_444 ? AOM_IMG_FMT_I444 :
									_sampling == WEBM_422 ? AOM_IMG_FMT_I422 :
									AOM_IMG_FMT_I420;
									
	const aom_img_fmt_t imgfmt16 = _sampling == WEBM_444 ? AOM_IMG_FMT_I44416 :
									_sampling == WEBM_422 ? AOM_IMG_FMT_I42216 :
									AOM_IMG_FMT_I42016;
									
	const aom_img_fmt_t &imgfmt = (_bitDepth > 8 ? imgfmt16 : imgfmt8);

			
	aom_image_t img_data;
	aom_image_t *img = aom_img_alloc(&img_data, imgfmt, width, height, 32);
	
	aom_codec_err_t err = AOM_CODEC_OK;
	
	if(img)
	{
		if(_alpha)
		{
			img->cp = AOM_CICP_CP_UNSPECIFIED;
			img->tc = AOM_CICP_TC_UNSPECIFIED;
			img->mc = AOM_CICP_MC_UNSPECIFIED;
			img->monochrome = FALSE; // I know...
			img->csp = AOM_CSP_UNKNOWN;
			img->range = AOM_CR_FULL_RANGE;
		}
		else
		{
			img->cp = (_colorSpace == WEBM_REC709 ? AOM_CICP_CP_BT_709 : AOM_CICP_CP_BT_601);
			img->tc = (_colorSpace == WEBM_REC709 ? AOM_CICP_TC_BT_709 : AOM_CICP_TC_BT_601);
			img->mc = (_colorSpace == WEBM_REC709 ? AOM_CICP_MC_BT_709 : AOM_CICP_MC_BT_601);
			img->monochrome = FALSE;
			img->csp = AOM_CSP_UNKNOWN;
			img->range = AOM_CR_STUDIO_RANGE;
		}
	
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
		yuv.y = img->planes[AOM_PLANE_Y];
		yuv.u = img->planes[AOM_PLANE_U];
		yuv.v = img->planes[AOM_PLANE_V];
		yuv.yRowbytes = img->stride[AOM_PLANE_Y];
		yuv.uRowbytes = img->stride[AOM_PLANE_U];
		yuv.vRowbytes = img->stride[AOM_PLANE_V];
		
		CopyPixToBuffer(yuv, frame);
	
		err = aom_codec_encode(&_encoder, img, time, duration, 0);
		
		if(err == AOM_CODEC_OK)
			_iter = NULL;
		
		aom_img_free(img);
	}
	else
		return false;
	
	if(err == AOM_CODEC_OK)
		checkForPackets();
	
	return (err == AOM_CODEC_OK);
}

bool
AOMEncoder::endOfStream()
{
	aom_codec_err_t err = aom_codec_encode(&_encoder, NULL, 0, 1, 0);
	
	if(err == AOM_CODEC_OK)
	{
		_iter = NULL;
		
		checkForPackets();
	}
	
	return (err == AOM_CODEC_OK);
}

void
AOMEncoder::checkForPackets()
{
	while(const aom_codec_cx_pkt_t *pkt = aom_codec_get_cx_data(&_encoder, &_iter))
	{
		if(_vbrPass)
		{
			if(pkt->kind == AOM_CODEC_STATS_PKT)
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
			if(pkt->kind == AOM_CODEC_CX_FRAME_PKT)
			{
				Packet *packet = new Packet;
				
				packet->size = pkt->data.frame.sz;
				packet->data = malloc(packet->size);
				if(packet->data == NULL)
					throw exportReturn_ErrMemory;
				memcpy(packet->data, pkt->data.frame.buf, packet->size);
				packet->time = pkt->data.frame.pts;
				packet->duration = pkt->data.frame.duration;
				packet->keyframe = (pkt->data.frame.flags & AOM_FRAME_IS_KEY);
				
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
AOMEncoder::applyCustomPre(aom_codec_enc_cfg_t &config, const std::string &custom)
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
			
			if(arg == "-t" || arg == "--threads")
			{	SetValue(config.g_threads, val); i++;	}
			
			else if(arg == "--global-error-resilient")
			{	SetValue(config.g_error_resilient, val); i++;	}
			
			else if(arg == "--lag-in-frames")
			{	SetValue(config.g_lag_in_frames, val); i++;	}
			
			else if(arg == "--drop-frame")
			{	SetValue(config.rc_dropframe_thresh, val); i++;	}
			
			else if(arg == "--resize-mode")
			{	SetValue(config.rc_resize_mode, val); i++;	}
			
			else if(arg == "--resize-denominator")
			{	SetValue(config.rc_resize_denominator, val); i++;	}

			else if(arg == "--superres-kf-denominator")
			{	SetValue(config.rc_resize_kf_denominator, val); i++;	}

			else if(arg == "--superres-mode")
			{
				const aom_superres_mode mode = val == "0" ? AOM_SUPERRES_NONE :
												val == "1" ? AOM_SUPERRES_FIXED :
												val == "2" ? AOM_SUPERRES_RANDOM :
												val == "3" ? AOM_SUPERRES_QTHRESH :
												val == "4" ? AOM_SUPERRES_AUTO :
												AOM_SUPERRES_AUTO;
				
				config.rc_superres_mode = mode;
				
				i++;
			}

			else if(arg == "--superres-denominator")
			{	SetValue(config.rc_superres_denominator, val); i++;	}

			else if(arg == "--superres-kf-denominator")
			{	SetValue(config.rc_superres_kf_denominator, val); i++;	}
			
			else if(arg == "--superres-qthresh")
			{	SetValue(config.rc_superres_qthresh, val); i++;	}
			
			else if(arg == "--superres-kf-qthresh")
			{	SetValue(config.rc_superres_kf_qthresh, val); i++;	}
			
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

			else if(arg == "--enable-fwd-kf")
			{	SetValue(config.fwd_kf_enabled, val); i++;	}

			else if(arg == "--disable-kf")
			{	config.kf_mode = AOM_KF_DISABLED;	}
			
			else if(arg == "--kf-min-dist")
			{	SetValue(config.kf_min_dist, val); i++;	}

			else if(arg == "--kf-max-dist")
			{	SetValue(config.kf_max_dist, val); i++;	}

			else if(arg == "--sframe-dist")
			{	SetValue(config.sframe_dist, val); i++;	}

			else if(arg == "--sframe-mode")
			{	SetValue(config.sframe_mode, val); i++;	}
			
			else if(arg == "--large-scale-tile")
			{	SetValue(config.large_scale_tile, val); i++;	}

			else if(arg == "--annexb")
			{	SetValue(config.save_as_annexb, val); i++;	}

			else if(arg == "--use-fixed-qp-offsets")
			{	SetValue(config.use_fixed_qp_offsets, val); i++;	}

			i++;
		}
	}
}

#define ConfigureAOMValue(encoder, ctrl_id, s) \
	do{							\
		std::stringstream ss;	\
		ss << s;				\
		int v = 0;				\
		ss >> v;				\
		aom_codec_err_t err = aom_codec_control(encoder, ctrl_id, v); \
		if(err != AOM_CODEC_OK)	\
			config_err = err;	\
	}while(0)

void
AOMEncoder::applyCustomPost(const std::string &custom)
{
	std::vector<std::string> args;
	
	if(quotedTokenize(custom, args, " =\t\r\n") && args.size() > 0)
	{
		aom_codec_err_t config_err = AOM_CODEC_OK;
		
		const int num_args = args.size();

		args.push_back(""); // so there's always an i+1
		
		int i = 0;
		
		while(i < num_args)
		{
			const std::string &arg = args[i];
			const std::string &val = args[i + 1];
		
			if(arg == "--cpu-used")
			{	ConfigureAOMValue(&_encoder, AOME_SET_CPUUSED, val); i++;	}

			else if(arg == "--auto-alt-ref")
			{	ConfigureAOMValue(&_encoder, AOME_SET_ENABLEAUTOALTREF, val); i++;	}

			else if(arg == "--sharpness")
			{	ConfigureAOMValue(&_encoder, AOME_SET_SHARPNESS, val); i++;	}

			else if(arg == "--static-thresh")
			{	ConfigureAOMValue(&_encoder, AOME_SET_STATIC_THRESHOLD, val); i++;	}

			else if(arg == "--arnr-maxframes")
			{	ConfigureAOMValue(&_encoder, AOME_SET_ARNR_MAXFRAMES, val); i++;	}

			else if(arg == "--arnr-strength")
			{	ConfigureAOMValue(&_encoder, AOME_SET_ARNR_STRENGTH, val); i++;	}

			else if(arg == "--tune")
			{
				unsigned int ival = val == "psnr" ? AOM_TUNE_PSNR :
									val == "ssim" ? AOM_TUNE_SSIM :
									val == "vmaf_with_preprocessing" ? AOM_TUNE_VMAF_WITH_PREPROCESSING :
									val == "vmaf_without_preprocessing" ? AOM_TUNE_VMAF_WITHOUT_PREPROCESSING :
									val == "vmaf" ? AOM_TUNE_VMAF_MAX_GAIN :
									val == "vmaf_neg" ? AOM_TUNE_VMAF_NEG_MAX_GAIN :
									val == "butteraugli" ? AOM_TUNE_BUTTERAUGLI :
									val == "vmaf_saliency_map" ? AOM_TUNE_VMAF_SALIENCY_MAP :
									AOM_TUNE_PSNR;
			
				ConfigureAOMValue(&_encoder, AOME_SET_TUNING, ival);
				i++;
			}
			else if(arg == "--cq-level")
			{	ConfigureAOMValue(&_encoder, AOME_SET_CQ_LEVEL, val); i++;	}
			
			else if(arg == "--max-intra-rate")
			{	ConfigureAOMValue(&_encoder, AOME_SET_MAX_INTRA_BITRATE_PCT, val); i++;	}

			// AOME_SET_NUMBER_SPATIAL_LAYERS ?

			else if(arg == "--max-inter-rate")
			{	ConfigureAOMValue(&_encoder, AV1E_SET_MAX_INTER_BITRATE_PCT, val); i++;	}

			else if(arg == "--gf-cbr-boost")
			{	ConfigureAOMValue(&_encoder, AV1E_SET_GF_CBR_BOOST_PCT, val); i++;	}

			else if(arg == "--lossless")
			{	ConfigureAOMValue(&_encoder, AV1E_SET_LOSSLESS, 1);	}
			
			else if(arg == "--row-mt")
			{	ConfigureAOMValue(&_encoder, AV1E_SET_ROW_MT, 1);	}

			else if(arg == "--tile-columns")
			{	ConfigureAOMValue(&_encoder, AV1E_SET_TILE_COLUMNS, val); i++;	}

			else if(arg == "--tile-rows")
			{	ConfigureAOMValue(&_encoder, AV1E_SET_TILE_ROWS, val); i++;	}
			
			else if(arg == "--enable-tpl-model")
			{	ConfigureAOMValue(&_encoder, AV1E_SET_ENABLE_TPL_MODEL, val); i++;	}

			else if(arg == "--enable-keyframe-filtering")
			{	ConfigureAOMValue(&_encoder, AV1E_SET_ENABLE_KEYFRAME_FILTERING, val); i++;	}

			else if(arg == "--frame-parallel")
			{	ConfigureAOMValue(&_encoder, AV1E_SET_FRAME_PARALLEL_DECODING, val);	i++;	}

			else if(arg == "--error-resilient")
			{	ConfigureAOMValue(&_encoder, AV1E_SET_ERROR_RESILIENT_MODE, val);	i++;	}

			else if(arg == "--sframe-mode")
			{	ConfigureAOMValue(&_encoder, AV1E_SET_S_FRAME_MODE, val);	i++;	}

			else if(arg == "--aq-mode")
			{	ConfigureAOMValue(&_encoder, AV1E_SET_AQ_MODE, val);	i++;	}

			else if(arg == "--frame_boost")
			{	ConfigureAOMValue(&_encoder, AV1E_SET_FRAME_PERIODIC_BOOST, val); i++;	}

			else if(arg == "--noise-sensitivity")
			{	ConfigureAOMValue(&_encoder, AV1E_SET_NOISE_SENSITIVITY, val); i++;	}

			else if(arg == "--tune-content")
			{
				unsigned int ival = val == "default" ? AOM_CONTENT_DEFAULT :
									val == "screen" ? AOM_CONTENT_SCREEN :
									val == "film" ? AOM_CONTENT_FILM :
									AOM_CONTENT_DEFAULT;
			
				ConfigureAOMValue(&_encoder, AV1E_SET_TUNE_CONTENT, ival);
				i++;
			}

			else if(arg == "--cdf-update-mode")
			{	ConfigureAOMValue(&_encoder, AV1E_SET_CDF_UPDATE_MODE, val); i++;	}

			else if(arg == "--color-primaries")
			{
				unsigned int ival = val == "unspecified" ? AOM_CICP_CP_UNSPECIFIED :
									val == "bt601" ? AOM_CICP_CP_BT_601 :
									val == "bt709" ? AOM_CICP_CP_BT_709 :
									val == "bt470m" ? AOM_CICP_CP_BT_470_M :
									val == "bt470bg" ? AOM_CICP_CP_BT_470_B_G :
									val == "smpte240" ? AOM_CICP_CP_SMPTE_240 :
									val == "film" ? AOM_CICP_CP_GENERIC_FILM :
									val == "bt2020" ? AOM_CICP_CP_BT_2020 :
									val == "xyz" ? AOM_CICP_CP_XYZ :
									val == "smpte431" ? AOM_CICP_CP_SMPTE_431 :
									val == "smpte432" ? AOM_CICP_CP_SMPTE_432 :
									val == "ebu3213" ? AOM_CICP_CP_EBU_3213 :
									AOM_CICP_CP_UNSPECIFIED;
			
				ConfigureAOMValue(&_encoder, AV1E_SET_COLOR_PRIMARIES, ival);
				i++;
			}
			
			else if(arg == "--transfer-characteristics")
			{
				unsigned int ival = val == "unspecified" ? AOM_CICP_CP_UNSPECIFIED :
									val == "bt601" ? AOM_CICP_TC_BT_601 :
									val == "bt709" ? AOM_CICP_TC_BT_709 :
									val == "bt470m" ? AOM_CICP_TC_BT_470_M :
									val == "bt470bg" ? AOM_CICP_TC_BT_470_B_G :
									val == "smpte240" ? AOM_CICP_TC_SMPTE_240 :
									val == "lin" ? AOM_CICP_TC_LINEAR :
									val == "log100" ? AOM_CICP_TC_LOG_100 :
									val == "log100sq10" ? AOM_CICP_TC_LOG_100_SQRT10 :
									val == "iec61966" ? AOM_CICP_TC_IEC_61966 :
									val == "srgb" ? AOM_CICP_TC_SRGB :
									val == "bt2020-10bit" ? AOM_CICP_TC_BT_2020_10_BIT :
									val == "bt2020-12bit" ? AOM_CICP_TC_BT_2020_12_BIT :
									val == "smpte2084" ? AOM_CICP_TC_SMPTE_2084 :
									val == "hlg" ? AOM_CICP_TC_HLG :
									val == "smpte428" ? AOM_CICP_TC_SMPTE_428 :
									AOM_CICP_CP_UNSPECIFIED;
			
				ConfigureAOMValue(&_encoder, AV1E_SET_TRANSFER_CHARACTERISTICS, ival);
				i++;
			}
			
			else if(arg == "--matrix-coefficients")
			{
				unsigned int ival = val == "unspecified" ? AOM_CICP_MC_UNSPECIFIED :
									val == "bt601" ? AOM_CICP_MC_BT_601 :
									val == "bt709" ? AOM_CICP_MC_BT_709 :
									val == "identity" ? AOM_CICP_MC_IDENTITY :
									val == "fcc73" ? AOM_CICP_MC_FCC :
									val == "bt470bg" ? AOM_CICP_MC_BT_470_B_G :
									val == "smpte240" ? AOM_CICP_CP_SMPTE_240 :
									val == "ycgco" ? AOM_CICP_MC_SMPTE_YCGCO :
									val == "bt2020ncl" ? AOM_CICP_MC_BT_2020_NCL :
									val == "bt2020cl" ? AOM_CICP_MC_BT_2020_CL :
									val == "smpte2085" ? AOM_CICP_MC_SMPTE_2085 :
									val == "chromncl" ? AOM_CICP_MC_CHROMAT_NCL :
									val == "chromcl" ? AOM_CICP_MC_CHROMAT_CL :
									val == "ictcp" ? AOM_CICP_MC_ICTCP :
									AOM_CICP_MC_UNSPECIFIED;
			
				ConfigureAOMValue(&_encoder, AV1E_SET_MATRIX_COEFFICIENTS, ival);
				i++;
			}
			
			else if(arg == "--chroma-sample-position")
			{
				unsigned int ival = val == "unknown" ? AOM_CSP_UNKNOWN :
									val == "vertical" ? AOM_CSP_VERTICAL :
									val == "colocated" ? AOM_CSP_COLOCATED :
									AOM_CSP_UNKNOWN;
			
				ConfigureAOMValue(&_encoder, AV1E_SET_CHROMA_SAMPLE_POSITION, ival);
				i++;
			}
			
			else if(arg == "--min-gf-interval")
			{	ConfigureAOMValue(&_encoder, AV1E_SET_MIN_GF_INTERVAL, val); i++;	}
			
			else if(arg == "--max-gf-interval")
			{	ConfigureAOMValue(&_encoder, AV1E_SET_MAX_GF_INTERVAL, val); i++;	}

			else if(arg == "--color-range")
			{
				unsigned int ival = val == "studio" ? AOM_CR_STUDIO_RANGE :
									val == "full" ? AOM_CR_FULL_RANGE :
									AOM_CR_STUDIO_RANGE;
			
				ConfigureAOMValue(&_encoder, AV1E_SET_COLOR_RANGE, ival);
				i++;
			}

			else if(arg == "--target-seq-level-idx")
			{	ConfigureAOMValue(&_encoder, AV1E_SET_TARGET_SEQ_LEVEL_IDX, val); i++;	}

			else if(arg == "--sb-size")
			{
				unsigned int ival = val == "dynamic" ? AOM_SUPERBLOCK_SIZE_DYNAMIC :
									val == "64" ? AOM_SUPERBLOCK_SIZE_64X64 :
									val == "128" ? AOM_SUPERBLOCK_SIZE_128X128 :
									AOM_SUPERBLOCK_SIZE_DYNAMIC;
			
				ConfigureAOMValue(&_encoder, AV1E_SET_SUPERBLOCK_SIZE, ival);
				i++;
			}

			else if(arg == "--enable-cdef")
			{	ConfigureAOMValue(&_encoder, AV1E_SET_ENABLE_CDEF, val); i++;	}

			else if(arg == "--enable-restoration")
			{	ConfigureAOMValue(&_encoder, AV1E_SET_ENABLE_RESTORATION, val); i++;	}

			else if(arg == "--enable-obmc")
			{	ConfigureAOMValue(&_encoder, AV1E_SET_ENABLE_OBMC, val); i++;	}

			else if(arg == "--disable-trellis-quant")
			{	ConfigureAOMValue(&_encoder, AV1E_SET_DISABLE_TRELLIS_QUANT, val); i++;	}

			else if(arg == "--enable-qm")
			{	ConfigureAOMValue(&_encoder, AV1E_SET_ENABLE_QM, val); i++;	}

			else if(arg == "--qm-min")
			{	ConfigureAOMValue(&_encoder, AV1E_SET_QM_MIN, val); i++;	}

			else if(arg == "--qm-max")
			{	ConfigureAOMValue(&_encoder, AV1E_SET_QM_MAX, val); i++;	}

			else if(arg == "--num-tile-groups")
			{	ConfigureAOMValue(&_encoder, AV1E_SET_NUM_TG, val); i++;	}

			else if(arg == "--mtu-size")
			{	ConfigureAOMValue(&_encoder, AV1E_SET_MTU, val); i++;	}

			else if(arg == "--enable-rect-partitions")
			{	ConfigureAOMValue(&_encoder, AV1E_SET_ENABLE_RECT_PARTITIONS, val); i++;	}

			else if(arg == "--enable-ab-partitions")
			{	ConfigureAOMValue(&_encoder, AV1E_SET_ENABLE_AB_PARTITIONS, val); i++;	}

			else if(arg == "--enable-1to4-partitions")
			{	ConfigureAOMValue(&_encoder, AV1E_SET_ENABLE_1TO4_PARTITIONS, val); i++;	}

			else if(arg == "--min-partition-size")
			{	ConfigureAOMValue(&_encoder, AV1E_SET_MIN_PARTITION_SIZE, val); i++;	}

			else if(arg == "--max-partition-size")
			{	ConfigureAOMValue(&_encoder, AV1E_SET_MAX_PARTITION_SIZE, val); i++;	}

			else if(arg == "--enable-intra-edge-filter")
			{	ConfigureAOMValue(&_encoder, AV1E_SET_ENABLE_INTRA_EDGE_FILTER, val); i++;	}

			else if(arg == "--enable-order-hint")
			{	ConfigureAOMValue(&_encoder, AV1E_SET_ENABLE_ORDER_HINT, val); i++;	}

			else if(arg == "--enable-tx64")
			{	ConfigureAOMValue(&_encoder, AV1E_SET_ENABLE_TX64, val); i++;	}

			else if(arg == "--enable-flip-idtx")
			{	ConfigureAOMValue(&_encoder, AV1E_SET_ENABLE_FLIP_IDTX, val); i++;	}

			else if(arg == "--enable-rect-tx")
			{	ConfigureAOMValue(&_encoder, AV1E_SET_ENABLE_RECT_TX, val); i++;	}

			else if(arg == "--enable-dist-wtd-comp")
			{	ConfigureAOMValue(&_encoder, AV1E_SET_ENABLE_DIST_WTD_COMP, val); i++;	}

			else if(arg == "--enable-ref-frame-mvs")
			{	ConfigureAOMValue(&_encoder, AV1E_SET_ENABLE_REF_FRAME_MVS, val); i++;	}

			else if(arg == "--enable-dual-filter")
			{	ConfigureAOMValue(&_encoder, AV1E_SET_ENABLE_DUAL_FILTER, val); i++;	}

			else if(arg == "--enable-chroma-deltaq")
			{	ConfigureAOMValue(&_encoder, AV1E_SET_ENABLE_CHROMA_DELTAQ, val); i++;	}

			else if(arg == "--enable-masked-comp")
			{	ConfigureAOMValue(&_encoder, AV1E_SET_ENABLE_MASKED_COMP, val); i++;	}

			else if(arg == "--enable-onesided-comp")
			{	ConfigureAOMValue(&_encoder, AV1E_SET_ENABLE_ONESIDED_COMP, val); i++;	}

			else if(arg == "--enable-interintra-comp")
			{	ConfigureAOMValue(&_encoder, AV1E_SET_ENABLE_INTERINTRA_COMP, val); i++;	}

			else if(arg == "--enable-smooth-interintra")
			{	ConfigureAOMValue(&_encoder, AV1E_SET_ENABLE_SMOOTH_INTERINTRA, val); i++;	}

			else if(arg == "--enable-diff-wtd-comp")
			{	ConfigureAOMValue(&_encoder, AV1E_SET_ENABLE_DIFF_WTD_COMP, val); i++;	}

			else if(arg == "--enable-interinter-wedge")
			{	ConfigureAOMValue(&_encoder, AV1E_SET_ENABLE_INTERINTER_WEDGE, val); i++;	}

			else if(arg == "--enable-interintra-wedge")
			{	ConfigureAOMValue(&_encoder, AV1E_SET_ENABLE_INTERINTRA_WEDGE, val); i++;	}

			else if(arg == "--enable-global-motion")
			{	ConfigureAOMValue(&_encoder, AV1E_SET_ENABLE_GLOBAL_MOTION, val); i++;	}

			else if(arg == "--enable-warped-motion")
			{	ConfigureAOMValue(&_encoder, AV1E_SET_ENABLE_WARPED_MOTION, val); i++;	}

			else if(arg == "--enable-filter-intra")
			{	ConfigureAOMValue(&_encoder, AV1E_SET_ENABLE_FILTER_INTRA, val); i++;	}

			else if(arg == "--enable-smooth-intra")
			{	ConfigureAOMValue(&_encoder, AV1E_SET_ENABLE_SMOOTH_INTRA, val); i++;	}

			else if(arg == "--enable-paeth-intra")
			{	ConfigureAOMValue(&_encoder, AV1E_SET_ENABLE_PAETH_INTRA, val); i++;	}

			else if(arg == "--enable-cfl-intra")
			{	ConfigureAOMValue(&_encoder, AV1E_SET_ENABLE_CFL_INTRA, val); i++;	}

			else if(arg == "--enable-overlay")
			{	ConfigureAOMValue(&_encoder, AV1E_SET_ENABLE_OVERLAY, val); i++;	}

			else if(arg == "--enable-palette")
			{	ConfigureAOMValue(&_encoder, AV1E_SET_ENABLE_PALETTE, val); i++;	}

			else if(arg == "--enable-intrabc")
			{	ConfigureAOMValue(&_encoder, AV1E_SET_ENABLE_INTRABC, val); i++;	}

			else if(arg == "--enable-angle-delta")
			{	ConfigureAOMValue(&_encoder, AV1E_SET_ENABLE_ANGLE_DELTA, val); i++;	}

			else if(arg == "--deltaq-mode")
			{	ConfigureAOMValue(&_encoder, AV1E_SET_DELTAQ_MODE, val); i++;	}

			else if(arg == "--delta-lf-mode")
			{	ConfigureAOMValue(&_encoder, AV1E_SET_DELTALF_MODE, val); i++;	}

			else if(arg == "--timing-info")
			{
				unsigned int ival = val == "unspecified" ? AOM_TIMING_UNSPECIFIED :
									val == "constant" ? AOM_TIMING_EQUAL :
									val == "model" ? AOM_TIMING_DEC_MODEL :
									AOM_TIMING_UNSPECIFIED;
			
				ConfigureAOMValue(&_encoder, AV1E_SET_TIMING_INFO_TYPE, ival);
				i++;
			}

			else if(arg == "--film-grain-test")
			{	ConfigureAOMValue(&_encoder, AV1E_SET_FILM_GRAIN_TEST_VECTOR, val); i++;	}
			
			// AV1E_SET_FILM_GRAIN_TABLE - a path?
			
			else if(arg == "--denoise-noise-level")
			{	ConfigureAOMValue(&_encoder, AV1E_SET_DENOISE_NOISE_LEVEL, val); i++;	}
						
			else if(arg == "--denoise-block-size")
			{	ConfigureAOMValue(&_encoder, AV1E_SET_DENOISE_BLOCK_SIZE, val); i++;	}
						
			else if(arg == "--reduced-tx-type-set")
			{	ConfigureAOMValue(&_encoder, AV1E_SET_REDUCED_TX_TYPE_SET, val); i++;	}
						
			else if(arg == "--use-intra-dct-only")
			{	ConfigureAOMValue(&_encoder, AV1E_SET_INTRA_DCT_ONLY, val); i++;	}
						
			else if(arg == "--use-inter-dct-only")
			{	ConfigureAOMValue(&_encoder, AV1E_SET_INTER_DCT_ONLY, val); i++;	}
						
			else if(arg == "--use-intra-default-tx-only")
			{	ConfigureAOMValue(&_encoder, AV1E_SET_INTRA_DEFAULT_TX_ONLY, val); i++;	}
						
			else if(arg == "--quant-b-adapt")
			{	ConfigureAOMValue(&_encoder, AV1E_SET_QUANT_B_ADAPT, val); i++;	}
						
			else if(arg == "--gf-min-pyr-height")
			{	ConfigureAOMValue(&_encoder, AV1E_SET_GF_MIN_PYRAMID_HEIGHT, val); i++;	}
						
			else if(arg == "--gf-max-pyr-height")
			{	ConfigureAOMValue(&_encoder, AV1E_SET_GF_MAX_PYRAMID_HEIGHT, val); i++;	}
						
			else if(arg == "--max-reference-frames")
			{	ConfigureAOMValue(&_encoder, AV1E_SET_MAX_REFERENCE_FRAMES, val); i++;	}
						
			else if(arg == "--reduced-reference-set")
			{	ConfigureAOMValue(&_encoder, AV1E_SET_REDUCED_REFERENCE_SET, val); i++;	}
						
			else if(arg == "--coeff-cost-upd-freq")
			{	ConfigureAOMValue(&_encoder, AV1E_SET_COEFF_COST_UPD_FREQ, val); i++;	}
						
			else if(arg == "--mode-cost-upd-freq")
			{	ConfigureAOMValue(&_encoder, AV1E_SET_MODE_COST_UPD_FREQ, val); i++;	}
						
			else if(arg == "--mv-cost-upd-freq")
			{	ConfigureAOMValue(&_encoder, AV1E_SET_MV_COST_UPD_FREQ, val); i++;	}
						
			else if(arg == "--set-tier-mask")
			{	ConfigureAOMValue(&_encoder, AV1E_SET_TIER_MASK, val); i++;	}
						
			else if(arg == "--min-cr")
			{	ConfigureAOMValue(&_encoder, AV1E_SET_MIN_CR, val); i++;	}
						
			else if(arg == "--vbr-corpus-complexity-lap")
			{	ConfigureAOMValue(&_encoder, AV1E_SET_VBR_CORPUS_COMPLEXITY_LAP, val); i++;	}
						
			else if(arg == "--enable-dnl-denoising")
			{	ConfigureAOMValue(&_encoder, AV1E_SET_ENABLE_DNL_DENOISING, val); i++;	}
						
			else if(arg == "--enable-diagonal-intra")
			{	ConfigureAOMValue(&_encoder, AV1E_SET_ENABLE_DIAGONAL_INTRA, val); i++;	}
						
			else if(arg == "--dv-cost-upd-freq")
			{	ConfigureAOMValue(&_encoder, AV1E_SET_DV_COST_UPD_FREQ, val); i++;	}
			
			// AV1E_SET_PARTITION_INFO_PATH - path?
			
			else if(arg == "--enable-directional-intra")
			{	ConfigureAOMValue(&_encoder, AV1E_SET_ENABLE_DIRECTIONAL_INTRA, val); i++;	}
									
			else if(arg == "--enable-tx-size-search")
			{	ConfigureAOMValue(&_encoder, AV1E_SET_ENABLE_TX_SIZE_SEARCH, val); i++;	}
									
			else if(arg == "--deltaq-strength")
			{	ConfigureAOMValue(&_encoder, AV1E_SET_DELTAQ_STRENGTH, val); i++;	}
									
			else if(arg == "--loopfilter-control")
			{	ConfigureAOMValue(&_encoder, AV1E_SET_LOOPFILTER_CONTROL, val); i++;	}
									
			else if(arg == "--auto-intra-tools-off")
			{	ConfigureAOMValue(&_encoder, AV1E_SET_AUTO_INTRA_TOOLS_OFF, val); i++;	}
									
			else if(arg == "--fp-mt")
			{	ConfigureAOMValue(&_encoder, AV1E_SET_FP_MT, val); i++;	}
									
			else if(arg == "--sb-qp-sweep")
			{	ConfigureAOMValue(&_encoder, AV1E_ENABLE_SB_QP_SWEEP, val); i++;	}
									
			else if(arg == "--enable-rate-guide-deltaq")
			{	ConfigureAOMValue(&_encoder, AV1E_ENABLE_RATE_GUIDE_DELTAQ, val); i++;	}
			
			else if(arg == "--rate-distribution-info")
			{	ConfigureAOMValue(&_encoder, AV1E_SET_RATE_DISTRIBUTION_INFO, val); i++;	}

			i++;
		}
		
		assert(config_err == AOM_CODEC_OK);
	}
}
