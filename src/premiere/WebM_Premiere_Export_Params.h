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


#ifndef WEBM_PREMIERE_EXPORT_PARAMS_H
#define WEBM_PREMIERE_EXPORT_PARAMS_H


#include "WebM_Premiere_Export.h"


typedef enum {
	WEBM_CODEC_VP8 = 0,
	WEBM_CODEC_VP9,
	WEBM_CODEC_AV1
} WebM_Video_Codec;

typedef enum {
	VP9_CODEC_AUTO = 0,
	VP9_CODEC_LIBVPX,
	VP9_CODEC_VPL
} VP9_Codec;

typedef enum {
	AV1_CODEC_AUTO = 0,
	AV1_CODEC_AOM,
	AV1_CODEC_SVT_AV1,
	AV1_CODEC_NVENC,
	AV1_CODEC_VPL
} AV1_Codec;

typedef enum {
	WEBM_METHOD_CONSTANT_QUALITY = 0,
	WEBM_METHOD_BITRATE,
	WEBM_METHOD_VBR,
	WEBM_METHOD_CONSTRAINED_QUALITY
} WebM_Video_Method;

typedef enum {
	WEBM_420 = 0,
	WEBM_422,
	WEBM_444
} WebM_Chroma_Sampling;

typedef enum {
	WEBM_REC601 = 0,
	WEBM_REC709
} WebM_ColorSpace;


#define WebMPluginVersion				"WebMPluginVersion"

#define WebMVideoCodec					"WebMVideoCodec"
#define WebMVP9Codec					"WebMVP9Codec"
#define WebMAV1Codec					"WebMAV1Codec"
#define WebMVideoMethod					"WebMVideoMethod"
#define WebMVideoQuality				"WebMVideoQuality"
#define WebMVideoBitrate				"WebMVideoBitrate"
#define WebMVideoTwoPass				"WebMVideoTwoPass"
#define WebMVideoKeyframeMaxDistance	"WebMVideoKeyframeMaxDistance"
#define WebMVideoSampling				"WebMVideoSampling"
#define WebMVideoBitDepth				"WebMVideoBitDepth"
#define WebMColorSpace					"WebMColorSpace"

#define WebMCustomGroup					"WebMCustomGroup"
#define WebMCustomArgs					"WebMCustomArgs"

#ifndef ADBEVideoAlpha
#define ADBEVideoAlpha					"ADBEVideoAlpha"
#endif


typedef enum {
	WEBM_CODEC_VORBIS = 0,
	WEBM_CODEC_OPUS
} WebM_Audio_Codec;

typedef enum {
	OGG_QUALITY = 0,
	OGG_BITRATE
} Ogg_Method;


#define WebMAudioCodec		"WebMAudioCodec"

#define WebMAudioMethod		"WebMAudioMethod"
#define WebMAudioQuality	"WebMAudioQuality"
#define WebMAudioBitrate	"WebMAudioBitrate"

#define WebMOpusAutoBitrate	"WebMOpusAutoBitrate"
#define WebMOpusBitrate		"WebMOpusBitrate"


prMALError
exSDKQueryOutputSettings(
	exportStdParms				*stdParmsP,
	exQueryOutputSettingsRec	*outputSettingsP);

prMALError
exSDKGenerateDefaultParams(
	exportStdParms				*stdParms, 
	exGenerateDefaultParamRec	*generateDefaultParamRec);

prMALError
exSDKPostProcessParams(
	exportStdParms			*stdParmsP, 
	exPostProcessParamsRec	*postProcessParamsRecP,
	bool haveNVENC,
	bool haveVPXVPL,
	bool haveAV1VPL);

prMALError
exSDKGetParamSummary(
	exportStdParms			*stdParmsP, 
	exParamSummaryRec		*summaryRecP);

prMALError
exSDKValidateParamChanged (
	exportStdParms		*stdParmsP, 
	exParamChangedRec	*validateParamChangedRecP);
	

#endif // WEBM_PREMIERE_EXPORT_PARAMS_H
