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


#include "WebM_Premiere_Codecs.h"

#include "WebM_Premiere_libvpx.h"
#include "WebM_Premiere_aom.h"
#include "WebM_Premiere_SVT-AV1.h"
#ifdef WEBM_HAVE_NVENC
#include "WebM_Premiere_NVENC.h"
#endif

#include "WebM_Premiere_Vorbis.h"
#include "WebM_Premiere_Opus.h"


#include <assert.h>



VideoEncoder::VideoEncoder(PrSDKPPixSuite *pixSuite, PrSDKPPix2Suite *pix2Suite, bool alpha) :
	_pixSuite(pixSuite),
	_pix2Suite(pix2Suite),
	_alpha(alpha)
{

}

VideoEncoder::~VideoEncoder()
{
	assert(_queue.empty());
}

const VideoEncoder::Packet *
VideoEncoder::getPacket()
{
	if(_queue.empty())
	{
		assert(false);
		
		return nullptr;
	}
	else
		return _queue.front();
}

void
VideoEncoder::returnPacket(const Packet *packet)
{
	if(packet == _queue.front())
	{
		free(packet->data);
		
		delete packet;
		
		_queue.pop();
	}
	else
		assert(false);
}

void
VideoEncoder::CopyPixToBuffer(const YUVBuffer &buf, const PPixHand &pix)
{
	prRect bounds;
	_pixSuite->GetBounds(pix, &bounds);
	
	assert(buf.width == (bounds.right - bounds.left));
	assert(buf.height == (bounds.bottom - bounds.top));
	
	PrPixelFormat pixFormat;
	_pixSuite->GetPixelFormat(pix, &pixFormat);
		
	if(pixFormat == PrPixelFormat_YUV_420_MPEG2_FRAME_PICTURE_PLANAR_8u_601 ||
		pixFormat == PrPixelFormat_YUV_420_MPEG2_FRAME_PICTURE_PLANAR_8u_709)
	{
		assert(buf.sampling == WEBM_420);
		assert(buf.bitDepth == 8);
		assert(buf.fullRange == false);
		assert(!_alpha);
		assert((buf.colorSpace == WEBM_REC601 && pixFormat == PrPixelFormat_YUV_420_MPEG2_FRAME_PICTURE_PLANAR_8u_601) ||
				(buf.colorSpace == WEBM_REC709 && pixFormat == PrPixelFormat_YUV_420_MPEG2_FRAME_PICTURE_PLANAR_8u_709));
		
		char *Y_PixelAddress, *U_PixelAddress, *V_PixelAddress;
		csSDK_uint32 Y_RowBytes, U_RowBytes, V_RowBytes;
		
		_pix2Suite->GetYUV420PlanarBuffers(pix, PrPPixBufferAccess_ReadOnly,
											&Y_PixelAddress, &Y_RowBytes,
											&U_PixelAddress, &U_RowBytes,
											&V_PixelAddress, &V_RowBytes);
		
		for(int y = 0; y < buf.height; y++)
		{
			unsigned char *bufY = buf.y + (buf.yRowbytes * y);
			
			const unsigned char *prY = (unsigned char *)Y_PixelAddress + (Y_RowBytes * y);
			
			memcpy(bufY, prY, buf.width * sizeof(unsigned char));
		}
		
		const int chroma_width = (buf.width / 2) + (buf.width % 2);
		const int chroma_height = (buf.height / 2) + (buf.height % 2);
		
		for(int y = 0; y < chroma_height; y++)
		{
			unsigned char *imgU = buf.u + (buf.uRowbytes * y);
			unsigned char *imgV = buf.v + (buf.vRowbytes * y);
			
			const unsigned char *prU = (unsigned char *)U_PixelAddress + (U_RowBytes * y);
			const unsigned char *prV = (unsigned char *)V_PixelAddress + (V_RowBytes * y);
			
			memcpy(imgU, prU, chroma_width * sizeof(unsigned char));
			memcpy(imgV, prV, chroma_width * sizeof(unsigned char));
		}
	}
	else
	{
		char *frameBufferP = NULL;
		csSDK_int32 rowbytes = 0;
		
		_pixSuite->GetPixels(pix, PrPPixBufferAccess_ReadOnly, &frameBufferP);
		_pixSuite->GetRowBytes(pix, &rowbytes);
		
		
		if(pixFormat == PrPixelFormat_UYVY_422_8u_601 ||
			pixFormat == PrPixelFormat_UYVY_422_8u_709)
		{
			assert(buf.sampling == WEBM_422);
			assert(buf.bitDepth == 8);
			assert(buf.fullRange == false);
			assert(!_alpha);
			assert((buf.colorSpace == WEBM_REC601 && pixFormat == PrPixelFormat_UYVY_422_8u_601) ||
					(buf.colorSpace == WEBM_REC709 && pixFormat == PrPixelFormat_UYVY_422_8u_709));
			
			for(int y = 0; y < buf.height; y++)
			{
				unsigned char *imgY = buf.y + (buf.yRowbytes * y);
				unsigned char *imgU = buf.u + (buf.uRowbytes * y);
				unsigned char *imgV = buf.v + (buf.vRowbytes * y);
			
				const unsigned char *prUYVY = (unsigned char *)frameBufferP + (rowbytes * y);
				
				for(int x=0; x < buf.width; x++)
				{
					if(x % 2 == 0)
						*imgU++ = *prUYVY++;
					else
						*imgV++ = *prUYVY++;
					
					*imgY++ = *prUYVY++;;
				}
			}
		}
		else if(pixFormat == PrPixelFormat_VUYX_4444_8u ||
				pixFormat == PrPixelFormat_VUYX_4444_8u_709)
		{
			assert(buf.sampling == WEBM_444);
			assert(buf.bitDepth == 8);
			assert(buf.fullRange == false);
			assert(!_alpha);
			assert((buf.colorSpace == WEBM_REC601 && pixFormat == PrPixelFormat_VUYX_4444_8u) ||
					(buf.colorSpace == WEBM_REC709 && pixFormat == PrPixelFormat_VUYX_4444_8u_709));
			
			CopyVUYAToBuffer<uint8_t, uint8_t>(buf, (uint8_t *)frameBufferP, rowbytes);
		}
		else if(pixFormat == PrPixelFormat_VUYA_4444_16u)
		{
			assert(buf.bitDepth > 8);
			assert(buf.colorSpace == WEBM_REC601);
			
			CopyVUYAToBuffer<uint16_t, uint16_t>(buf, (uint8_t *)frameBufferP, rowbytes);
		}
		else if(pixFormat == PrPixelFormat_BGRA_4444_16u || pixFormat == PrPixelFormat_BGRX_4444_16u)
		{
			assert(pixFormat == PrPixelFormat_BGRA_4444_16u || !_alpha);
		
			if(buf.bitDepth > 8)
				CopyBGRAToBuffer<uint16_t, uint16_t, false>(buf, (uint8_t *)frameBufferP, rowbytes);
			else
				CopyBGRAToBuffer<uint16_t, uint8_t, false>(buf, (uint8_t *)frameBufferP, rowbytes);
		}
		else if(pixFormat == PrPixelFormat_BGRA_4444_8u || pixFormat == PrPixelFormat_BGRX_4444_8u)
		{
			assert(pixFormat == PrPixelFormat_BGRA_4444_8u || !_alpha);
		
			if(buf.bitDepth > 8)
				CopyBGRAToBuffer<uint8_t, uint16_t, false>(buf, (uint8_t *)frameBufferP, rowbytes);
			else
				CopyBGRAToBuffer<uint8_t, uint8_t, false>(buf, (uint8_t *)frameBufferP, rowbytes);
		}
		else if(pixFormat == PrPixelFormat_ARGB_4444_8u || pixFormat == PrPixelFormat_XRGB_4444_8u)
		{
			assert(pixFormat == PrPixelFormat_ARGB_4444_8u || !_alpha);
		
			if(buf.bitDepth > 8)
				CopyBGRAToBuffer<uint8_t, uint16_t, true>(buf, (uint8_t *)frameBufferP, rowbytes);
			else
				CopyBGRAToBuffer<uint8_t, uint8_t, true>(buf, (uint8_t *)frameBufferP, rowbytes);
		}
		else
			assert(false);
	}
}

// converting from Adobe 16-bit to regular 16-bit
#define PF_HALF_CHAN16			16384

static inline uint16_t
Promote(const uint16_t &val)
{
	return (val > PF_HALF_CHAN16 ? ( (val - 1) << 1 ) + 1 : val << 1);
}


template <typename BGRA_PIX, typename IMG_PIX>
static inline IMG_PIX
DepthConvert(const BGRA_PIX &val, const int &depth);

template<>
static inline uint16_t
DepthConvert<uint16_t, uint16_t>(const uint16_t &val, const int &depth)
{
	return (Promote(val) >> (16 - depth));
}

template<>
static inline uint16_t
DepthConvert<uint8_t, uint16_t>(const uint8_t &val, const int &depth)
{
	return ((unsigned short)val << (depth - 8)) | (val >> (16 - depth));
}

template<>
static inline uint8_t
DepthConvert<uint16_t, uint8_t>(const uint16_t &val, const int &depth)
{
	assert(depth == 8);
	return ( (((long)(val) * 255) + 16384) / 32768);
}

template<>
static inline uint8_t
DepthConvert<uint8_t, uint8_t>(const unsigned char &val, const int &depth)
{
	assert(depth == 8);
	return val;
}

template <typename VUYA_PIX, typename BUF_PIX>
void
VideoEncoder::CopyVUYAToBuffer(const YUVBuffer &buf, const uint8_t *frameBufferP, ptrdiff_t rowbytes)
{
	const unsigned int sub_x = (buf.sampling == WEBM_444 ? 1 : 2);
	const unsigned int sub_y = (buf.sampling == WEBM_420 ? 2 : 1);
	
	if(_alpha)
	{
		for(int y = 0; y < buf.height; y++)
		{
			BUF_PIX *imgY = (BUF_PIX *)(buf.y + (buf.yRowbytes * y));
			BUF_PIX *imgU = (BUF_PIX *)(buf.u + (buf.uRowbytes * (y / sub_y)));
			BUF_PIX *imgV = (BUF_PIX *)(buf.v + (buf.vRowbytes * (y / sub_y)));
		
			const VUYA_PIX *prVUYA = (VUYA_PIX *)(frameBufferP + (rowbytes * (buf.height - 1 - y)));
			
			const VUYA_PIX *prA = prVUYA + 3;
			
			for(int x=0; x < buf.width; x++)
			{
				*imgY++ = DepthConvert<VUYA_PIX, BUF_PIX>(*prA, buf.bitDepth);
				
				if( (y % sub_y == 0) && (x % sub_x == 0) )
				{
					*imgU++ = DepthConvert<VUYA_PIX, BUF_PIX>(128, buf.bitDepth);
					*imgV++ = DepthConvert<VUYA_PIX, BUF_PIX>(128, buf.bitDepth);
				}
				
				prA += 4;
			}
		}
	}
	else
	{
		for(int y = 0; y < buf.height; y++)
		{
			BUF_PIX *imgY = (BUF_PIX *)(buf.y + (buf.yRowbytes * y));
			BUF_PIX *imgU = (BUF_PIX *)(buf.u + (buf.uRowbytes * (y / sub_y)));
			BUF_PIX *imgV = (BUF_PIX *)(buf.v + (buf.vRowbytes * (y / sub_y)));
		
			const VUYA_PIX *prVUYA = (VUYA_PIX *)(frameBufferP + (rowbytes * (buf.height - 1 - y)));
			
			const VUYA_PIX *prV = prVUYA + 0;
			const VUYA_PIX *prU = prVUYA + 1;
			const VUYA_PIX *prY = prVUYA + 2;
			
			for(int x=0; x < buf.width; x++)
			{
				*imgY++ = DepthConvert<VUYA_PIX, BUF_PIX>(*prY, buf.bitDepth);
				
				if( (y % sub_y == 0) && (x % sub_x == 0) )
				{
					*imgU++ = DepthConvert<VUYA_PIX, BUF_PIX>(*prU, buf.bitDepth);
					*imgV++ = DepthConvert<VUYA_PIX, BUF_PIX>(*prV, buf.bitDepth);
				}
				
				prY += 4;
				prU += 4;
				prV += 4;
			}
		}
	}
}

template <typename BGRA_PIX, typename BUF_PIX, bool isARGB>
void
VideoEncoder::CopyBGRAToBuffer(const YUVBuffer &buf, const uint8_t *frameBufferP, ptrdiff_t rowbytes)
{
	const unsigned int sub_x = (buf.sampling == WEBM_444 ? 1 : 2);
	const unsigned int sub_y = (buf.sampling == WEBM_420 ? 2 : 1);

	if(_alpha)
	{
		for(int y = 0; y < buf.height; y++)
		{
			BUF_PIX *imgY = (BUF_PIX *)(buf.y + (buf.yRowbytes * y));
			BUF_PIX *imgU = (BUF_PIX *)(buf.u + (buf.uRowbytes * (y / sub_y)));
			BUF_PIX *imgV = (BUF_PIX *)(buf.v + (buf.vRowbytes * (y / sub_y)));
			
			const BGRA_PIX *prBGRA = (BGRA_PIX *)(frameBufferP + (rowbytes * (buf.height - 1 - y)));
			
			const BGRA_PIX *prA = prBGRA + 3;
			
			if(isARGB)
			{
				prA = prBGRA + 0;
			}
			
			assert(buf.fullRange == true);
			
			for(int x=0; x < buf.width; x++)
			{
				*imgY++ = DepthConvert<BGRA_PIX, BUF_PIX>(*prA, buf.bitDepth);
				
				if( (y % sub_y == 0) && (x % sub_x == 0) )
				{
					*imgV++ = DepthConvert<uint8_t, BUF_PIX>(128, buf.bitDepth);
					*imgU++ = DepthConvert<uint8_t, BUF_PIX>(128, buf.bitDepth);
				}
				
				prA += 4;
			}
		}
	}
	else
	{
		for(int y = 0; y < buf.height; y++)
		{
			BUF_PIX *imgY = (BUF_PIX *)(buf.y + (buf.yRowbytes * y));
			BUF_PIX *imgU = (BUF_PIX *)(buf.u + (buf.uRowbytes * (y / sub_y)));
			BUF_PIX *imgV = (BUF_PIX *)(buf.v + (buf.vRowbytes * (y / sub_y)));
			
			const BGRA_PIX *prBGRA = (BGRA_PIX *)(frameBufferP + (rowbytes * (buf.height - 1 - y)));
			
			const BGRA_PIX *prB = prBGRA + 0;
			const BGRA_PIX *prG = prBGRA + 1;
			const BGRA_PIX *prR = prBGRA + 2;
			
			if(isARGB)
			{
				// Media Encoder CS5 insists on handing us this format in some cases,
				// even though we didn't list it as an option
				prR = prBGRA + 1;
				prG = prBGRA + 2;
				prB = prBGRA + 3;
			}
			
			// These are the pixels below the current one for MPEG-2 chroma siting
			const BGRA_PIX *prBb = prB - (rowbytes / sizeof(BGRA_PIX));
			const BGRA_PIX *prGb = prG - (rowbytes / sizeof(BGRA_PIX));
			const BGRA_PIX *prRb = prR - (rowbytes / sizeof(BGRA_PIX));
			
			// unless this is the last line and there is no pixel below
			if(y == (buf.height - 1) || sub_y != 2)
			{
				prBb = prB;
				prGb = prG;
				prRb = prR;
			}
			
			assert(buf.fullRange == false);
			
			// these are part of the RGBtoYUV math (uses Adobe 16-bit)
			const int Yadd = (sizeof(BGRA_PIX) > 1 ? 20565000 : 165000);    // to be divided by 10000
			const int UVadd = (sizeof(BGRA_PIX) > 1 ? 164495000 : 1285000); // includes extra 5000 for rounding
			
			if(buf.colorSpace == WEBM_REC709)
			{
				// https://en.wikipedia.org/wiki/YCbCr#ITU-R_BT.709_conversion
				
				for(int x=0; x < buf.width; x++)
				{
					*imgY++ = DepthConvert<BGRA_PIX, BUF_PIX>( ((1826 * (int)*prR) + (6142 * (int)*prG) + ( 620 * (int)*prB) + Yadd) / 10000, buf.bitDepth);
					
					if(sub_y > 1)
					{
						if( (y % sub_y == 0) && (x % sub_x == 0) )
						{
							*imgU++ = DepthConvert<BGRA_PIX, BUF_PIX>( ((-(1007 * (int)*prR) - (3385 * (int)*prG) + (4392 * (int)*prB) + UVadd) +
												(-(1007 * (int)*prRb) - (3385 * (int)*prGb) + (4392 * (int)*prBb) + UVadd)) / 20000, buf.bitDepth);
							*imgV++ = DepthConvert<BGRA_PIX, BUF_PIX>( (((4392 * (int)*prR) - (3990 * (int)*prG) - ( 402 * (int)*prB) + UVadd) +
												((4392 * (int)*prRb) - (3990 * (int)*prGb) - ( 402 * (int)*prBb) + UVadd)) / 20000, buf.bitDepth);
						}
						
						prRb += 4;
						prGb += 4;
						prBb += 4;
					}
					else
					{
						if(x % sub_x == 0)
						{
							*imgU++ = DepthConvert<BGRA_PIX, BUF_PIX>( ((-(1007 * (int)*prR) - (3385 * (int)*prG) + (4392 * (int)*prB) + UVadd) ) / 10000, buf.bitDepth);
							*imgV++ = DepthConvert<BGRA_PIX, BUF_PIX>( (((4392 * (int)*prR) - (3990 * (int)*prG) - ( 402 * (int)*prB) + UVadd)) / 10000, buf.bitDepth);
						}
					}
					
					prR += 4;
					prG += 4;
					prB += 4;
				}
			}
			else
			{
				assert(buf.colorSpace == WEBM_REC601);
			
				// using the conversion found here: http://www.fourcc.org/fccyvrgb.php
				// and 601 spec here: http://www.itu.int/dms_pubrec/itu-r/rec/bt/R-REC-BT.601-7-201103-I!!PDF-E.pdf
				
				for(int x=0; x < buf.width; x++)
				{
					*imgY++ = DepthConvert<BGRA_PIX, BUF_PIX>( ((2568 * (int)*prR) + (5041 * (int)*prG) + ( 979 * (int)*prB) + Yadd) / 10000, buf.bitDepth);
					
					if(sub_y > 1)
					{
						if( (y % sub_y == 0) && (x % sub_x == 0) )
						{
							*imgU++ = DepthConvert<BGRA_PIX, BUF_PIX>( ((-(1482 * (int)*prR) - (2910 * (int)*prG) + (4392 * (int)*prB) + UVadd) +
												(-(1482 * (int)*prRb) - (2910 * (int)*prGb) + (4392 * (int)*prBb) + UVadd)) / 20000, buf.bitDepth);
							*imgV++ = DepthConvert<BGRA_PIX, BUF_PIX>( (((4392 * (int)*prR) - (3678 * (int)*prG) - ( 714 * (int)*prB) + UVadd) +
												((4392 * (int)*prRb) - (3678 * (int)*prGb) - ( 714 * (int)*prBb) + UVadd)) / 20000, buf.bitDepth);
						}
						
						prRb += 4;
						prGb += 4;
						prBb += 4;
					}
					else
					{
						if(x % sub_x == 0)
						{
							*imgU++ = DepthConvert<BGRA_PIX, BUF_PIX>( ((-(1482 * (int)*prR) - (2910 * (int)*prG) + (4392 * (int)*prB) + UVadd) ) / 10000, buf.bitDepth);
							*imgV++ = DepthConvert<BGRA_PIX, BUF_PIX>( (((4392 * (int)*prR) - (3678 * (int)*prG) - ( 714 * (int)*prB) + UVadd)) / 10000, buf.bitDepth);
						}
					}
					
					prR += 4;
					prG += 4;
					prB += 4;
				}
			}
		}
	}
}

bool
VideoEncoder::quotedTokenize(const std::string &str,
				  std::vector<std::string> &tokens,
				  const std::string &delimiters)
{
	// this function will respect quoted strings when tokenizing
	// the quotes will be included in the returned strings
	
	int i = 0;
	bool in_quotes = false;
	
	// if there are un-quoted delimiters in the beginning, skip them
	while(i < str.size() && str[i] != '\"' && std::string::npos != delimiters.find(str[i]) )
		i++;
	
	std::string::size_type lastPos = i;
	
	while(i < str.size())
	{
		if(str[i] == '\"' && (i == 0 || str[i-1] != '\\'))
			in_quotes = !in_quotes;
		else if(!in_quotes)
		{
			if( std::string::npos != delimiters.find(str[i]) )
			{
				tokens.push_back(str.substr(lastPos, i - lastPos));
				
				lastPos = i + 1;
				
				// if there are more delimiters ahead, push forward
				while(lastPos < str.size() && (str[lastPos] != '\"' || str[lastPos-1] != '\\') && std::string::npos != delimiters.find(str[lastPos]) )
					lastPos++;
					
				i = lastPos;
				continue;
			}
		}
		
		i++;
	}
	
	if(in_quotes)
		return false;
	
	// we're at the end, was there anything left?
	if(str.size() - lastPos > 0)
		tokens.push_back( str.substr(lastPos) );
	
	return true;
}

void
VideoEncoder::initialize()
{
#ifdef WEBM_HAVE_NVENC
	NVENCEncoder::initialize();
#endif
}

bool
VideoEncoder::haveCodec(AV1_Codec av1Codec)
{
	if(av1Codec == AV1_CODEC_NVENC)
	{
	#ifdef WEBM_HAVE_NVENC
		return NVENCEncoder::available();
	#else
		return false;
	#endif
	}
	else
		return true;
}

typedef enum {
	LIBVPX,
	AOM,
	SVT_AV1,
	NVENC
} VideoEncoderLibrary;

static VideoEncoderLibrary
WhichVideoEncoder(WebM_Video_Codec codec, AV1_Codec av1Codec, WebM_Video_Method method, WebM_Chroma_Sampling sampling, int bitDepth, uint32_t width, uint32_t height, bool alpha)
{
	if(codec == WEBM_CODEC_AV1)
	{	
		if(av1Codec == AV1_CODEC_AUTO)
		{
			VideoEncoderLibrary lib = (VideoEncoder::haveCodec(AV1_CODEC_NVENC) ? NVENC : SVT_AV1);

			if(lib == NVENC)
			{
				if(sampling != WEBM_420 || bitDepth > 10)
					lib = SVT_AV1;
			}

			if(lib == SVT_AV1)
			{
				if(sampling != WEBM_420 || bitDepth > 10 || width % 2 != 0 || height % 2 != 0 || (method == WEBM_METHOD_BITRATE && alpha))
					lib = AOM;
			}

			return lib;
		}
		else if(av1Codec == AV1_CODEC_AOM)
			return AOM;
		else if(av1Codec == AV1_CODEC_NVENC)
			return NVENC;
		else if (av1Codec == AV1_CODEC_SVT_AV1)
			return SVT_AV1;
	}
	else
		return LIBVPX;
}

bool
VideoEncoder::twoPassCapable(WebM_Video_Codec codec, AV1_Codec av1Codec, WebM_Video_Method method, WebM_Chroma_Sampling sampling, int bitDepth, uint32_t width, uint32_t height, bool alpha)
{
	const VideoEncoderLibrary library = WhichVideoEncoder(codec, av1Codec, method, sampling, bitDepth, width, height, alpha);
	
	return (library == LIBVPX || library == AOM);
}

VideoEncoder *
VideoEncoder::makeEncoder(int width, int height, const exRatioValue &pixelAspect,
							const exRatioValue &fps,
							WebM_Video_Codec codec, AV1_Codec av1Codec,
							WebM_Video_Method method, int quality, int bitrate,
							bool twoPass, bool vbrPass, void *vbrBuffer, size_t vbrBufferSize,
							int keyframeMaxDistance, bool forceKeyframes,
							WebM_Chroma_Sampling sampling, int bitDepth,
							WebM_ColorSpace colorSpace, const std::string &custom,
							PrSDKPPixSuite *pixSuite, PrSDKPPix2Suite *pix2Suite, bool alpha)
{
	VideoEncoderLibrary library = WhichVideoEncoder(codec, av1Codec, method, sampling, bitDepth, width, height, alpha);
	
	if(library == NVENC)
	{
	#ifdef WEBM_HAVE_NVENC
		if(NVENCEncoder::available())
		{
			try
			{
				return new NVENCEncoder(width, height, pixelAspect,
										fps,
										method, quality, bitrate,
										twoPass, vbrPass, vbrBuffer, vbrBufferSize,
										keyframeMaxDistance, forceKeyframes,
										sampling, bitDepth,
										colorSpace, custom,
										pixSuite, pix2Suite, alpha);
			}
			catch(...){}
		}
	#endif //WEBM_HAVE_NVENC

		assert(false);
		
		if(av1Codec == AV1_CODEC_NVENC)
			throw exportReturn_InternalError;
		else
			library = SVT_AV1;
	}

	if(library == SVT_AV1)
	{
		try
		{
			return new SVTAV1Encoder(width, height, pixelAspect,
										fps,
										method, quality, bitrate,
										twoPass, vbrPass, vbrBuffer, vbrBufferSize,
										keyframeMaxDistance, forceKeyframes,
										sampling, bitDepth,
										colorSpace, custom,
										pixSuite, pix2Suite, alpha);
		}
		catch (...) {}

		if(av1Codec == AV1_CODEC_SVT_AV1)
			throw exportReturn_InternalError;
		else
			library = AOM;
	}

	if(library == AOM)
	{
		return new AOMEncoder(width, height, pixelAspect,
								fps,
								method, quality, bitrate,
								twoPass, vbrPass, vbrBuffer, vbrBufferSize,
								keyframeMaxDistance, forceKeyframes,
								sampling, bitDepth,
								colorSpace, custom,
								pixSuite, pix2Suite, alpha);
	}
	
	if(library == LIBVPX)
	{
		return new LibVPXEncoder(width, height, pixelAspect,
									fps,
									(codec == WEBM_CODEC_VP8 ? LibVPXEncoder::VP8 : LibVPXEncoder::VP9),
									method, quality, bitrate,
									twoPass, vbrPass, vbrBuffer, vbrBufferSize,
									keyframeMaxDistance, forceKeyframes,
									sampling, bitDepth,
									colorSpace, custom,
									pixSuite, pix2Suite, alpha);
	}

	throw exportReturn_InternalError;
}


const AudioEncoder::Packet *
AudioEncoder::getPacket()
{
	if(_queue.empty())
		return nullptr;
	else
		return _queue.front();
}

void
AudioEncoder::returnPacket(const Packet *packet)
{
	if(packet == _queue.front())
	{
		free(packet->data);
		
		delete packet;
		
		_queue.pop();
	}
	else
		assert(false);
}

PrAudioChannelLabel *
AudioEncoder::channelOrder(WebM_Audio_Codec codec, PrAudioChannelType channelType)
{
	static PrAudioChannelLabel monoOrder[1] = { kPrAudioChannelLabel_Discrete };
											
	static PrAudioChannelLabel stereoOrder[2] = { kPrAudioChannelLabel_FrontLeft,
													kPrAudioChannelLabel_FrontRight };
												
	// Premiere uses Left, Right, Left Rear, Right Rear, Center, LFE
	// Opus and Vorbis use Left, Center, Right, Left Rear, Right Rear, LFE
	// http://www.xiph.org/vorbis/doc/Vorbis_I_spec.html#x1-800004.3.9
	static PrAudioChannelLabel surroundOrder[6] = { kPrAudioChannelLabel_FrontLeft,
													kPrAudioChannelLabel_FrontCenter,
													kPrAudioChannelLabel_FrontRight,
													kPrAudioChannelLabel_RearSurroundLeft,
													kPrAudioChannelLabel_RearSurroundRight,
													kPrAudioChannelLabel_LowFrequency };

	return (channelType == kPrAudioChannelType_51 ? surroundOrder :
			channelType == kPrAudioChannelType_Stereo ? stereoOrder :
			channelType == kPrAudioChannelType_Mono ? monoOrder :
			stereoOrder);
}

AudioEncoder *
AudioEncoder::makeEncoder(int channels, float sampleRate,
							WebM_Audio_Codec codec,
							Ogg_Method method, float quality, int bitrate, bool autoBitrate)
{
	if(codec == WEBM_CODEC_VORBIS)
	{
		return new VorbisEncoder(channels, sampleRate, method, quality, bitrate);
	}
	else if(codec == WEBM_CODEC_OPUS)
	{
		return new OpusEncoder(channels, sampleRate, method, quality, bitrate, autoBitrate);
	}
	else
		throw exportReturn_InternalError;
}
