ext
===

This directory holds git submodules that point to libraries needed by the WebM plug-in.

You will need to manually add the following to this directory because the owners don't have a git repository I can embed:

* [Premiere CC 2019 (13.0) SDK](https://developer.adobe.com/content/udp/en/apis/creativecloud/premierepro.html)
* [Nvidia Video Codec SDK (NVENC)](https://developer.nvidia.com/nvidia-video-codec-sdk/download)

If the submodule contents are missing, you should be able to get them by typing:

`git submodule init`
`git submodule update`

Both Mac and Windows require the "yasm" shell program be installed for libvpx to compile.  On Windows you get it via Cygwin.

