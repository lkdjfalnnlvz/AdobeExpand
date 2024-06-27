After flailing with Xcode, I won't even bother trying to make a clean Visual Studio project for SVT-AV1.  So I'll do what many before me have done, and leave it to the user.

Run CMake and point it to the SVT-AV1 source inside ext/ and build the binaries in this folder.  Turn off BUILD_SHARED_LIBS, and set CMAKE_OUTPUT_DIRECTORY to {this}/Bin.

The VC projects that gets made will be ugly.  For projects using assembler, I had to flip \ to / in the include directory paths.