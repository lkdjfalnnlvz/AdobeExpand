Well, I give up.  I tried, but am not able to make a clean Xcode project to build SVT-AV1.  I got close, but something always kept it from getting through it.  So I'll do what many before me have done, and leave it to the user.

Run CMake and point it to the SVT-AV1 source inside ext/ and build the binaries in this folder.  Turn off BUILD_SHARED_LIBS, and set CMAKE_OUTPUT_DIRECTORY to {this}/Bin.

The Xcode project that gets made will be ugly.  It'll only be for the architecture you're running, so I renamed the project to set-av1_arm64.xcodeproj.  When it builds it will put all kinds of files everywhere, but hopefully it does build and you can get on with your life, like I plan to do.