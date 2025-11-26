cmake_minimum_required(VERSION 3.15)
project(AdobeExpand)

set(SOURCES
    src/premiere/WebM_Premiere_Import.cpp
    src/premiere/win/WebM_Import.rc
)

add_library(${PROJECT_NAME} SHARED ${SOURCES})

target_include_directories(${PROJECT_NAME} PRIVATE
    src/premiere
    ext/ffmpeg/include
    "ext/Premiere Pro CC 13.0 Win SDK/Examples/Headers"
    "ext/Premiere Pro CC 13.0 Win SDK/Examples/Utils"
)

target_link_directories(${PROJECT_NAME} PRIVATE
    ext/ffmpeg/lib
)

target_link_libraries(${PROJECT_NAME} PRIVATE
    avformat avcodec avutil swscale swresample
    winmm comctl32
)

target_compile_definitions(${PROJECT_NAME} PRIVATE
    ISOLATION_AWARE_ENABLED=1
    WIN32 _WIN64 _WINDOWS PRWIN_ENV MSWindows
)

set_target_properties(${PROJECT_NAME} PROPERTIES 
    SUFFIX ".prm"
    PREFIX ""
)
