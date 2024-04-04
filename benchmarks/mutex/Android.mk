LOCAL_PATH := $(call my-dir)

include $(CLEAR_VARS)

LOCAL_MODULE    := mutex_benchmark
LOCAL_C_INCLUDES := ../../easyvk/src
LOCAL_SRC_FILES := mutex_test.cpp ../../easyvk/src/easyvk.cpp
LOCAL_LDLIBS    += -lvulkan -llog

#include $(BUILD_SHARED_LIBRARY)
include $(BUILD_EXECUTABLE)