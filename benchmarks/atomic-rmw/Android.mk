LOCAL_PATH := $(call my-dir)

include $(CLEAR_VARS)

LOCAL_MODULE    := rmw_benchmark
LOCAL_C_INCLUDES := ../../easyvk/src ../../easyvk/volk
LOCAL_SRC_FILES := atomic_rmw_test.cpp vk_utils.cpp ../../easyvk/src/easyvk.cpp
LOCAL_LDLIBS    += -lvulkan -llog

#include $(BUILD_SHARED_LIBRARY)
include $(BUILD_EXECUTABLE)