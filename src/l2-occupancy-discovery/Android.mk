LOCAL_PATH := $(call my-dir)

include $(CLEAR_VARS)

LOCAL_MODULE    := occupancy_discovery_benchmark 
LOCAL_C_INCLUDES := ../../easyvk/src
LOCAL_SRC_FILES := occupancy_discovery.cpp ../../easyvk/src/easyvk.cpp
LOCAL_LDLIBS    += -lvulkan -llog

include $(BUILD_EXECUTABLE)