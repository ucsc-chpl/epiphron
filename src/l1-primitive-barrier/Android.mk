LOCAL_PATH := $(call my-dir)

include $(CLEAR_VARS)

LOCAL_MODULE    := primitive_barrier_benchmark
LOCAL_C_INCLUDES := ../../easyvk/src
LOCAL_SRC_FILES := primitive_barrier_benchmark.cpp ../../easyvk/src/easyvk.cpp
LOCAL_LDLIBS    += -lvulkan -llog

include $(BUILD_EXECUTABLE)