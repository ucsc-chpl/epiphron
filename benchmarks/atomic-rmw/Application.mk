MY_APP_DIR := $(call my-dir)
APP_BUILD_SCRIPT := $(MY_APP_DIR)/Android.mk
APP_STL := c++_static 
APP_ABI := arm64-v8a 
APP_CPPFLAGS := -fexceptions
APP_PLATFORM := 24 # min version that supports Vulkan