[app]
title = 国服CR识别
package.name = cr.ai
package.domain = com.yourdomain.cr
source.dir = .
source.include_exts = py,onnx,jpg,png,spec
version = 0.1
# p4a recipe 名是 opencv（不是 opencv-python）
requirements = python3,kivy,opencv,numpy
android.permissions = INTERNET, WRITE_EXTERNAL_STORAGE, READ_EXTERNAL_STORAGE, CAMERA
android.api = 33
android.minapi = 21
android.ndk = 25b
android.archs = arm64-v8a
orientation = landscape
fullscreen = 0

[buildozer]
log_level = 2
warn_on_root = 0
