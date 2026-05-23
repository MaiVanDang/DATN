# Keep TFLite native bindings
-keep class org.tensorflow.lite.** { *; }
-dontwarn org.tensorflow.lite.**
# Keep model classes used by reflection
-keep class com.datn.authenticator.model.** { *; }
