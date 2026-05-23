plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "com.datn.authenticator"
    compileSdk = 36

    defaultConfig {
        applicationId = "com.datn.authenticator"
        minSdk = 29
        targetSdk = 36
        versionCode = 1
        versionName = "1.0.0-demo"

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
        vectorDrawables.useSupportLibrary = true
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
        debug {
            isDebuggable = true
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_11
        targetCompatibility = JavaVersion.VERSION_11
    }
    kotlinOptions { jvmTarget = "11" }

    buildFeatures {
        viewBinding = true
        buildConfig = true
    }

    // Don't compress these asset extensions
    androidResources {
        noCompress.addAll(listOf("tflite", "json", "npy"))
    }

    packaging {
        resources.excludes.add("META-INF/{AL2.0,LGPL2.1}")
    }
}

dependencies {
    // Core
    implementation("androidx.core:core-ktx:1.13.1")
    implementation("androidx.appcompat:appcompat:1.7.0")
    implementation("com.google.android.material:material:1.12.0")
    implementation("androidx.constraintlayout:constraintlayout:2.1.4")

    // Lifecycle & coroutines
    implementation("androidx.lifecycle:lifecycle-runtime-ktx:2.8.4")
    implementation("androidx.lifecycle:lifecycle-service:2.8.4")
    implementation("androidx.lifecycle:lifecycle-viewmodel-ktx:2.8.4")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.8.1")

    // TFLite (CPU-only — model 317 KB, latency CPU ~2-5ms is more than enough)
    // GPU delegate dropped to avoid version-mismatch issues; can be re-added later.
    implementation("org.tensorflow:tensorflow-lite:2.14.0")

    // Encrypted preferences for shake pattern — see Mục 5.4.1
    implementation("androidx.security:security-crypto:1.1.0-alpha06")

    // JSON parsing for scaler_params.json + manifest
    implementation("org.json:json:20240303")

    // Test
    testImplementation("junit:junit:4.13.2")
    androidTestImplementation("androidx.test.ext:junit:1.2.1")
    androidTestImplementation("androidx.test.espresso:espresso-core:3.6.1")
}
