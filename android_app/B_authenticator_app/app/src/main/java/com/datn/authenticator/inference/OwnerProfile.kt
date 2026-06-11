package com.datn.authenticator.inference

import android.content.Context
import android.util.Log
import java.io.DataInputStream
import java.io.DataOutputStream
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream

class OwnerProfile(context: Context) {
    private val file: File = File(context.filesDir, FILE_NAME)

    private var cachedAnchors: List<FloatArray>? = null
    private var cachedRfInertial: RandomForestClassifier? = null
    private var cachedRfTouch: RandomForestClassifier? = null
    private var cachedFusionW: Float = FusionEngine.DEFAULT_W
    // ZNORM: tham số chuẩn hóa cohort (mặc định 0/1 = không chuẩn hóa, suy biến về cos_mean)
    private var cachedCohortMean: Float = 0f
    private var cachedCohortStd:  Float = 1f
    // PER-OWNER THRESHOLD: ngưỡng quyết định riêng cho owner (NaN = chưa hiệu chuẩn → dùng ngưỡng manifest)
    private var cachedThrInertial: Float = ThresholdCalibrator.UNCALIBRATED

    fun getCohortMean(): Float { if (cachedAnchors == null) loadFromDisk(); return cachedCohortMean }
    fun getCohortStd():  Float { if (cachedAnchors == null) loadFromDisk(); return cachedCohortStd }
    fun getThrInertial(): Float { if (cachedAnchors == null) loadFromDisk(); return cachedThrInertial }

    fun hasEnrollment(): Boolean = file.exists() && getAnchors().isNotEmpty()

    fun getAnchors(): List<FloatArray> {
        if (cachedAnchors != null) return cachedAnchors!!
        loadFromDisk()
        return cachedAnchors ?: emptyList()
    }

    fun getRfInertial(): RandomForestClassifier? {
        if (cachedRfInertial != null) return cachedRfInertial
        loadFromDisk()
        return cachedRfInertial
    }

    fun getRfTouch(): RandomForestClassifier? {
        if (cachedRfTouch != null) return cachedRfTouch
        loadFromDisk()
        return cachedRfTouch
    }

    fun getFusionW(): Float {
        if (cachedAnchors != null) return cachedFusionW
        loadFromDisk()
        return cachedFusionW
    }

    fun save(
        anchors: List<FloatArray>,
        rfInertial: RandomForestClassifier,
        rfTouch: RandomForestClassifier?,
        fusionW: Float,
        cohortMean: Float = 0f,
        cohortStd: Float = 1f,
        thrInertial: Float = ThresholdCalibrator.UNCALIBRATED,
    ) {
        require(anchors.isNotEmpty()) { "Cannot save empty anchor list" }
        require(anchors.all { it.size == InferenceEngine.EMBED_DIM }) {
            "All anchors must have dim ${InferenceEngine.EMBED_DIM}"
        }

        DataOutputStream(FileOutputStream(file)).use { dos ->
            dos.writeInt(MAGIC_V4)
            dos.writeInt(anchors.size)
            dos.writeInt(InferenceEngine.EMBED_DIM)
            for (a in anchors) for (v in a) dos.writeFloat(v)

            dos.writeFloat(fusionW)
            // ZNORM: ghi 2 tham số cohort ngay sau fusionW
            dos.writeFloat(cohortMean)
            dos.writeFloat(cohortStd)
            // PER-OWNER THRESHOLD: ghi ngưỡng riêng ngay sau cohort
            dos.writeFloat(thrInertial)

            val rfIBytes = rfInertial.toByteArray()
            dos.writeByte(if (rfTouch != null) 1 else 0)
            dos.writeInt(rfIBytes.size)
            dos.write(rfIBytes)

            if (rfTouch != null) {
                val rfTBytes = rfTouch.toByteArray()
                dos.writeInt(rfTBytes.size)
                dos.write(rfTBytes)
            }
        }

        cachedAnchors   = anchors.map { it.copyOf() }
        cachedRfInertial = rfInertial
        cachedRfTouch    = rfTouch
        cachedFusionW    = fusionW
        cachedCohortMean = cohortMean
        cachedCohortStd  = cohortStd
        cachedThrInertial = thrInertial

        Log.i(TAG, "Profile saved: ${anchors.size} anchors, fusion_w=$fusionW, " +
                "cohort=($cohortMean,$cohortStd), thr=$thrInertial, touch=${rfTouch != null}")
    }

    fun saveAnchors(anchors: List<FloatArray>) {
        require(anchors.isNotEmpty())
        DataOutputStream(FileOutputStream(file)).use { dos ->
            dos.writeInt(MAGIC_V1)
            dos.writeInt(anchors.size)
            dos.writeInt(InferenceEngine.EMBED_DIM)
            for (a in anchors) for (v in a) dos.writeFloat(v)
        }
        cachedAnchors    = anchors.map { it.copyOf() }
        cachedRfInertial = null
        cachedRfTouch    = null
        cachedFusionW    = FusionEngine.DEFAULT_W
        cachedCohortMean = 0f; cachedCohortStd = 1f
        cachedThrInertial = ThresholdCalibrator.UNCALIBRATED
        Log.i(TAG, "Anchors-only profile saved: ${anchors.size} anchors")
    }

    fun clear() {
        if (file.exists()) file.delete()
        cachedAnchors = null; cachedRfInertial = null
        cachedRfTouch = null; cachedFusionW = FusionEngine.DEFAULT_W
        cachedCohortMean = 0f; cachedCohortStd = 1f
        cachedThrInertial = ThresholdCalibrator.UNCALIBRATED
        Log.i(TAG, "Owner profile cleared")
    }

    private fun loadFromDisk() {
        if (!file.exists()) return
        try {
            DataInputStream(FileInputStream(file)).use { dis ->
                val magic = dis.readInt()
                when (magic) {
                    MAGIC_V1 -> loadV1(dis)
                    MAGIC_V2 -> loadV2(dis)
                    MAGIC_V3 -> loadV3(dis)
                    MAGIC_V4 -> loadV4(dis)
                    else -> Log.e(TAG, "Unknown magic 0x${magic.toString(16)}")
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to read profile: ${e.message}", e)
            cachedAnchors = emptyList()
        }
    }

    private fun loadV1(dis: DataInputStream) {
        val n = dis.readInt(); val dim = dis.readInt()
        val list = ArrayList<FloatArray>(n)
        repeat(n) { val v = FloatArray(dim) { dis.readFloat() }; list.add(v) }
        cachedAnchors    = list
        cachedRfInertial = null
        cachedRfTouch    = null
        cachedFusionW    = FusionEngine.DEFAULT_W
        cachedCohortMean = 0f; cachedCohortStd = 1f   // ZNORM: profile cũ → không chuẩn hóa
        cachedThrInertial = ThresholdCalibrator.UNCALIBRATED
        Log.i(TAG, "Loaded V1 profile: $n anchors")
    }

    private fun loadV4(dis: DataInputStream) {
        val n = dis.readInt(); val dim = dis.readInt()
        val list = ArrayList<FloatArray>(n)
        repeat(n) { val v = FloatArray(dim) { dis.readFloat() }; list.add(v) }
        cachedAnchors = list

        cachedFusionW    = dis.readFloat()
        cachedCohortMean = dis.readFloat()   // ZNORM
        cachedCohortStd  = dis.readFloat()   // ZNORM
        cachedThrInertial = dis.readFloat()  // PER-OWNER THRESHOLD
        val hasTouch = dis.readByte().toInt() == 1

        val rfISize = dis.readInt()
        val rfIBytes = ByteArray(rfISize).also { dis.readFully(it) }
        cachedRfInertial = RandomForestClassifier().apply { readFrom(rfIBytes.inputStream()) }

        if (hasTouch) {
            val rfTSize = dis.readInt()
            val rfTBytes = ByteArray(rfTSize).also { dis.readFully(it) }
            cachedRfTouch = RandomForestClassifier().apply { readFrom(rfTBytes.inputStream()) }
        }
        Log.i(TAG, "Loaded V4 profile: $n anchors, fusion_w=$cachedFusionW, " +
                "cohort=($cachedCohortMean,$cachedCohortStd), thr=$cachedThrInertial, touch=$hasTouch")
    }

    private fun loadV3(dis: DataInputStream) {
        val n = dis.readInt(); val dim = dis.readInt()
        val list = ArrayList<FloatArray>(n)
        repeat(n) { val v = FloatArray(dim) { dis.readFloat() }; list.add(v) }
        cachedAnchors = list

        cachedFusionW    = dis.readFloat()
        cachedCohortMean = dis.readFloat()   // ZNORM
        cachedCohortStd  = dis.readFloat()   // ZNORM
        cachedThrInertial = ThresholdCalibrator.UNCALIBRATED  // V3 chưa có → dùng ngưỡng manifest
        val hasTouch = dis.readByte().toInt() == 1

        val rfISize = dis.readInt()
        val rfIBytes = ByteArray(rfISize).also { dis.readFully(it) }
        cachedRfInertial = RandomForestClassifier().apply { readFrom(rfIBytes.inputStream()) }

        if (hasTouch) {
            val rfTSize = dis.readInt()
            val rfTBytes = ByteArray(rfTSize).also { dis.readFully(it) }
            cachedRfTouch = RandomForestClassifier().apply { readFrom(rfTBytes.inputStream()) }
        }
        Log.i(TAG, "Loaded V3 profile: $n anchors, fusion_w=$cachedFusionW, " +
                "cohort=($cachedCohortMean,$cachedCohortStd), touch=$hasTouch")
    }

    private fun loadV2(dis: DataInputStream) {
        val n = dis.readInt(); val dim = dis.readInt()
        val list = ArrayList<FloatArray>(n)
        repeat(n) { val v = FloatArray(dim) { dis.readFloat() }; list.add(v) }
        cachedAnchors = list

        cachedFusionW = dis.readFloat()
        cachedCohortMean = 0f; cachedCohortStd = 1f   // ZNORM: V2 chưa có cohort
        cachedThrInertial = ThresholdCalibrator.UNCALIBRATED
        val hasTouch = dis.readByte().toInt() == 1

        val rfISize = dis.readInt()
        val rfIBytes = ByteArray(rfISize).also { dis.readFully(it) }
        cachedRfInertial = RandomForestClassifier().apply { readFrom(rfIBytes.inputStream()) }

        if (hasTouch) {
            val rfTSize = dis.readInt()
            val rfTBytes = ByteArray(rfTSize).also { dis.readFully(it) }
            cachedRfTouch = RandomForestClassifier().apply { readFrom(rfTBytes.inputStream()) }
        }
        Log.i(TAG, "Loaded V2 profile: $n anchors, fusion_w=$cachedFusionW, touch=${hasTouch}")
    }

    companion object {
        private const val TAG       = "OwnerProfile"
        private const val FILE_NAME = "owner_anchors.bin"
        private const val MAGIC_V1  = 0xACAF0001.toInt()
        private const val MAGIC_V2  = 0xACAF0002.toInt()
        private const val MAGIC_V3  = 0xACAF0003.toInt()
        private const val MAGIC_V4  = 0xACAF0004.toInt()
    }
}

private fun RandomForestClassifier.toByteArray(): ByteArray {
    val buf = java.io.ByteArrayOutputStream()
    writeTo(buf)
    return buf.toByteArray()
}
