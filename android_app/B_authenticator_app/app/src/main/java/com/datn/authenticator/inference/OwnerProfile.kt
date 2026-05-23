package com.datn.authenticator.inference

import android.content.Context
import android.util.Log
import java.io.DataInputStream
import java.io.DataOutputStream
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream

/**
 * Persists the owner's full biometric profile:
 *   - 6 IMU anchor embeddings (128-D)  → for cosine-sim fallback scoring
 *   - RF_inertial model                → primary inertial verifier
 *   - RF_touch model (optional)        → touch verifier (null if not enrolled)
 *   - fusion_w                         → inertial weight tuned at enrollment
 *
 * File format (binary, big-endian via Java DataOutputStream):
 *   [int  MAGIC_V2 = 0xACAF0002]
 *   [int  n_anchors]
 *   [int  embed_dim = 128]
 *   [float[] anchor_data]              n_anchors × embed_dim floats
 *   [float   fusion_w]
 *   [byte    has_touch_rf]             1 = present, 0 = absent
 *   [int     rf_inertial_bytes]        byte length of serialised RF
 *   [byte[]  rf_inertial_data]
 *   IF has_touch_rf:
 *     [int     rf_touch_bytes]
 *     [byte[]  rf_touch_data]
 */
class OwnerProfile(context: Context) {

    private val file: File = File(context.filesDir, FILE_NAME)

    private var cachedAnchors: List<FloatArray>? = null
    private var cachedRfInertial: RandomForestClassifier? = null
    private var cachedRfTouch: RandomForestClassifier? = null
    private var cachedFusionW: Float = FusionEngine.DEFAULT_W

    // ── Public read API ───────────────────────────────────────────────────

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
        if (cachedAnchors != null) return cachedFusionW  // already loaded
        loadFromDisk()
        return cachedFusionW
    }

    // ── Save ──────────────────────────────────────────────────────────────

    fun save(
        anchors: List<FloatArray>,
        rfInertial: RandomForestClassifier,
        rfTouch: RandomForestClassifier?,
        fusionW: Float,
    ) {
        require(anchors.isNotEmpty()) { "Cannot save empty anchor list" }
        require(anchors.all { it.size == InferenceEngine.EMBED_DIM }) {
            "All anchors must have dim ${InferenceEngine.EMBED_DIM}"
        }

        DataOutputStream(FileOutputStream(file)).use { dos ->
            dos.writeInt(MAGIC_V2)
            dos.writeInt(anchors.size)
            dos.writeInt(InferenceEngine.EMBED_DIM)
            for (a in anchors) for (v in a) dos.writeFloat(v)

            dos.writeFloat(fusionW)

            // RF inertial (always present)
            val rfIBytes = rfInertial.toByteArray()
            dos.writeByte(if (rfTouch != null) 1 else 0)
            dos.writeInt(rfIBytes.size)
            dos.write(rfIBytes)

            // RF touch (optional)
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

        Log.i(TAG, "Profile saved: ${anchors.size} anchors, fusion_w=$fusionW, touch=${rfTouch != null}")
    }

    /** Legacy save (anchors only, no RF) — keeps backward compat with old enrollment. */
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
        Log.i(TAG, "Anchors-only profile saved: ${anchors.size} anchors")
    }

    fun clear() {
        if (file.exists()) file.delete()
        cachedAnchors = null; cachedRfInertial = null
        cachedRfTouch = null; cachedFusionW = FusionEngine.DEFAULT_W
        Log.i(TAG, "Owner profile cleared")
    }

    // ── Load ──────────────────────────────────────────────────────────────

    private fun loadFromDisk() {
        if (!file.exists()) return
        try {
            DataInputStream(FileInputStream(file)).use { dis ->
                val magic = dis.readInt()
                when (magic) {
                    MAGIC_V1 -> loadV1(dis)
                    MAGIC_V2 -> loadV2(dis)
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
        Log.i(TAG, "Loaded V1 profile: $n anchors")
    }

    private fun loadV2(dis: DataInputStream) {
        val n = dis.readInt(); val dim = dis.readInt()
        val list = ArrayList<FloatArray>(n)
        repeat(n) { val v = FloatArray(dim) { dis.readFloat() }; list.add(v) }
        cachedAnchors = list

        cachedFusionW = dis.readFloat()
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
    }
}

// ── Extension helpers ─────────────────────────────────────────────────────────

private fun RandomForestClassifier.toByteArray(): ByteArray {
    val buf = java.io.ByteArrayOutputStream()
    writeTo(buf)
    return buf.toByteArray()
}
