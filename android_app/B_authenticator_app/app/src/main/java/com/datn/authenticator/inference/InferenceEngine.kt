package com.datn.authenticator.inference

import android.content.Context
import android.util.Log
import com.datn.authenticator.model.ExportManifest
import com.datn.authenticator.model.ScalerParams
import com.datn.authenticator.model.SensorWindow
import org.json.JSONObject
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import kotlin.math.exp
import kotlin.math.sqrt

/**
 * TFLite Inference Engine — v2 CPU-only (backbone + on-device enrollment).
 *
 * Model contract (matches export from training pipeline v5):
 *   - file        : assets/backbone.tflite       (~308 KB float32)
 *   - input  shape: [1, 200, 9]   NWC  (timestep-first; 200 samples × 9 channels)
 *   - output shape: [1, 128]       128-D inertial embedding
 *
 * Authentication strategy:
 *   1. extractEmbedding(window)  -> 128-D vector
 *   2. cosineSim against OwnerProfile.getAnchors() (6 enrollment embeddings)
 *   3. sigmoid-scaled mean similarity -> probability in [0, 1]
 *
 * If no owner enrolled, predict() returns 0.5 (neutral) so the rest of the app
 * can prompt user to run enrollment first.
 *
 * GPU delegate removed: model is 317 KB and inference is ~2-5ms on CPU,
 * which is plenty for the app's 1 inference / 4s requirement.
 *
 * Public API (predict, warmUp, close, PredictionResult, Backend) is UNCHANGED
 * from the previous mock implementation - Service/Aggregator/UI don't need edits.
 */
class InferenceEngine private constructor(
    private val interpreter: Interpreter?,
    private val scaler: ScalerParams,
    private val ownerProfile: OwnerProfile,
    val manifest: ExportManifest?,
    val isMockMode: Boolean,
    val backend: Backend,
) : AutoCloseable {

    enum class Backend { CPU_4_THREAD, CPU_1_THREAD, MOCK }

    // Input buffer for the backbone: 200 timesteps x 9 channels = 1800 floats (NWC)
    private val inputBuffer: ByteBuffer = ByteBuffer
        .allocateDirect(TIMESTEPS * N_CHANNELS * BYTES_PER_FLOAT)
        .order(ByteOrder.nativeOrder())

    // Output buffer for the backbone: 128-D embedding
    private val embedBuffer: ByteBuffer = ByteBuffer
        .allocateDirect(EMBED_DIM * BYTES_PER_FLOAT)
        .order(ByteOrder.nativeOrder())

    private val scratchEmbed = FloatArray(EMBED_DIM)

    /**
     * Extract one 128-D embedding from a raw sensor window.
     * Used by predict() AND by OwnerEnrollmentActivity to save anchors.
     */
    fun extractEmbedding(window: SensorWindow): FloatArray {
        val normalized = scaler.normalize(window.data)

        // SensorWindow stores data timestep-major: data[t * 9 + c] = NWC layout
        // Backbone expects NWC [1, 200, 9] — copy directly, no transpose needed
        inputBuffer.rewind()
        for (f in normalized) inputBuffer.putFloat(f)

        if (isMockMode || interpreter == null) {
            // Deterministic mock embedding so dev/demo flow works without model
            val seed = (window.startTimestampMs xor window.endTimestampMs).toInt()
            val rng = java.util.Random(seed.toLong())
            for (i in 0 until EMBED_DIM) scratchEmbed[i] = rng.nextFloat() - 0.5f
            return scratchEmbed.copyOf()
        }

        embedBuffer.rewind()
        interpreter.run(inputBuffer, embedBuffer)
        embedBuffer.rewind()
        for (i in 0 until EMBED_DIM) scratchEmbed[i] = embedBuffer.float
        return scratchEmbed.copyOf()
    }

    /**
     * Run inference + scoring against owner profile.
     *
     * If the profile contains a trained RF_inertial model, uses RF scoring.
     * Otherwise falls back to cosine-similarity (legacy / mock mode).
     *
     * Returns PredictionResult with probabilityLegitimate = p_inertial in [0,1].
     * For fused score, call predictFused() which also needs a touch feature vector.
     */
    fun predict(window: SensorWindow): PredictionResult {
        val t0 = System.nanoTime()
        val embed = extractEmbedding(window)
        val tEmbed = System.nanoTime()

        val score: Float
        val rawScore: Float

        val anchors = ownerProfile.getAnchors()
        if (anchors.isNotEmpty() && !isMockMode) {
            // ── Cosine similarity (primary path) ─────────────────────────
            // RF on-device với <30 enrollment samples không generalise tốt
            // trong 128-D space (pool training chứa data của chính owner).
            // Cosine similarity so sánh trực tiếp với anchors — robust hơn
            // khi enrollment data ít.
            val meanSim = meanCosineSimilarity(embed, anchors)
            rawScore = meanSim

            // RF làm reference log nhưng KHÔNG dùng để score
            val rfInertial = ownerProfile.getRfInertial()
            val rfScore = if (rfInertial != null && rfInertial.isTrained)
                rfInertial.predictProba(embed) else -1f

            score = sigmoid(SCORE_SCALE * (meanSim - SCORE_BIAS))
            Log.d(TAG, "predict: cosine=${"%.3f".format(meanSim)} → score=${"%.3f".format(score)}" +
                    (if (rfScore >= 0) "  [RF_ref=${"%.3f".format(rfScore)}]" else ""))
        } else {
            score = 0.5f; rawScore = 0f
        }

        val tScore = System.nanoTime()
        return PredictionResult(
            probabilityLegitimate = score,
            rawLogit = rawScore,
            normalizationLatencyNs = 0L,
            copyLatencyNs = 0L,
            inferenceLatencyNs = tEmbed - t0,
            totalLatencyNs = tScore - t0,
            driftWarning = false,
            window = window,
            backend = backend,
        )
    }

    /**
     * Full V5-synced fusion prediction:
     *   1. CNN → embedding → RF_inertial → p_inertial (per window)
     *   2. touchVec → RF_touch → p_touch (session-level, may be null)
     *   3. fused = fusion_w * p_inertial + (1-fusion_w) * p_touch
     *
     * @param window        current IMU window
     * @param touchVec      48-D touch feature vector (null if no touch data)
     * @param touchScaler   (mean, scale) arrays from touch_scaler.json
     */
    fun predictFused(
        window: SensorWindow,
        touchVec: FloatArray?,
        touchScaler: Pair<FloatArray, FloatArray>?,
    ): FusedResult {
        val inertialResult = predict(window)
        val pInertial = inertialResult.probabilityLegitimate

        var pTouch: Float? = null
        val rfTouch = ownerProfile.getRfTouch()
        if (rfTouch != null && rfTouch.isTrained && touchVec != null && touchScaler != null) {
            val (mean, scale) = touchScaler
            val scaled = FloatArray(touchVec.size) { i ->
                val s = scale[i].takeIf { it > 0f } ?: 1f
                (touchVec[i] - mean[i]) / s
            }
            pTouch = rfTouch.predictProba(scaled)
            Log.d(TAG, "predictFused: p_touch=$pTouch")
        }

        val w = ownerProfile.getFusionW()
        val fused = FusionEngine.fuse(floatArrayOf(pInertial), pTouch, w)

        return FusedResult(inertialResult, pTouch, w, fused)
    }

    data class FusedResult(
        val inertial: PredictionResult,
        val pTouch: Float?,
        val fusionW: Float,
        val fusedScore: Float,
    )

    fun warmUp(iterations: Int = 5) {
        val dummy = SensorWindow(
            data = FloatArray(TIMESTEPS * N_CHANNELS) { 0f },
            startTimestampMs = 0L,
            endTimestampMs = 0L,
        )
        repeat(iterations) { extractEmbedding(dummy) }
    }

    /** True if user has completed biometric enrollment. */
    fun hasOwnerEnrollment(): Boolean = ownerProfile.hasEnrollment()

    fun ownerProfile(): OwnerProfile = ownerProfile

    override fun close() {
        try { interpreter?.close() } catch (e: Exception) {
            Log.w(TAG, "Interpreter close failed: ${e.message}")
        }
    }

    // Scoring helpers

    private fun meanCosineSimilarity(embed: FloatArray, anchors: List<FloatArray>): Float {
        if (anchors.isEmpty()) return 0f
        var total = 0f
        for (a in anchors) total += cosineSimilarity(embed, a)
        return total / anchors.size
    }

    private fun cosineSimilarity(a: FloatArray, b: FloatArray): Float {
        require(a.size == b.size) { "Embedding dim mismatch: ${a.size} vs ${b.size}" }
        var dot = 0f
        var na = 0f
        var nb = 0f
        for (i in a.indices) {
            dot += a[i] * b[i]
            na  += a[i] * a[i]
            nb  += b[i] * b[i]
        }
        val denom = sqrt(na.toDouble()) * sqrt(nb.toDouble()) + 1e-9
        return (dot / denom).toFloat()
    }

    data class PredictionResult(
        val probabilityLegitimate: Float,
        val rawLogit: Float,
        val normalizationLatencyNs: Long,
        val copyLatencyNs: Long,
        val inferenceLatencyNs: Long,
        val totalLatencyNs: Long,
        val driftWarning: Boolean,
        val window: SensorWindow,
        val backend: Backend,
    ) {
        val totalLatencyMs:     Float get() = totalLatencyNs     / 1_000_000f
        val inferenceLatencyMs: Float get() = inferenceLatencyNs / 1_000_000f
    }

    companion object {
        private const val TAG = "InferenceEngine"
        private const val BYTES_PER_FLOAT = 4
        const val EMBED_DIM = 128
        const val N_CHANNELS = SensorWindow.CHANNELS    // 9
        const val TIMESTEPS = SensorWindow.TIMESTEPS    // 200
        private const val DEFAULT_MODEL_ASSET = "backbone.tflite"

        // Sigmoid mapping: score = sigmoid(SCORE_SCALE * (cosine_sim - SCORE_BIAS))
        // cosine_sim=0.50 → score≈0.88 TRUSTED | cosine_sim=0.35 → score≈0.73 WARNING
        // cosine_sim=0.25 → score≈0.55 WARNING  | cosine_sim=0.10 → score≈0.23 UNKNOWN
        private const val SCORE_SCALE = 8f
        private const val SCORE_BIAS  = 0.25f

        fun sigmoid(x: Float): Float = 1f / (1f + exp(-x))

        @Suppress("UNUSED_PARAMETER")
        fun load(
            context: Context,
            useGpu: Boolean = false,            // kept for API compat, GPU dropped
            numThreads: Int = 4,
            forceCpuOneThread: Boolean = false,
        ): InferenceEngine {
            val ownerProfile = OwnerProfile(context)

            val scaler = try {
                ScalerParams.loadFromAssets(context, "scaler_params.json")
            } catch (e: Exception) {
                Log.w(TAG, "scaler_params.json not found. Using per-window Z-score default.")
                ScalerParams.defaultPerWindow()
            }
            val manifest = ExportManifest.loadFromAssets(context, "export_manifest.json")

            val modelBuffer = tryLoadModel(context, DEFAULT_MODEL_ASSET)
            if (modelBuffer == null) {
                Log.w(TAG, "$DEFAULT_MODEL_ASSET not found. Running in MOCK mode.")
                return InferenceEngine(
                    interpreter = null,
                    scaler = scaler,
                    ownerProfile = ownerProfile,
                    manifest = manifest,
                    isMockMode = true,
                    backend = Backend.MOCK,
                )
            }

            val effectiveThreads = if (forceCpuOneThread) 1 else numThreads
            val chosenBackend = if (effectiveThreads == 1) Backend.CPU_1_THREAD else Backend.CPU_4_THREAD

            val options = Interpreter.Options().apply {
                setNumThreads(effectiveThreads)
            }

            val interp = Interpreter(modelBuffer, options)

            val inShape  = interp.getInputTensor(0).shape()
            val outShape = interp.getOutputTensor(0).shape()
            val expectedIn  = intArrayOf(1, TIMESTEPS, N_CHANNELS)
            val expectedOut = intArrayOf(1, EMBED_DIM)
            if (!inShape.contentEquals(expectedIn)) {
                Log.w(TAG, "Input shape mismatch - expected ${expectedIn.toList()}, model says ${inShape.toList()}")
            }
            if (!outShape.contentEquals(expectedOut)) {
                Log.w(TAG, "Output shape mismatch - expected ${expectedOut.toList()}, model says ${outShape.toList()}")
            }

            Log.i(TAG, "InferenceEngine loaded: backend=$chosenBackend, " +
                    "enrolled=${ownerProfile.hasEnrollment()}, anchors=${ownerProfile.getAnchors().size}")

            return InferenceEngine(
                interpreter = interp,
                scaler = scaler,
                ownerProfile = ownerProfile,
                manifest = manifest,
                isMockMode = false,
                backend = chosenBackend,
            )
        }

        /** Load (mean, scale) from assets/touch_scaler.json. Returns null if file missing. */
        fun loadTouchScaler(context: Context): Pair<FloatArray, FloatArray>? {
            return try {
                val json = context.assets.open("touch_scaler.json").bufferedReader().readText()
                val obj = JSONObject(json)
                val meanArr = obj.getJSONArray("mean")
                val scaleArr = obj.getJSONArray("scale")
                val mean  = FloatArray(meanArr.length())  { meanArr.getDouble(it).toFloat() }
                val scale = FloatArray(scaleArr.length()) { scaleArr.getDouble(it).toFloat() }
                mean to scale
            } catch (e: Exception) {
                Log.w(TAG, "touch_scaler.json not found or invalid: ${e.message}")
                null
            }
        }

        private fun tryLoadModel(context: Context, asset: String): MappedByteBuffer? {
            return try {
                val fd = context.assets.openFd(asset)
                FileInputStream(fd.fileDescriptor).channel.use { channel ->
                    channel.map(FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength)
                }
            } catch (e: Exception) {
                Log.d(TAG, "Could not memory-map $asset: ${e.message}")
                null
            }
        }
    }
}
