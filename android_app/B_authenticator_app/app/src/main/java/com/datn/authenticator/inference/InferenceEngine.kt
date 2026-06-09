package com.datn.authenticator.inference

import android.content.Context
import android.util.Log
import com.datn.authenticator.model.ScalerParams
import com.datn.authenticator.model.SensorWindow
import com.datn.authenticator.util.ContextMode
import org.json.JSONObject
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import kotlin.math.exp
import kotlin.math.sqrt

class InferenceEngine private constructor(
    private val interpreter: Interpreter?,
    private val scaler: ScalerParams,
    private val ownerProfile: OwnerProfile,
    private val adaptiveBuffer: AdaptiveAnchorBuffer,
    val isMockMode: Boolean,
    val backend: Backend,
) : AutoCloseable {
    // ZNORM: tham số chuẩn hóa cohort, đọc từ OwnerProfile (mặc định 0/1 = không chuẩn hóa)
    private val cohortMean: Float get() = ownerProfile.getCohortMean()
    private val cohortStd:  Float get() = ownerProfile.getCohortStd()
    enum class Backend { CPU_4_THREAD, CPU_1_THREAD, MOCK }

    private val inputBuffer: ByteBuffer = ByteBuffer
        .allocateDirect(TIMESTEPS * N_CHANNELS * BYTES_PER_FLOAT)
        .order(ByteOrder.nativeOrder())

    private val embedBuffer: ByteBuffer = ByteBuffer
        .allocateDirect(EMBED_DIM * BYTES_PER_FLOAT)
        .order(ByteOrder.nativeOrder())

    private val scratchEmbed = FloatArray(EMBED_DIM)

    fun extractEmbedding(window: SensorWindow): FloatArray {
        val normalized = scaler.normalize(window.data)

        inputBuffer.rewind()
        for (f in normalized) inputBuffer.putFloat(f)

        if (isMockMode || interpreter == null) {
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

    fun predict(window: SensorWindow): PredictionResult {
        val t0 = System.nanoTime()
        val embed = extractEmbedding(window)

        val score: Float
        val coreAnchors = ownerProfile.getAnchors()
        if (coreAnchors.isNotEmpty() && !isMockMode) {
            val adaptive = adaptiveBuffer.all()
            val allAnchors = if (adaptive.isEmpty()) coreAnchors else coreAnchors + adaptive
            val meanSim = meanCosineSimilarity(embed, allAnchors)

            val rfInertial = ownerProfile.getRfInertial()
            val rfScore = if (rfInertial != null && rfInertial.isTrained)
                rfInertial.predictProba(embed) else -1f

            // ZNORM: chuẩn hóa điểm cosine theo phân bố cohort trước khi qua sigmoid.
            // Nếu chưa có cohort (std<=0, profile cũ) thì zSim = meanSim → suy biến về cos_mean.
            val zSim = if (cohortStd > 1e-6f) (meanSim - cohortMean) / cohortStd else meanSim
            score = sigmoid(SCORE_SCALE * (zSim - SCORE_BIAS))
            Log.d(TAG, "predict: cosine=${"%.3f".format(meanSim)} z=${"%.3f".format(zSim)} → score=${"%.3f".format(score)}" +
                    "  [core=${coreAnchors.size} adapt=${adaptive.size}]" +
                    (if (rfScore >= 0) "  [RF_ref=${"%.3f".format(rfScore)}]" else ""))
        } else {
            score = 0.5f
        }

        return PredictionResult(
            probabilityLegitimate = score,
            totalLatencyNs = System.nanoTime() - t0,
            backend = backend,
            embedding = embed,
        )
    }

    fun predictFused(window: SensorWindow): FusedResult {
        val inertialResult = predict(window)
        val pInertial = inertialResult.probabilityLegitimate
        return FusedResult(inertialResult, pInertial)
    }

    data class FusedResult(
        val inertial: PredictionResult,
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

    fun ownerProfile(): OwnerProfile = ownerProfile

    fun adaptiveBuffer(): AdaptiveAnchorBuffer = adaptiveBuffer

    override fun close() {
        try { interpreter?.close() } catch (e: Exception) {
            Log.w(TAG, "Interpreter close failed: ${e.message}")
        }
    }

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
        val totalLatencyNs: Long,
        val backend: Backend,
        val embedding: FloatArray,
    ) {
        val totalLatencyMs: Float get() = totalLatencyNs / 1_000_000f
    }

    companion object {
        private const val TAG = "InferenceEngine"
        private const val BYTES_PER_FLOAT = 4
        const val EMBED_DIM = 128
        const val N_CHANNELS = SensorWindow.CHANNELS
        const val TIMESTEPS = SensorWindow.TIMESTEPS
        private const val DEFAULT_MODEL_ASSET = "backbone.tflite"

        // Sau ZNORM, điểm zSim là z-score (≈ phân bố quanh 0, đơn vị độ lệch chuẩn của cohort).
        // SCORE_BIAS đặt theo z-score tại điểm EER (xác định offline); SCORE_SCALE điều chỉnh độ dốc.
        // Lưu ý: nếu profile chưa có cohort, zSim = meanSim (thang cosine cũ) → vẫn hoạt động hợp lý.
        private const val SCORE_SCALE = 3.0f
        private const val SCORE_BIAS  = 2.0f

        fun sigmoid(x: Float): Float = 1f / (1f + exp(-x))

        /**
         * ZNORM: tính (mean, std) của điểm cos_mean mà tập anchor tạo ra trên cohort impostor.
         * Gọi MỘT LẦN lúc enroll, lưu vào OwnerProfile. Trả về (0f, 1f) nếu pool rỗng.
         */
        fun fitCohort(anchors: List<FloatArray>, impostorPool: Array<FloatArray>): Pair<Float, Float> {
            if (anchors.isEmpty() || impostorPool.isEmpty()) return 0f to 1f
            val scores = FloatArray(impostorPool.size)
            for (i in impostorPool.indices) {
                var total = 0f
                for (a in anchors) total += staticCosine(impostorPool[i], a)
                scores[i] = total / anchors.size
            }
            var mean = 0f
            for (s in scores) mean += s
            mean /= scores.size
            var varSum = 0f
            for (s in scores) varSum += (s - mean) * (s - mean)
            val std = sqrt((varSum / scores.size).toDouble()).toFloat()
            return mean to std
        }

        private fun staticCosine(a: FloatArray, b: FloatArray): Float {
            var dot = 0f; var na = 0f; var nb = 0f
            for (i in a.indices) { dot += a[i]*b[i]; na += a[i]*a[i]; nb += b[i]*b[i] }
            return (dot / (sqrt(na.toDouble()) * sqrt(nb.toDouble()) + 1e-9)).toFloat()
        }

        fun load(
            context: Context,
            numThreads: Int = 4,
            forceCpuOneThread: Boolean = false,
            mode: ContextMode = ContextMode.loadOrDefault(context),
        ): InferenceEngine {
            val ownerProfile = OwnerProfile(context)
            val adaptiveBuffer = AdaptiveAnchorBuffer(context)

            if (!ownerProfile.hasEnrollment() && adaptiveBuffer.size() > 0) {
                Log.w(TAG, "Owner not enrolled — clearing stale adaptive pool (${adaptiveBuffer.size()} anchors)")
                adaptiveBuffer.clear()
            }

            val scalerPath = ContextMode.assetPath(mode, "scaler_params.json")
            val modelPath  = ContextMode.assetPath(mode, DEFAULT_MODEL_ASSET)

            Log.i(TAG, "Loading model for mode=${mode.key}: $modelPath")

            val scaler = try {
                ScalerParams.loadFromAssets(context, scalerPath)
            } catch (e: Exception) {
                Log.w(TAG, "$scalerPath not found. Using per-window Z-score default.")
                ScalerParams.defaultPerWindow()
            }

            val modelBuffer = tryLoadModel(context, modelPath)
            if (modelBuffer == null) {
                Log.w(TAG, "$modelPath not found. Running in MOCK mode.")
                return InferenceEngine(
                    interpreter = null,
                    scaler = scaler,
                    ownerProfile = ownerProfile,
                    adaptiveBuffer = adaptiveBuffer,
                    isMockMode = true,
                    backend = Backend.MOCK,
                )
            }

            val effectiveThreads = if (forceCpuOneThread) 1 else numThreads
            val chosenBackend = if (effectiveThreads == 1) Backend.CPU_1_THREAD else Backend.CPU_4_THREAD

            val options = Interpreter.Options().apply { setNumThreads(effectiveThreads) }
            val interp = Interpreter(modelBuffer, options)

            validateModelShape(interp)

            Log.i(TAG, "InferenceEngine loaded: backend=$chosenBackend, " +
                    "enrolled=${ownerProfile.hasEnrollment()}, anchors=${ownerProfile.getAnchors().size}")

            return InferenceEngine(
                interpreter = interp,
                scaler = scaler,
                ownerProfile = ownerProfile,
                adaptiveBuffer = adaptiveBuffer,
                isMockMode = false,
                backend = chosenBackend,
            )
        }

        private fun validateModelShape(interp: Interpreter) {
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
