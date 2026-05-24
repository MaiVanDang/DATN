package com.datn.authenticator.model

import android.content.Context
import org.json.JSONObject
import kotlin.math.max
import kotlin.math.sqrt

data class ScalerParams(
    val mode: String,
    val channelOrder: List<String>,
    val epsilon: Float,
    val fittedMean: FloatArray? = null,
    val fittedScale: FloatArray? = null,
    val warnIfOutside: Pair<Float, Float>? = null,
) {
    fun normalize(rawData: FloatArray): FloatArray {
        require(rawData.size == SensorWindow.TIMESTEPS * SensorWindow.CHANNELS)
        return when (mode) {
            "per_window_zscore" -> normalizePerWindow(rawData)
            "fitted_standard_scaler" -> normalizeFitted(rawData)
            else -> error("Unknown normalization mode: $mode")
        }
    }

    private fun normalizePerWindow(raw: FloatArray): FloatArray {
        val out = FloatArray(raw.size)
        val sums = FloatArray(SensorWindow.CHANNELS)
        val sumSq = FloatArray(SensorWindow.CHANNELS)

        for (t in 0 until SensorWindow.TIMESTEPS) {
            for (c in 0 until SensorWindow.CHANNELS) {
                val v = raw[t * SensorWindow.CHANNELS + c]
                sums[c] += v
                sumSq[c] += v * v
            }
        }
        val mean = FloatArray(SensorWindow.CHANNELS) { sums[it] / SensorWindow.TIMESTEPS }
        val std = FloatArray(SensorWindow.CHANNELS) {
            val variance = sumSq[it] / SensorWindow.TIMESTEPS - mean[it] * mean[it]
            sqrt(max(variance, 0f))
        }

        for (t in 0 until SensorWindow.TIMESTEPS) {
            for (c in 0 until SensorWindow.CHANNELS) {
                val idx = t * SensorWindow.CHANNELS + c
                out[idx] = (raw[idx] - mean[c]) / (std[c] + epsilon)
            }
        }
        return out
    }

    private fun normalizeFitted(raw: FloatArray): FloatArray {
        val mean = checkNotNull(fittedMean) { "fittedMean missing" }
        val scale = checkNotNull(fittedScale) { "fittedScale missing" }
        val out = FloatArray(raw.size)
        for (t in 0 until SensorWindow.TIMESTEPS) {
            for (c in 0 until SensorWindow.CHANNELS) {
                val idx = t * SensorWindow.CHANNELS + c
                out[idx] = (raw[idx] - mean[c]) / (scale[c] + epsilon)
            }
        }
        return out
    }

    fun isLikelyDriftedFromTraining(normalized: FloatArray): Boolean {
        val (lo, hi) = warnIfOutside ?: return false
        return normalized.any { it < lo || it > hi }
    }

    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is ScalerParams) return false
        return mode == other.mode &&
                channelOrder == other.channelOrder &&
                epsilon == other.epsilon &&
                fittedMean.contentEqualsNullable(other.fittedMean) &&
                fittedScale.contentEqualsNullable(other.fittedScale)
    }

    override fun hashCode(): Int {
        var r = mode.hashCode()
        r = 31 * r + channelOrder.hashCode()
        r = 31 * r + epsilon.hashCode()
        r = 31 * r + (fittedMean?.contentHashCode() ?: 0)
        r = 31 * r + (fittedScale?.contentHashCode() ?: 0)
        return r
    }

    companion object {
        fun loadFromAssets(context: Context, assetPath: String = "scaler_params.json"): ScalerParams {
            val text = context.assets.open(assetPath).bufferedReader().use { it.readText() }
            val root = JSONObject(text)
            val mode = root.getString("normalization_mode")
            val chOrderJson = root.getJSONArray("channel_order")
            val channelOrder = List(chOrderJson.length()) { chOrderJson.getString(it) }
            val epsilon = root.optDouble("epsilon", 1e-6).toFloat()

            var fittedMean: FloatArray? = null
            var fittedScale: FloatArray? = null
            if (mode == "fitted_standard_scaler" && root.has("fitted_params")) {
                val fp = root.getJSONObject("fitted_params")
                val meanArr = fp.getJSONArray("mean")
                val scaleArr = fp.getJSONArray("scale")
                fittedMean = FloatArray(meanArr.length()) { meanArr.getDouble(it).toFloat() }
                fittedScale = FloatArray(scaleArr.length()) { scaleArr.getDouble(it).toFloat() }
            }

            var warnRange: Pair<Float, Float>? = null
            val expRanges = root.optJSONObject("expected_input_ranges_after_normalization")
            if (expRanges != null) {
                val w = expRanges.optJSONArray("warn_if_outside")
                if (w != null && w.length() == 2) {
                    warnRange = w.getDouble(0).toFloat() to w.getDouble(1).toFloat()
                }
            }

            return ScalerParams(
                mode = mode,
                channelOrder = channelOrder,
                epsilon = epsilon,
                fittedMean = fittedMean,
                fittedScale = fittedScale,
                warnIfOutside = warnRange,
            )
        }

        fun defaultPerWindow(): ScalerParams = ScalerParams(
            mode = "per_window_zscore",
            channelOrder = listOf(
                "acc_x", "acc_y", "acc_z",
                "gyro_x", "gyro_y", "gyro_z",
                "mag_x", "mag_y", "mag_z",
            ),
            epsilon = 1e-6f,
            warnIfOutside = -6.0f to 6.0f,
        )
    }
}

private fun FloatArray?.contentEqualsNullable(other: FloatArray?): Boolean {
    if (this == null && other == null) return true
    if (this == null || other == null) return false
    return this.contentEquals(other)
}
