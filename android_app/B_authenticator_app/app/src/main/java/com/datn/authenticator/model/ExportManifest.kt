package com.datn.authenticator.model

import android.content.Context
import android.util.Log
import org.json.JSONObject

data class ExportManifest(
    val ownerId: String,
    val modelFile: String,
    val scalerFile: String,
    val inputShape: IntArray,
    val outputShape: IntArray,
    val decisionThresholdEer: Float,
    val warningThreshold: Float,
    val trustedThreshold: Float,
    val scoreAggregatorAlpha: Float,
    val scoreAggregatorWindow: Int,
    val exportedAt: String,
    val pipelineVersion: String,
) {
    override fun equals(other: Any?): Boolean = this === other ||
            (other is ExportManifest && ownerId == other.ownerId && exportedAt == other.exportedAt)
    override fun hashCode(): Int = (ownerId + exportedAt).hashCode()

    companion object {
        fun loadFromAssets(context: Context, assetPath: String = "export_manifest.json"): ExportManifest? {
            return try {
                val text = context.assets.open(assetPath).bufferedReader().use { it.readText() }
                val root = JSONObject(text)
                ExportManifest(
                    ownerId = root.getString("owner_id"),
                    modelFile = root.optString("model_file", "model.tflite"),
                    scalerFile = root.optString("scaler_file", "scaler_params.json"),
                    inputShape = root.getJSONArray("input_shape").let { arr ->
                        IntArray(arr.length()) { arr.getInt(it) }
                    },
                    outputShape = root.getJSONArray("output_shape").let { arr ->
                        IntArray(arr.length()) { arr.getInt(it) }
                    },
                    decisionThresholdEer = root.optDouble("decision_threshold_eer", 0.5).toFloat(),
                    warningThreshold = root.optDouble("warning_threshold", 0.45).toFloat(),
                    trustedThreshold = root.optDouble("trusted_threshold", 0.75).toFloat(),
                    scoreAggregatorAlpha = root.optDouble("score_aggregator_alpha", 0.8).toFloat(),
                    scoreAggregatorWindow = root.optInt("score_aggregator_window", 5),
                    exportedAt = root.optString("exported_at", "unknown"),
                    pipelineVersion = root.optString("export_pipeline_version", "unknown"),
                )
            } catch (e: Exception) {
                Log.w("ExportManifest", "Cannot load $assetPath: ${e.message}")
                null
            }
        }
    }
}
