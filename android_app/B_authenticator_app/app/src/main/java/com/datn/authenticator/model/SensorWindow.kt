package com.datn.authenticator.model

data class SensorWindow(
    val data: FloatArray,
    val startTimestampMs: Long,
    val endTimestampMs: Long,
    val activityHint: String = "unknown",
) {
    init {
        require(data.size == TIMESTEPS * CHANNELS) {
            "SensorWindow.data must be ${TIMESTEPS * CHANNELS} floats, got ${data.size}"
        }
    }

    val durationMs: Long get() = endTimestampMs - startTimestampMs

    fun toMatrix(): Array<FloatArray> = Array(TIMESTEPS) { t ->
        FloatArray(CHANNELS) { c -> data[t * CHANNELS + c] }
    }

    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is SensorWindow) return false
        return startTimestampMs == other.startTimestampMs &&
                endTimestampMs == other.endTimestampMs &&
                data.contentEquals(other.data)
    }

    override fun hashCode(): Int {
        var result = data.contentHashCode()
        result = 31 * result + startTimestampMs.hashCode()
        result = 31 * result + endTimestampMs.hashCode()
        return result
    }

    companion object {
        const val TIMESTEPS = 200
        const val CHANNELS = 9
        const val SAMPLE_RATE_HZ = 50

        const val CH_ACC_X = 0
        const val CH_ACC_Y = 1
        const val CH_ACC_Z = 2
        const val CH_GYRO_X = 3
        const val CH_GYRO_Y = 4
        const val CH_GYRO_Z = 5
        const val CH_MAG_X = 6
        const val CH_MAG_Y = 7
        const val CH_MAG_Z = 8
    }
}
