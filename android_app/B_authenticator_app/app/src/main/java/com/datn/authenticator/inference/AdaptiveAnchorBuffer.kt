package com.datn.authenticator.inference

import android.content.Context
import android.util.Log
import java.io.DataInputStream
import java.io.DataOutputStream
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.util.ArrayDeque

class AdaptiveAnchorBuffer(
    context: Context,
    private val maxAdaptive: Int    = 20,
    private val adaptThreshold: Float = 0.92f,
    private val requiredStreak: Int = 5,
    private val cooldownMs: Long    = 5 * 60_000L,    // sau fallback
    private val rateLimitMs: Long   = 5 * 60_000L,    // giữa các add
) {
    private val file: File = File(context.filesDir, FILE_NAME)
    private val pool = ArrayDeque<TimedAnchor>()

    @Volatile private var streak = 0
    @Volatile private var lastAddMs = 0L
    @Volatile private var lastFallbackMs = 0L

    init { loadFromDisk() }

    @Synchronized
    fun maybeAdd(embed: FloatArray, fusedScore: Float): Boolean {
        val now = System.currentTimeMillis()

        if (now - lastFallbackMs < cooldownMs) {
            streak = 0
            return false
        }

        if (fusedScore < adaptThreshold) {
            streak = 0
            return false
        }

        streak++
        if (streak < requiredStreak) return false

        if (now - lastAddMs < rateLimitMs) return false

        if (pool.size >= maxAdaptive) pool.removeFirst()
        pool.addLast(TimedAnchor(embed.copyOf(), now))
        lastAddMs = now
        streak = 0

        try {
            saveToDisk()
        } catch (e: Exception) {
            Log.w(TAG, "Persist failed: ${e.message}")
        }
        Log.i(TAG, "Adaptive anchor added " +
                "(pool=${pool.size}/$maxAdaptive, fused=${"%.3f".format(fusedScore)})")
        return true
    }

    @Synchronized
    fun onFallbackTriggered() {
        lastFallbackMs = System.currentTimeMillis()
        streak = 0
        Log.i(TAG, "Fallback triggered — adapt cooldown ${cooldownMs / 1000}s")
    }

    @Synchronized
    fun all(): List<FloatArray> = pool.map { it.embed }

    @Synchronized
    fun size(): Int = pool.size

    @Synchronized
    fun clear() {
        pool.clear()
        streak = 0
        lastAddMs = 0L
        if (file.exists()) file.delete()
        Log.i(TAG, "Adaptive pool cleared")
    }

    private fun loadFromDisk() {
        if (!file.exists()) return
        try {
            DataInputStream(FileInputStream(file)).use { dis ->
                val magic = dis.readInt()
                if (magic != MAGIC) {
                    Log.w(TAG, "Bad magic 0x${magic.toString(16)} — discarding")
                    return
                }
                val n = dis.readInt()
                val dim = dis.readInt()
                if (dim != InferenceEngine.EMBED_DIM) {
                    Log.w(TAG, "Embed dim mismatch ($dim vs ${InferenceEngine.EMBED_DIM}) — discarding")
                    return
                }
                repeat(n) {
                    val ts = dis.readLong()
                    val v = FloatArray(dim) { dis.readFloat() }
                    pool.addLast(TimedAnchor(v, ts))
                }
            }
            Log.i(TAG, "Loaded ${pool.size} adaptive anchors")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to load: ${e.message}")
            pool.clear()
        }
    }

    private fun saveToDisk() {
        DataOutputStream(FileOutputStream(file)).use { dos ->
            dos.writeInt(MAGIC)
            dos.writeInt(pool.size)
            dos.writeInt(InferenceEngine.EMBED_DIM)
            for (a in pool) {
                dos.writeLong(a.tMs)
                for (v in a.embed) dos.writeFloat(v)
            }
        }
    }

    private data class TimedAnchor(val embed: FloatArray, val tMs: Long)

    companion object {
        private const val TAG = "AdaptiveAnchor"
        private const val FILE_NAME = "adaptive_anchors.bin"
        private const val MAGIC = 0xADAF0001.toInt()
    }
}
