package com.datn.authenticator.inference

import android.content.Context
import android.util.Log
import java.io.InputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Minimal .npy reader for 2-D float32 arrays stored in assets.
 *
 * Supports NumPy format v1.0 / v2.0, row-major (C order), dtype '<f4' (float32 LE).
 * Sufficient for loading impostor_pool_inertial.npy and impostor_pool_touch.npy.
 */
object NpyReader {

    private const val TAG = "NpyReader"
    private val MAGIC = byteArrayOf(0x93.toByte(), 'N'.code.toByte(), 'U'.code.toByte(),
        'M'.code.toByte(), 'P'.code.toByte(), 'Y'.code.toByte())

    /**
     * Read a 2-D float32 .npy from assets.
     * Returns Array<FloatArray> where result[row][col].
     */
    fun readFloat32_2D(context: Context, assetPath: String): Array<FloatArray> {
        context.assets.open(assetPath).use { stream ->
            return parseStream(stream, assetPath)
        }
    }

    private fun parseStream(stream: InputStream, name: String): Array<FloatArray> {
        // Verify magic
        val magic = ByteArray(6)
        check(stream.read(magic) == 6 && magic.contentEquals(MAGIC)) {
            "$name: not a valid .npy file"
        }

        val major = stream.read()
        stream.read() // minor (ignored)

        // Header length: 2 bytes LE for v1, 4 bytes LE for v2
        val headerLen = if (major == 1) {
            val lo = stream.read()
            val hi = stream.read()
            lo or (hi shl 8)
        } else {
            val b = ByteArray(4)
            stream.read(b)
            ByteBuffer.wrap(b).order(ByteOrder.LITTLE_ENDIAN).int
        }

        val headerBytes = ByteArray(headerLen)
        stream.read(headerBytes)
        val header = String(headerBytes, Charsets.US_ASCII)

        val shape = parseShape(header)
        check(shape.size == 2) { "$name: expected 2-D array, got shape $shape" }
        check(header.contains("'<f4'") || header.contains("\"<f4\"")) {
            "$name: only float32 little-endian ('<f4') is supported"
        }
        check(!header.contains("True")) { "$name: Fortran order not supported" }

        val rows = shape[0]
        val cols = shape[1]
        val dataBytes = ByteArray(rows * cols * 4)
        var offset = 0
        while (offset < dataBytes.size) {
            val n = stream.read(dataBytes, offset, dataBytes.size - offset)
            if (n < 0) break
            offset += n
        }
        check(offset == dataBytes.size) { "$name: truncated data" }

        val buf = ByteBuffer.wrap(dataBytes).order(ByteOrder.LITTLE_ENDIAN)
        return Array(rows) { FloatArray(cols) { buf.float } }.also {
            Log.i(TAG, "Loaded $name: shape=[$rows, $cols]")
        }
    }

    /** Parse shape from numpy dict header string, e.g. "'shape': (100, 128)," */
    private fun parseShape(header: String): List<Int> {
        val m = Regex("""'shape'\s*:\s*\(([^)]*)\)""").find(header)
            ?: error("Cannot parse shape from header: $header")
        return m.groupValues[1].split(",")
            .map { it.trim() }
            .filter { it.isNotEmpty() }
            .map { it.toInt() }
    }
}
