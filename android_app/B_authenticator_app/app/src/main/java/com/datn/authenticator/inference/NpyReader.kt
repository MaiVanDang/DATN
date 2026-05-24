package com.datn.authenticator.inference

import android.content.Context
import android.util.Log
import java.io.InputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder

object NpyReader {
    private const val TAG = "NpyReader"
    private val MAGIC = byteArrayOf(0x93.toByte(), 'N'.code.toByte(), 'U'.code.toByte(),
        'M'.code.toByte(), 'P'.code.toByte(), 'Y'.code.toByte())

    fun readFloat32_2D(context: Context, assetPath: String): Array<FloatArray> {
        context.assets.open(assetPath).use { stream ->
            return parseStream(stream, assetPath)
        }
    }

    private fun parseStream(stream: InputStream, name: String): Array<FloatArray> {
        val magic = ByteArray(6)
        check(stream.read(magic) == 6 && magic.contentEquals(MAGIC)) {
            "$name: not a valid .npy file"
        }

        val major = stream.read()
        stream.read()

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

    private fun parseShape(header: String): List<Int> {
        val m = Regex("""'shape'\s*:\s*\(([^)]*)\)""").find(header)
            ?: error("Cannot parse shape from header: $header")
        return m.groupValues[1].split(",")
            .map { it.trim() }
            .filter { it.isNotEmpty() }
            .map { it.toInt() }
    }
}
