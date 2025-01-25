package com.example.vocalai

import android.content.Context
import android.content.res.AssetManager
import android.util.Log
import com.google.android.gms.tasks.Task
import com.google.android.gms.tasks.TaskCompletionSource
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.metadata.MetadataExtractor
import java.io.BufferedReader
import java.io.FileInputStream
import java.io.IOException
import java.io.InputStream
import java.io.InputStreamReader
import java.nio.ByteBuffer
import java.nio.FloatBuffer
import java.nio.channels.FileChannel
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

private const val TAG = "SpeechApp"

class SpeechRecognition(private val context: Context) {
    private val executorService: ExecutorService = Executors.newCachedThreadPool()


    fun initialize(): Task<Void?> {
        Log.d(TAG,"initialize")

        val task = TaskCompletionSource<Void?>()
        executorService.execute {
            try {
                initializeInterpreter()
                task.setResult(null)
            } catch (e: IOException) {
                task.setException(e)
            }
        }
        return task.task
    }
    private var modelPath="1.tflite"
    private var interpreter: Interpreter? = null

    private fun audioBuffer(intArray: IntArray): FloatArray {
        // Create float array of same length
        val floatArray = FloatArray(intArray.size)

        // Int16 has range of [-32768, 32767]
        // We'll divide by 32768.0f to normalize to [-1.0, 1.0]
        val normalizationFactor = 32768.0f

        for (i in intArray.indices) {
            floatArray[i] = intArray[i] / normalizationFactor
        }

        return floatArray
    }
    private fun initializeInterpreter(): String {
        Log.d(TAG,"initializeInterpreter")
        val assetManager = context.assets
        val model = loadModelFile(assetManager, modelPath)
        val interpreter = Interpreter(model)
        val recResult=testingInput()



        val inputSound=interpreter.getInputTensor(0).shape()
        val modelInputLength = inputSound[0]

        val inputBuffer = FloatBuffer.allocate(modelInputLength)

        val inputResize=interpreter.resizeInput(0, (recResult))
        interpreter.allocateTensors()
        this.interpreter=interpreter
        Log.d(TAG,"INITIALIZED INTERPRETER")

        val output = Array(1) { FloatArray(521) }

        interpreter?.run(inputBuffer, output)
        val scores= interpreter.getOutputTensor(0)
        Log.d(TAG,"the answer was: ${scores.shape()[0]}")

        val resultString=""
        val resultData = output[0]
        val topClassIndex = resultData.indices.maxByOrNull { resultData[it] } ?: -1
        val probability = resultData[topClassIndex]

        Log.d(TAG, "The probability is $probability, the word was $topClassIndex")
        return resultString
    }

    private fun testingInput(): IntArray {
        val sampleRate = 16000
        val duration = 0.975
        val inputSize = (duration * sampleRate).toInt()
        val waveform = intArrayOf(inputSize)
        return waveform
    }

    private fun loadModelFile(assetManager: AssetManager, filename: String): ByteBuffer {
        val fileDescriptor = assetManager.openFd(filename)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    /** Retrieve Map<String, Int> from metadata file */
    private suspend fun readFileInputStream(inputStream: InputStream): List<String> {
        return withContext(Dispatchers.IO) {
            val reader = BufferedReader(InputStreamReader(inputStream))

            val list = mutableListOf<String>()
            var index = 0
            var line = ""
            while (reader.readLine().also { if (it != null) line = it } != null) {
                list.add(line)
                index++
            }

            reader.close()
            list
        }
    }

}
