package com.rrrfffrrr.unity.mediapipe.handtracking

import android.app.Activity
import android.opengl.EGL14
import android.util.Log
import com.google.mediapipe.formats.proto.LandmarkProto
import com.google.mediapipe.framework.*
import com.google.mediapipe.glutil.EglManager
import java.util.concurrent.atomic.AtomicBoolean

@Suppress("unused")
class Graph constructor(private val context: Activity, private val callback: DataCallback) {
    companion object {
        const val TAG = "hand_tracking"

        const val BINARY_GRAPH_NAME = "hand_tracking_mobile_gpu.binarypb"

        const val INPUT_VIDEO_STREAM_NAME = "input_video"
        const val INPUT_NUM_HANDS_SIDE_PACKET_NAME = "num_hands" // Int

        const val OUTPUT_VIDEO_STREAM_NAME = "output_video"
        const val OUTPUT_LANDMARKS_STREAM_NAME = "hand_landmarks" // std::vector<NormalizedLandmarkList>

        const val MAX_NUM_HANDS = 8

        init {
            System.loadLibrary("mediapipe_jni");
            System.loadLibrary("opencv_java3");
        }
    }

    private val started = AtomicBoolean(false)

    private val eglManager = EglManager(EGL14.eglGetCurrentContext())
    private val graph = Graph()

    private val packetCreator: PacketCreator
    private val inputSidePackets: MutableMap<String, Packet> = HashMap()

    init {
        AndroidAssetUtil.initializeNativeAssetManager(context)

        graph.loadBinaryGraph(AndroidAssetUtil.getAssetBytes(context.assets, BINARY_GRAPH_NAME))
        packetCreator = AndroidPacketCreator(graph)

        graph.setParentGlContext(eglManager.nativeContext)

        inputSidePackets[INPUT_NUM_HANDS_SIDE_PACKET_NAME] = packetCreator.createInt32(MAX_NUM_HANDS)
        graph.setInputSidePackets(inputSidePackets)

        ///region callbacks
        graph.addPacketCallback(OUTPUT_LANDMARKS_STREAM_NAME) { packet ->
            Log.v(TAG, "Received multi-hand landmarks packet.")
            try {
                var hands = PacketGetter.getProtoVector(packet, LandmarkProto.NormalizedLandmarkList.parser())
                val list = mutableListOf<Float>()
                hands.forEach { hand ->
                    hand.landmarkList.forEach { landmark ->
                        list.add(landmark.x)
                        list.add(landmark.y)
                        list.add(landmark.z)
                        list.add(landmark.presence)
                        list.add(landmark.visibility)
                    }
                }
                callback.onData(list.toFloatArray())
            } catch (e: Exception) {
                Log.e(TAG, "Error while get packet: ", e)
            }
        }
        ///endregion
    }

    private fun startGraph() : Boolean {
        if (!started.getAndSet(true)) {
            graph.startRunningGraph()
            return true
        }
        return false
    }

    ///region Unity interface
    private var timestamp: Long = 0
    @JvmName("Input")
    fun input(texture: Int, width: Int, height: Int) {
        startGraph()

        val appTexture = AppTextureFrame(texture, width, height)
        appTexture.timestamp = timestamp++

        graph.addConsumablePacketToInputStream(INPUT_VIDEO_STREAM_NAME, packetCreator.createGpuBuffer(appTexture), appTexture.timestamp)
    }

    @JvmName("Close")
    fun close() {
        if (started.get()) {
            try {
                graph.closeAllPacketSources()
                graph.waitUntilGraphDone()
            } catch (e: Exception) {
                Log.e(TAG, "Mediapipe wrapper error: ", e)
            }

            try {
                graph.tearDown()
            } catch (e: Exception) {
                Log.e(TAG, "Mediapipe wrapper error: ", e)
            }
        }
    }
    ///endregion
}