package com.example.vocalai.playback

import java.io.File

interface AudioPlayer {
    fun playFile(file: File)
    fun stop()
}