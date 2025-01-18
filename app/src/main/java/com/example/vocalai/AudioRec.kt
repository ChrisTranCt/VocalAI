package com.example.vocalai

import java.io.File

interface AudioRec {
    fun start(outputFile: File)
    fun stop()
}