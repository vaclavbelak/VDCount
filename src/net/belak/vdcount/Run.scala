package net.belak.vdcount

import java.io.File
import org.opencv.core.Core

object Run {

  def main(args: Array[String]) {
    System.loadLibrary(Core.NATIVE_LIBRARY_NAME)
    Core.setNumThreads(2)
    val eDetector = new EllipseDetector(new File("/home/vaclav/IdeaProjects/VDCount/resources/VDSource"),
      new File("/home/vaclav/IdeaProjects/VDCount/resources/VDTest"))
    eDetector.convertAll
  }
}
