package net.belak.vdcount

import java.io.{IOException, File}

import org.opencv.core.Core

object Run {

  def main(args: Array[String]) {
    System.loadLibrary(Core.NATIVE_LIBRARY_NAME)
    Core.setNumThreads(2)
    val eDetector: EllipseDetector = new EllipseDetector(new File("/home/vaclav/IdeaProjects/VDCount/resources/VDSource"),
      new File("/home/vaclav/IdeaProjects/VDCount/resources/VDTest"))
    eDetector.convertAll
  }
}
