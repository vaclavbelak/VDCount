package net.belak.vdcount

import java.io.File
import org.opencv.core.Core

object RunEllipseDetector {

  def main(args: Array[String]) {
    System.loadLibrary(Core.NATIVE_LIBRARY_NAME)
    Core.setNumThreads(2)
    val eDetector = new VarroaDetector(new File(args(0)), new File(args(1)))
    eDetector.convertAll()
  }
}
