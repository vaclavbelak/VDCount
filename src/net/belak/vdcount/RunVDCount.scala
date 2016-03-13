package net.belak.vdcount

import java.io.File
import org.opencv.core.Core

object RunVDCount {

  def main(args: Array[String]) {
    System.loadLibrary(Core.NATIVE_LIBRARY_NAME)
    Core.setNumThreads(2)
    val eDetector = new EllipseDetector(new File(args(0)), new File(args(1)))
    eDetector.convertAll()
//    eDetector.detect(new File("""C:\Users\belak\Documents\Personal\Vcely\VDCount\VDCount\resources\VDSource\IMG_1448.JPG"""), None)
//    eDetector.detect(new File("""C:\Users\belak\Documents\Personal\Vcely\VDCount\VDCount\resources\VDSource\IMG_1447.JPG"""), None)
//    eDetector.detect(new File("""C:\Users\belak\Documents\Personal\Vcely\VDCount\VDCount\resources\VDSource\IMG_1319.JPG"""), None)
  }
}
