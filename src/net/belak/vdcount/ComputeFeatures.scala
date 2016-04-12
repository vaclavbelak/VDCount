package net.belak.vdcount

import java.io.{PrintWriter, File}

import org.opencv.core.{MatOfFloat, Size, Core}
import org.opencv.imgcodecs.Imgcodecs
import org.opencv.objdetect.HOGDescriptor

/**
  * Created by vaclav on 18.3.16.
  */
object ComputeFeatures extends App {

  System.loadLibrary(Core.NATIVE_LIBRARY_NAME)

  val hd = new HOGDescriptor(new Size(24,24), new Size(16, 16), new Size(8, 8), new Size(8, 8), 9)
  val inDir = new File(args(0))

  val featureFile = new PrintWriter(new File(inDir, "features.csv"))

  def process(label: String) {
    assert(label == "positive" || label == "negative")
    for (imageFile <- new File(args(0), s"$label/rotated/").listFiles() if imageFile.getName.endsWith(".png")) {
      val features = new MatOfFloat
      val image = Imgcodecs.imread(imageFile.getAbsolutePath, Imgcodecs.CV_LOAD_IMAGE_GRAYSCALE)
      hd.compute(image, features)
      featureFile.println(imageFile.getAbsolutePath + "," + label + "," + features.toArray.mkString(","))
    }
  }

  process("positive")
  process("negative")

  featureFile.close()
}
