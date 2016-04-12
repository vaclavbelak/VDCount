package net.belak.vdcount
import java.io.File

import org.opencv.core.{Core, Size, Point, Mat}
import org.opencv.imgcodecs.Imgcodecs
import org.opencv.imgproc.Imgproc

/**
  * Rotates all images in the input dir (arg1) and stores them to the output dir (arg2). If the output dir does not
  * exist, it is created.
  * If arg3 is true, then the original file is deleted (false by default).
  * @author vaclav@belak.net
  */
object RotatePatches extends App with Constants {

  val srcDir = new File(args(0))
  val outDir = new File(args(1))
  outDir.mkdirs()
  System.loadLibrary(Core.NATIVE_LIBRARY_NAME)
  Core.setNumThreads(3)

  def rotate(image: Mat, degrees: Int): Mat = {
    val center = new Point(image.width / 2, image.height / 2)
    val rotationMat = Imgproc.getRotationMatrix2D(center, degrees, 1)
    val result = new Mat
    Imgproc.warpAffine(image, result, rotationMat, new Size(image.width, image.height))

    result.submat(center.y.toInt - INNER_PATCH_SIZE / 2, center.y.toInt + INNER_PATCH_SIZE / 2,
      center.x.toInt - INNER_PATCH_SIZE / 2, center.x.toInt + INNER_PATCH_SIZE / 2)
  }

  def varyExamples(image: Mat, name: String, degIncrement: Int) {
    for (deg <- degIncrement to 360 by degIncrement) {
      val rotatedImg: Mat = rotate(image, deg)
      Imgcodecs.imwrite(name + "_rot" + deg + ".png", rotatedImg)
    }
  }

  def imgName(srcImgFile: File) = outDir.getAbsolutePath + File.separator + srcImgFile.getName.split("\\.")(0)

  srcDir.listFiles().filter(_.getName.endsWith(".png")).foreach(imgFile => {
    varyExamples(Imgcodecs.imread(imgFile.getAbsolutePath), imgName(imgFile), 360)
    if (args.length > 2 && args(2) == "true") imgFile.delete()
  })
}