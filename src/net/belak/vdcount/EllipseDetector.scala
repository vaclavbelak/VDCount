package net.belak.vdcount

import java.io.{IOException, File}

import org.opencv.core._
import org.opencv.imgcodecs.Imgcodecs
import org.opencv.imgproc.Imgproc

/**
  * Created by vaclav on 6.3.16.
  */
class EllipseDetector(srcDir: File, outDir: File) {

  @throws(classOf[IOException])
  def convertAll {
    srcDir.listFiles().filter(_.getName.toLowerCase.endsWith("jpg")).foreach(detect(_))
  }

  @throws(classOf[IOException])
  def detect(imgFile: File) {
    val image: Mat = Imgcodecs.imread(imgFile.getAbsolutePath, Imgcodecs.CV_LOAD_IMAGE_UNCHANGED)
    val outImage: Mat = image.clone
    Imgproc.bilateralFilter(image, outImage, 10, 30, 30)
    Imgproc.cvtColor(outImage, outImage, Imgproc.COLOR_RGB2GRAY)
    Imgproc.threshold(outImage, outImage, 80, 255, Imgproc.THRESH_BINARY)
    val contours = new java.util.ArrayList[MatOfPoint]
    val hierarchy = new Mat
    Imgproc.findContours(outImage.clone, contours, hierarchy, Imgproc.RETR_TREE, Imgproc.CHAIN_APPROX_SIMPLE, new Point(0, 0))
    import scala.collection.JavaConversions.iterableAsScalaIterable
    val minEllipse = contours.filter(_.toArray.length >= 5).map(contour => Imgproc.fitEllipse(new org.opencv.core.MatOfPoint2f(contour.toArray: _*))).toList

    val drawing: Mat = image.clone
    val color: Scalar = new Scalar(0, 153, 0)
    val vdir: File = new File(outDir, imgName(imgFile) + "_vd")
    vdir.mkdirs
    val innerBoxSize: Int = 12

    for (i <- minEllipse.indices) {
      val ellipse: RotatedRect = minEllipse(i)
      val outerBox = boxDims(innerBoxSize, ellipse, image)
      val box: Mat = image.submat(outerBox.rowStart, outerBox.rowEnd, outerBox.colStart, outerBox.colEnd)
      val boxName: String = vdir.getAbsolutePath + File.separator + imgName(imgFile) + "_box" + i
      varyExamples(box, boxName, innerBoxSize, 90)
      Imgcodecs.imwrite(boxName + ".png", box.submat(box.width / 2 - innerBoxSize, box.width / 2 + innerBoxSize,
        box.height / 2 - innerBoxSize, box.height / 2 + innerBoxSize))
      Imgproc.ellipse(drawing, ellipse, color, 2, 8)
    }

    Imgcodecs.imwrite(outDir.getAbsolutePath + File.separator + imgName(imgFile) + "_test_bf_.png", outImage)
    Imgcodecs.imwrite(outDir.getAbsolutePath + File.separator + imgName(imgFile) + "_test_bf_contours.png", drawing)
  }

  private def varyExamples(image: Mat, name: String, innerBoxSize: Int, degIncrement: Int) {
    for (deg <- degIncrement to 360 by degIncrement) {
      val rotatedImg: Mat = rotate(image, deg, innerBoxSize)
      Imgcodecs.imwrite(name + "_rot" + deg + ".png", rotatedImg)
    }
  }

  private def rotate(image: Mat, degrees: Int, innerBoxSize: Int): Mat = {
    val center: Point = new Point(image.width / 2, image.height / 2)
    val rotationMat: Mat = Imgproc.getRotationMatrix2D(center, degrees, 1)
    val result: Mat = new Mat
    Imgproc.warpAffine(image, result, rotationMat, new Size(image.width, image.height))

    result.submat(center.y.toInt - innerBoxSize, center.y.toInt + innerBoxSize,
      center.x.toInt - innerBoxSize, center.x.toInt + innerBoxSize)
  }

  private def imgName(img: File): String = img.getName.split("\\.").head

  private def boxDims(innerBoxSize: Int, ellipse: RotatedRect, image: Mat): BoxDim = {
    val d = Math.ceil(Math.sqrt(2 * Math.pow(innerBoxSize, 2))).toInt
    val x = ellipse.center.x.round.toInt
    val y = ellipse.center.y.round.toInt

    val (rowStart: Int, rowEnd: Int) =
      if (x - d < 0) (0, 2 * d)
      else if (x + d >= image.cols) (image.cols - 1 - 2 * d, image.cols - 1)
      else (x - 2, x + d)

    val (colStart: Int, colEnd: Int) =
      if (y - d < 0) (0, 2 * d)
      else if (y + d >= image.rows) (image.rows - 1 - 2 * d, image.rows - 1)
      else (y - d, y + d)

    new BoxDim(colStart, colEnd, rowStart, rowEnd)
  }
}

case class BoxDim(val rowStart: Int, val rowEnd: Int, val colStart: Int, val colEnd: Int)
