package net.belak.vdcount

import java.io.{PrintWriter, IOException, File}

import org.opencv.core._
import org.opencv.imgcodecs.Imgcodecs
import org.opencv.imgproc.Imgproc
import scala.collection.JavaConversions.iterableAsScalaIterable

/**
  * Created by vaclav on 6.3.16.
  */
class EllipseDetector(srcDir: File, outDir: File) {

  val MAX_VD_WIDTH = 50
  val MIN_VD_WIDTH = 5

  val MAX_VD_HEIGHT = 50
  val MIN_VD_HEIGHT = 5

  val MAX_VD_HW_RATIO = 1.5
  val MIN_VD_HW_RATIO = 1.1

  val INNER_PATCH_SIZE = 12

  @throws(classOf[IOException])
  def convertAll {
    srcDir.listFiles().filter(_.getName.toLowerCase.endsWith("jpg")).foreach(detect(_))
  }

  @throws(classOf[IOException])
  def detect(imgFile: File) {
    println(s"Converting ${imgFile.getName}")

    val image: Mat = Imgcodecs.imread(imgFile.getAbsolutePath, Imgcodecs.CV_LOAD_IMAGE_UNCHANGED)
    val outImage: Mat = image.clone
//    Imgproc.bilateralFilter(image, outImage, 10, 30, 30)
    Imgproc.cvtColor(outImage, outImage, Imgproc.COLOR_RGB2GRAY)
    Imgproc.threshold(outImage, outImage, 80, 255, Imgproc.THRESH_BINARY)
    val contours = new java.util.ArrayList[MatOfPoint]
    val hierarchy = new Mat
    Imgproc.findContours(outImage.clone, contours, hierarchy, Imgproc.RETR_TREE, Imgproc.CHAIN_APPROX_SIMPLE, new Point(0, 0))
    val minEllipse = contours.filter(_.toArray.length >= 5).map(contour => Imgproc.fitEllipse(new MatOfPoint2f(contour.toArray: _*))).toList

    val drawing: Mat = image.clone
    val color: Scalar = new Scalar(0, 153, 0)
    val vdir: File = new File(outDir, imgName(imgFile) + "_vd")
    vdir.mkdirs

    val patchStats = new PrintWriter(new File(vdir, "patch-stats.csv"))
    patchStats.println("patch,height,width")
    for (i <- minEllipse.indices if acceptEllipse(minEllipse(i))) {
      val ellipse: RotatedRect = minEllipse(i)
      val outerPatch = patchDims(INNER_PATCH_SIZE, ellipse, image)
      val patch: Mat = image.submat(outerPatch.rowStart, outerPatch.rowEnd, outerPatch.colStart, outerPatch.colEnd)
      val patchName: String = vdir.getAbsolutePath + File.separator + imgName(imgFile) + "_patch" + i
      varyExamples(patch, patchName, INNER_PATCH_SIZE, 90)
      Imgcodecs.imwrite(patchName + ".png", patch.submat(patch.width / 2 - INNER_PATCH_SIZE, patch.width / 2 + INNER_PATCH_SIZE,
        patch.height / 2 - INNER_PATCH_SIZE, patch.height / 2 + INNER_PATCH_SIZE))
      Imgproc.ellipse(drawing, ellipse, color, 2, 8)
      patchStats.println(s"$i,${ellipse.size.height},${ellipse.size.width}")
    }
    patchStats.close()

    Imgcodecs.imwrite(outDir.getAbsolutePath + File.separator + imgName(imgFile) + "_test_bf_.png", outImage)
    Imgcodecs.imwrite(outDir.getAbsolutePath + File.separator + imgName(imgFile) + "_test_bf_contours.png", drawing)
  }

  private def acceptEllipse(ellipse: RotatedRect) = {
    val HWRatio = ellipse.size.height / ellipse.size.width

    ellipse.size.height >= MIN_VD_HEIGHT &
      ellipse.size.height <= MAX_VD_HEIGHT & ellipse.size.width >= MIN_VD_WIDTH & ellipse.size.width <= MAX_VD_WIDTH &
      HWRatio >= MIN_VD_HW_RATIO & HWRatio <= MAX_VD_HW_RATIO
  }

  private def varyExamples(image: Mat, name: String, innerPatchSize: Int, degIncrement: Int) {
    for (deg <- degIncrement until 360 by degIncrement) {
      val rotatedImg: Mat = rotate(image, deg, innerPatchSize)
      Imgcodecs.imwrite(name + "_rot" + deg + ".png", rotatedImg)
    }
  }

  private def rotate(image: Mat, degrees: Int, innerPatchSize: Int): Mat = {
    val center = new Point(image.width / 2, image.height / 2)
    val rotationMat = Imgproc.getRotationMatrix2D(center, degrees, 1)
    val result = new Mat
    Imgproc.warpAffine(image, result, rotationMat, new Size(image.width, image.height))

    result.submat(center.y.toInt - innerPatchSize, center.y.toInt + innerPatchSize,
      center.x.toInt - innerPatchSize, center.x.toInt + innerPatchSize)
  }

  private def imgName(img: File): String = img.getName.split("\\.").head

  private def patchDims(innerPatchSize: Int, ellipse: RotatedRect, image: Mat): PatchSize = {
    val d = Math.ceil(Math.sqrt(2 * Math.pow(innerPatchSize, 2))).toInt
    val x = ellipse.center.x.round.toInt
    val y = ellipse.center.y.round.toInt

    val (colStart, colEnd) =
      if (x - d < 0) (0, 2 * d)
      else if (x + d >= image.cols) (image.cols - 1 - 2 * d, image.cols - 1)
      else (x - d, x + d)

    val (rowStart, rowEnd) =
      if (y - d < 0) (0, 2 * d)
      else if (y + d >= image.rows) (image.rows - 1 - 2 * d, image.rows - 1)
      else (y - d, y + d)

    new PatchSize(rowStart, rowEnd, colStart, colEnd)
  }
}

case class PatchSize(val rowStart: Int, val rowEnd: Int, val colStart: Int, val colEnd: Int) {
  def height = rowEnd - rowStart
  def width = colEnd - colStart
}
