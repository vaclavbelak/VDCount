package net.belak.vdcount

import java.io.{PrintWriter, IOException, File}

import org.opencv.core._
import org.opencv.imgcodecs.Imgcodecs
import org.opencv.imgproc.Imgproc
import scala.collection.JavaConverters.iterableAsScalaIterable

/**
  * Detects ellipses that resemble Varroa Destructor bee mite.
  * @param srcDir Where the source images are stored. Have to be JPEGs.
  * @param outDir Where the output files should be stored?
  * @author vaclav@belak.net
  */
class VarroaDetector(srcDir: File, outDir: File) extends Constants {

  if (!outDir.exists()) outDir.mkdirs()

  /**
    * Converts all images found in the srcDir.
    */
  def convertAll() {
    val patchStats = new PrintWriter(new File(outDir, "patch-stats.csv"))
    srcDir.listFiles().filter(_.getName.toLowerCase.endsWith("jpg")).foreach(file => detect(file, Option(patchStats)))
    patchStats.close()
  }

  /**
    * Converts one file.
    * @param imgFile The file to be converted.
    */
  def detect(imgFile: File, patchStats: Option[PrintWriter]) {
    println(s"Converting ${imgFile.getName}")

    val image: Mat = Imgcodecs.imread(imgFile.getAbsolutePath, Imgcodecs.CV_LOAD_IMAGE_UNCHANGED)
    val outImage: Mat = image.clone
    //    Imgproc.bilateralFilter(image, outImage, 10, 30, 30)
    Imgproc.cvtColor(outImage, outImage, Imgproc.COLOR_RGB2GRAY)
    Imgproc.threshold(outImage, outImage, 80, 255, Imgproc.THRESH_BINARY)
    val contours = new java.util.ArrayList[MatOfPoint]
    val hierarchy = new Mat
    Imgproc.findContours(outImage.clone, contours, hierarchy, Imgproc.RETR_TREE, Imgproc.CHAIN_APPROX_SIMPLE, new Point(0, 0))
    val minEllipse = iterableAsScalaIterable(contours).filter(_.toArray.length >= 5).map(contour => Imgproc.fitEllipse(new MatOfPoint2f(contour.toArray: _*))).toList

    val drawing: Mat = image.clone
    val color: Scalar = new Scalar(0, 153, 0)
    val vdir: File = new File(outDir, imgName(imgFile) + "_vd")
    vdir.mkdirs

    if (patchStats.isDefined) patchStats.get.println("image,patch,height,width")
    for (i <- minEllipse.indices if acceptEllipse(minEllipse(i))) {
      val ellipse: RotatedRect = minEllipse(i)
      val outerPatch = patchDims(ellipse, image)
      val patch: Mat = image.submat(outerPatch.rowStart, outerPatch.rowEnd, outerPatch.colStart, outerPatch.colEnd)
      val patchName: String = vdir.getAbsolutePath + File.separator + imgName(imgFile) + "_patch" + i
      Imgcodecs.imwrite(patchName + ".png", patch)
      Imgproc.ellipse(drawing, ellipse, color, 1, 8)
      if (patchStats.isDefined) patchStats.get.println(s"${imgFile.getName},$i,${ellipse.size.height},${ellipse.size.width}")
    }

    Imgcodecs.imwrite(outDir.getAbsolutePath + File.separator + imgName(imgFile) + "_test_bf_.png", outImage)
    Imgcodecs.imwrite(outDir.getAbsolutePath + File.separator + imgName(imgFile) + "_test_bf_contours.png", drawing)
  }

  /**
    * Method deciding whether a detected ellipse will be accepted.
    * @param ellipse The candidate ellipse.
    * @return True if accepted, false otherwise.
    */
  private def acceptEllipse(ellipse: RotatedRect) = {
    val HWRatio = ellipse.size.height / ellipse.size.width

    ellipse.size.height >= MIN_VD_HEIGHT &
      ellipse.size.height <= MAX_VD_HEIGHT & ellipse.size.width >= MIN_VD_WIDTH & ellipse.size.width <= MAX_VD_WIDTH &
      HWRatio >= MIN_VD_HW_RATIO & HWRatio <= MAX_VD_HW_RATIO
  }

  /**
    * Extracts the fullpath except the file type extension.
    * @param img The file.
    * @return The fullpath except the file type extension.
    */
  private def imgName(img: File): String = img.getName.split("\\.").head

  /**
    * Computes the patch dimension so that after rotation there will be no blanks around.
    * @param ellipse The candidate ellipse to be extracted from the image.
    * @param image The image to extract the eclipse from.
    * @return The enlarged patch size as an instance of PatchSize.
    */
  private def patchDims(ellipse: RotatedRect, image: Mat): PatchSize = {
    val d = Math.ceil(Math.sqrt(2 * Math.pow(INNER_PATCH_SIZE, 2))).toInt
    val half: Int = d / 2
    val x = ellipse.center.x.round.toInt
    val y = ellipse.center.y.round.toInt

    val (colStart, colEnd) =
      if (x - half < 0) (0, 2 * half)
      else if (x + half >= image.cols) (image.cols - 1 - 2 * half, image.cols - 1)
      else (x - half, x + half)

    val (rowStart, rowEnd) =
      if (y - half < 0) (0, 2 * half)
      else if (y + half >= image.rows) (image.rows - 1 - 2 * half, image.rows - 1)
      else (y - half, y + half)

    new PatchSize(rowStart, rowEnd, colStart, colEnd)
  }
}

case class PatchSize(val rowStart: Int, val rowEnd: Int, val colStart: Int, val colEnd: Int)