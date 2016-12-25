package net.belak.vdcount

import java.io.File

import org.opencv.core._
import org.opencv.imgcodecs.Imgcodecs
import org.opencv.ml.{SVM, Ml}
import org.opencv.objdetect.HOGDescriptor

import scala.util.Random

/**
  * Created by Václav on 3/25/2016.
  */
object TrainSVM extends App with Constants {

  System.loadLibrary(Core.NATIVE_LIBRARY_NAME)
  val hd = new HOGDescriptor(new Size(24, 24), new Size(16, 16), new Size(8, 8), new Size(8, 8), 9)
  val COLNUM = 144

  println("Loading data...")
  val data = prepareData(new File(args(0)), 0.8)
  println(data)
  val svm = SVM.create()
  println("Data loaded, training SVM")
  svm.train(data.trainX, Ml.ROW_SAMPLE, data.trainY)
  println("Training finished")
  val predLabels = Array.ofDim[Float](data.testX.rows())
  var correctCount = 0d
  for (i <- 0 until data.testX.rows()) {
    if (svm.predict(data.testX.row(i)).toInt == data.testY.get(i, 0)(0).toInt) {
      correctCount += 1
    }
  }
  print(s"Error rate ${correctCount / data.testY.rows()}")
  svm.save("/Users/belak/Projects/VDCount/svm-opencv.xml")

  def prepareData(dir: File, p: Double) = {
    assert(p <= 1 & p > 0)
    val rnd = new Random(25)
    val posFiles = new File(dir, "positive/rotated").listFiles().filter(_.getName.endsWith(".png")).toList.sortBy(_.getName).
      map(f => (1, f))
    val negFiles = new File(dir, "negative/rotated").listFiles().filter(_.getName.endsWith(".png")).toList.sortBy(_.getName).
      map(f => (-1, f))
    val allFiles = rnd.shuffle(posFiles ::: negFiles ::: Nil)
    val pivot = (allFiles.size * p).toInt
    val trainFiles = allFiles.take(pivot)
    val trailLabels = new Mat()
    val testFiles = allFiles.drop(pivot)

    new Data(trainX = loadDataMatrix(trainFiles.map(_._2)), trainY = labelsMatrix(trainFiles),
      testX = loadDataMatrix(testFiles.map(_._2)), testY = labelsMatrix(testFiles))
  }

  def labelsMatrix(fileList: List[(Int, File)]) = {
    val m = new Mat(fileList.size, 1, CvType.CV_32S)
    for (row <- fileList.indices) {
      m.put(row, 0, fileList(row)._1)
    }
    m
  }

  def loadDataMatrix(imgFiles: List[File]) = {
    val mat = new Mat(imgFiles.size, COLNUM, CvType.CV_32FC1)
    for (imgFileIdx <- imgFiles.indices) {
      val features = new MatOfFloat
      hd.compute(Imgcodecs.imread(imgFiles(imgFileIdx).getAbsolutePath, Imgcodecs.CV_LOAD_IMAGE_GRAYSCALE), features)
      val featuresArray = features.toArray
      assert(COLNUM == featuresArray.length)
      for (col <- 0 until COLNUM) {
        mat.put(imgFileIdx, col, featuresArray(col))
      }
    }
    mat
  }
}

case class Data(trainX: Mat, trainY: Mat, testX: Mat, testY: Mat)