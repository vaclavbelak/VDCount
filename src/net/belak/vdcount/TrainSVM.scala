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
  val hd = new HOGDescriptor("/Users/belak/Projects/VDCount-scala/hog-config.xml")
  val COLNUM = 144

  println("Loading data...")
  val data = prepareData(new File(args(0)))
  println(data)
  val svm = SVM.create()
//  svm.setC(1)
//  svm.setGamma(0.5)
  svm.setKernel(SVM.RBF)
//  svm.setType(SVM.C_SVC)
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
//  svm.save("/Users/belak/Projects/VDCount-scala/svm-model.xml")

  def prepareData(dir: File) = {
    val rnd = new Random(25)
    val posFilesTrain = new File(dir, "train/positive/rotated").listFiles().filter(_.getName.endsWith(".png")).toList.sortBy(_.getName).
      map(f => (1, f))
    val posFilesTest = new File(dir, "test/positive/rotated").listFiles().filter(_.getName.endsWith(".png")).toList.sortBy(_.getName).
      map(f => (1, f))
    val negFilesTrain = new File(dir, "train/negative/rotated").listFiles().filter(_.getName.endsWith(".png")).toList.sortBy(_.getName).
      map(f => (-1, f))
    val negFilesTest = new File(dir, "test/negative/rotated").listFiles().filter(_.getName.endsWith(".png")).toList.sortBy(_.getName).
      map(f => (-1, f))
    val trainFiles = rnd.shuffle(posFilesTrain ::: negFilesTrain ::: Nil)
    val testFiles = rnd.shuffle(posFilesTest ::: negFilesTest ::: Nil)

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