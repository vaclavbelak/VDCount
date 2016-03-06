package net.belak.vdcount;

/**
 * Created by vaclav on 27.2.16.
 */

import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.io.File;
import java.io.FileFilter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

class EllipseDetectorJava {

    private File srcDir;
    private File outDir;

    public EllipseDetectorJava(File srdDir, File outDir) {
        this.srcDir = srdDir;
        this.outDir = outDir;
    }

    public void convertAll() throws IOException {
        for (File f : srcDir.listFiles(new FileFilter() {
            @Override
            public boolean accept(File pathname) {
                return pathname.getName().toLowerCase().endsWith("jpg");
            }
        })) {
            System.out.println("Converting file " + f.getName());
            detect(f);
        }
    }

    public void detect(File imgFile) throws IOException {
        Mat image = Imgcodecs.imread(imgFile.getAbsolutePath(), Imgcodecs.CV_LOAD_IMAGE_UNCHANGED);
        Mat outImage = image.clone();
        Imgproc.bilateralFilter(image, outImage, 10, 30, 30);
        Imgproc.cvtColor(outImage, outImage, Imgproc.COLOR_RGB2GRAY);
        Imgproc.threshold(outImage, outImage, 80, 255, Imgproc.THRESH_BINARY);
        List<MatOfPoint> contours = new ArrayList<MatOfPoint>();
        Mat hierarchy = new Mat();
        Imgproc.findContours(outImage.clone(), contours, hierarchy, Imgproc.RETR_TREE, Imgproc.CHAIN_APPROX_SIMPLE, new Point(0, 0));

        List<RotatedRect> minRect = new ArrayList<>(contours.size());
        List<RotatedRect> minEllipse = new ArrayList<>(contours.size());

        for (MatOfPoint contour : contours) {
            minRect.add(Imgproc.minAreaRect(new MatOfPoint2f(contour.toArray())));
            if (contour.toArray().length >= 5) {
                minEllipse.add(Imgproc.fitEllipse(new MatOfPoint2f(contour.toArray())));
            }
        }

        Mat drawing = image.clone();

        Scalar color = new Scalar(0, 153, 0);
        File vdir = new File(outDir, imgName(imgFile) + "_vd");
        vdir.mkdirs();
        int innerBoxSize = 12;
//        System.out.println(image.rows() + ", " + image.cols());
        for (int i = 0; i < minEllipse.size(); i++) {
            RotatedRect ellipse = minEllipse.get(i);

            int[] outerBox = boxDims(innerBoxSize, ellipse, image);

            Mat box = image.submat(outerBox[0], outerBox[1], outerBox[2], outerBox[3]);
            String boxName = vdir.getAbsolutePath() + File.separator + imgName(imgFile) + "_box" + i;
            varyExamples(box, boxName, innerBoxSize, 90);
            Imgcodecs.imwrite(boxName + ".png", box.submat(box.width() / 2 - innerBoxSize, box.width() / 2 + innerBoxSize,
                    box.height() / 2 - innerBoxSize, box.height() / 2 + innerBoxSize));

            Imgproc.ellipse(drawing, ellipse, color, 2, 8);
        }

        Imgcodecs.imwrite(outDir.getAbsolutePath() + File.separator + imgName(imgFile) + "_test_bf_.png", outImage);
        Imgcodecs.imwrite(outDir.getAbsolutePath() + File.separator + imgName(imgFile) + "_test_bf_contours.png", drawing);
    }

    private void varyExamples(Mat image, String name, int innerBoxSize, int degIncrement) {
        for (int deg = degIncrement; deg < 360; deg += degIncrement) {
            Mat rotatedImg = rotate(image, deg, innerBoxSize);
            Imgcodecs.imwrite(name + "_rot" + deg + ".png", rotatedImg);
        }
    }

    private Mat rotate(Mat image, int degrees, int innerBoxSize) {
        Point center = new Point(image.width() / 2, image.height() / 2);
        Mat rotationMat = Imgproc.getRotationMatrix2D(center, degrees, 1);
        Mat result = new Mat();
        Imgproc.warpAffine(image, result, rotationMat, new Size(image.width(), image.height()));
        return(result.submat((int)center.y - innerBoxSize, (int)center.y + innerBoxSize,
                (int)center.x - innerBoxSize, (int)center.x + innerBoxSize));
    }

    private String imgName(File img) {
        return img.getName().split("\\.")[0];
    }

    private int[] boxDims(int innerBoxSize, RotatedRect ellipse, Mat image) {
        int d = (int) Math.ceil(Math.sqrt(2 * Math.pow(innerBoxSize, 2)));
        int x = (int) Math.round(ellipse.center.x);
        int y = (int) Math.round(ellipse.center.y);
        int x1 = -1, x2 = -1, y1 = -1, y2 = -1;

        if (x - d < 0) {
            x1 = 0;
            x2 = 2 * d;
        } else if (x + d >= image.cols()) {
            x1 = image.cols() - 1 - 2 * d;
            x2 = image.cols() - 1;
        } else {
            x1 = x - d;
            x2 = x + d;
        }

        if (y - d < 0) {
            y1 = 0;
            y2 = 2 * d;
        } else if (y + d >= image.rows()) {
            y1 = image.rows() - 1 - 2 * d;
            y2 = image.rows() - 1;
        } else {
            y1 = y - d;
            y2 = y + d;
        }

        return new int[] {y1, y2, x1, x2};
    }
}