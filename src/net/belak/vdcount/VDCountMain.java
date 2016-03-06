package net.belak.vdcount;

import org.opencv.core.*;

import java.io.File;
import java.io.IOException;

/**
 * Created by vaclav on 27.2.16.
 */
public class VDCountMain {
    public static void main(String[] args) {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        Core.setNumThreads(2);
        EllipseDetectorJava eDetector = new EllipseDetectorJava(new File("/home/vaclav/IdeaProjects/VDCount/resources/VDSource"),
                new File("/home/vaclav/IdeaProjects/VDCount/resources/VDTest"));
        try {
        eDetector.convertAll();
//        eDetector.detect(new File("/home/vaclav/IdeaProjects/VDCount/resources/VDSource/IMG_1448.JPG"));
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
