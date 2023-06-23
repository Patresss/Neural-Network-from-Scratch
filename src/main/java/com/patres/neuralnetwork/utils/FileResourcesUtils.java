package com.patres.neuralnetwork.utils;

import com.patres.neuralnetwork.exception.NeuralNetworkDataException;

import java.io.File;
import java.net.URL;

public class FileResourcesUtils {

    private FileResourcesUtils() {
    }

    public static File getFileFromResource(final String fileName) {
        try {
            final ClassLoader classLoader = FileResourcesUtils.class.getClassLoader();
            final URL resource = classLoader.getResource(fileName);
            if (resource == null) {
                throw new IllegalArgumentException("File not found! " + fileName);
            } else {
                return new File(resource.toURI());
            }
        } catch (Exception e) {
            throw new NeuralNetworkDataException("Cannot load file: " + fileName, e);
        }

    }
}
