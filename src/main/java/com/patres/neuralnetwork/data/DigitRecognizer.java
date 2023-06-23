package com.patres.neuralnetwork.data;

import com.patres.neuralnetwork.FullyConnectedLayer;
import com.patres.neuralnetwork.data.data.ImageInput;
import com.patres.neuralnetwork.data.data.ImageDataReader;
import com.patres.neuralnetwork.network.NeuralNetwork;
import com.patres.neuralnetwork.network.activation.ActivationFunction;
import com.patres.neuralnetwork.network.cost.CostFunction;

import java.io.File;
import java.util.List;

import static java.util.Collections.shuffle;

public class DigitRecognizer {

    private final File testDataFile;
    private final File trainDataFile;
    private final int rows;
    private final int columns;

    public DigitRecognizer(File trainDataFile, File testDataFile, int rows, int columns) {
        this.testDataFile = testDataFile;
        this.trainDataFile = trainDataFile;
        this.rows = rows;
        this.columns = columns;
    }

    public void run() {
        System.out.println("Starting data loading...");
        ImageDataReader testImageDataReader = new ImageDataReader(testDataFile, rows, columns);
        ImageDataReader trainImageDataReader = new ImageDataReader(trainDataFile, rows, columns);
        List<ImageInput> imagesTest = testImageDataReader.readData();
        List<ImageInput> imagesTrain = trainImageDataReader.readData();

//        System.out.println(imagesTest.get(0).toString());

        NeuralNetwork neuralNetwork = new NeuralNetwork(List.of(
                new FullyConnectedLayer(28 * 28, 8, 0.1, ActivationFunction.RE_LU, CostFunction.SQUARE),
                new FullyConnectedLayer(8, 10, 0.1, ActivationFunction.RE_LU, CostFunction.SQUARE)
        ));

        int epochs = 3;
        for(int i = 1; i < epochs + 1; i++){
            shuffle(imagesTrain);
            neuralNetwork.train(imagesTrain);
            double rate = neuralNetwork.calculateAccuracy(imagesTest);
            System.out.println("Success rate after round " + i + ": " + rate);
        }





    }
}
