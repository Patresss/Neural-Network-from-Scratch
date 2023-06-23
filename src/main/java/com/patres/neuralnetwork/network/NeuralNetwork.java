package com.patres.neuralnetwork.network;

import com.patres.neuralnetwork.Layer;
import com.patres.neuralnetwork.data.data.ImageInput;
import com.patres.neuralnetwork.data.data.Input;
import com.patres.neuralnetwork.math.Vector;

import java.util.List;

import static com.patres.neuralnetwork.network.cost.CostFunction.JAVA_POINT_SQUARE;

public class NeuralNetwork {

    private final List<Layer> layers;

    public NeuralNetwork(List<Layer> layers) {
        this.layers = layers;
    }

    public void train(List<? extends Input> dataToTrainList) {
        dataToTrainList.forEach(this::train);
    }

    public void train(final Input dataToTrain) {
        getOutput(dataToTrain);

//        Vector expectedOutputs = getErrors(output, dataToTrain.getLabels());
        Vector expectedOutputs = dataToTrain.getLabels();
        for (int i = layers.size(); i-- > 0; ) {
            expectedOutputs = layers.get(i).backPropagation(expectedOutputs);
        }
    }

    private Vector getOutput(Input dataToTrain) {
        Vector output = dataToTrain.getData();
        for (Layer layer : layers) {
            output = layer.forward(output);
        }
        return output;
    }

    public double calculateAccuracy(List<? extends Input> inputs) {
        int correct = 0;
        for (Input input : inputs) {
            int guess = guess(input);
            if (input.getLabel().equals(guess)) {
                correct++;
            }
        }
        return ((double) correct / inputs.size());
    }

    public int guess(Input input) {
        Vector output = getOutput(input);
        return output.getMaxIndex();
    }

    public Vector test(ImageInput imageInput) {
        System.out.println("-----------------------------------------------");
        System.out.println("Expected: " + imageInput.getLabelAsString());
        Vector output = getOutput(imageInput);
        System.out.println("Result:" + output.getMaxIndex());
        System.out.println(output);
        System.out.println("-----------------------------------------------");
        return output;
    }

//    public Vector getErrors(final Vector networkOutput, final int correctAnswer) {
//        final Vector expected = Vector.emptyOf(networkOutput.getSize());
//        expected.setValue(correctAnswer, 1);
//        return networkOutput.add(expected.multiply(-1));
//    }

    public Vector getErrors(final Vector networkOutput, final Vector expected) {
        return JAVA_POINT_SQUARE.calculateCost(networkOutput, expected);
    }


}
