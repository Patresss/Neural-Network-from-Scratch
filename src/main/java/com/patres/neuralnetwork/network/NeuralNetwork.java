package com.patres.neuralnetwork.network;

import com.patres.neuralnetwork.Layer;
import com.patres.neuralnetwork.data.data.ImageInput;
import com.patres.neuralnetwork.data.data.Input;
import com.patres.neuralnetwork.math.Vector;

import java.util.List;

public class NeuralNetwork {

    private final List<Layer> layers;

    public NeuralNetwork(List<Layer> layers) {
        this.layers = layers;
    }

    public void train(List<? extends Input> dataToTrainList) {
        dataToTrainList.forEach(this::train);
    }

//    public void train(final Input dataToTrain) {
//        ;
//
//        Vector output = getErrors(getOutput(dataToTrain), dataToTrain.getLabels());
//
//        for (int i = layers.size(); i-- > 0; ) {
//            output = layers.get(i).backPropagationNew(output);
//        }
//
//
//    }

    public void train(final Input dataToTrain) {

        Vector expectedOutputs = getErrors(getOutput(dataToTrain), dataToTrain.getLabels());

        Layer outputLayer = layers.get(layers.size() - 1);
        outputLayer.backPropagationOutputLayer(dataToTrain.getLabels());
        outputLayer.updateGradients();
        List<Neuron> neurons = outputLayer.getNeurons();
        for (int i = layers.size() - 1; i-- > 0; ) {
            Layer hiddenLayer = layers.get(i);
            hiddenLayer.backPropagationHiddenLayer(neurons);
            hiddenLayer.updateGradients();

            neurons = hiddenLayer.getNeurons();
        }

        for (int i = layers.size(); i-- > 0; ) {
            Layer layer = layers.get(i);
            layer.applyGradients();
        }

    }


    private Vector getOutput(Input dataToTrain) {
        Vector output = dataToTrain.getData().multiply((1.0 / (256.0 * 10.0))); // TODO nie wiem dlaczego ale dziala

        for (Layer layer : layers) {
            output = layer.fullyConnectedForwardPass(output);
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

    public Vector getErrors(final Vector networkOutput, final Vector expected) {
        return networkOutput.add(expected.multiply(-1));
    }

//    public Vector getErrors(final Vector networkOutput, final Vector expected) {
//        return JAVA_POINT_SQUARE.calculateCost(networkOutput, expected);
//    }


}
