package com.patres.neuralnetwork;

import com.patres.neuralnetwork.math.Vector;
import com.patres.neuralnetwork.network.Neuron;
import com.patres.neuralnetwork.network.activation.ActivationFunction;
import com.patres.neuralnetwork.network.cost.CostFunction;

import java.util.List;

import static java.util.stream.IntStream.range;

public class FullyConnectedLayer extends Layer {

    private final List<Neuron> neurons;
    private double learningRatio;
    private int numberOfInputs;


    public FullyConnectedLayer(int numberOfInputs, int numberOfNeurons, double learningRatio, ActivationFunction activation, CostFunction cost) {
        this.neurons = range(0, numberOfNeurons)
                .mapToObj(neuron -> new Neuron(numberOfInputs, activation, cost))
                .toList();
        this.numberOfInputs = numberOfInputs;
        this.learningRatio = learningRatio;
    }

    public FullyConnectedLayer(int numberOfInputs, double learningRatio, final List<Neuron> neurons) {
        this.neurons = neurons;
        this.numberOfInputs = numberOfInputs;
        this.learningRatio = learningRatio;
    }

    @Override
    public Vector forward(Vector inputs) {
        Vector scaledInput = inputs.multiply((1.0 / (256.0 * 100.0))); // TODO nie wiem dlaczego ale dziala


        double[] output = neurons.stream()
                .mapToDouble(neuron -> neuron.forward(scaledInput))
                .toArray();
        return Vector.of(output);
    }

    @Override
    public Vector backPropagation(Vector expectedOutputs) {
        List<Vector> list = neurons.stream()
                .map(neuron -> neuron.learn(expectedOutputs.getValue(neurons.indexOf(neuron)), learningRatio))
                .toList();
        Vector reduce = list.stream()
                .reduce(Vector.emptyOf(numberOfInputs), Vector::add);
        return reduce;
    }

}
