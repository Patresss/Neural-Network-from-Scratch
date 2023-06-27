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
    private int numberOfOutputs;

    private Vector lastInput;
    private Vector lastOutput;
    private Vector lastOutputBeforeActivation;
    private Vector costGradientW;

    private final ActivationFunction activation;
    private final CostFunction cost;


    public FullyConnectedLayer(int numberOfInputs, int numberOfOutputs, double learningRatio, ActivationFunction activation, CostFunction cost) {
        this.neurons = range(0, numberOfOutputs)
                .mapToObj(neuron -> new Neuron(numberOfInputs, activation, cost))
                .toList();
        this.numberOfInputs = numberOfInputs;
        this.numberOfOutputs = numberOfOutputs;
        this.learningRatio = learningRatio;
        this.activation = activation;
        this.cost = cost;
        this.costGradientW = Vector.emptyOf(numberOfOutputs);
    }

    public FullyConnectedLayer(int numberOfInputs, double learningRatio, final List<Neuron> neurons, ActivationFunction activation, CostFunction cost) {
        this.neurons = neurons;
        this.numberOfInputs = numberOfInputs;
        this.learningRatio = learningRatio;
        this.activation = ActivationFunction.RE_LU;
        this.cost = CostFunction.SQUARE;
    }

    @Override
    public Vector forward(Vector inputs) {
        lastInput = inputs;
        lastOutputBeforeActivation = Vector.of(
                neurons.stream()
                        .mapToDouble(neuron -> inputs.dotProduct(neuron.getWeights()) + neuron.getBias())
                        .toArray()
        );
        lastOutput = activation.activate(lastOutputBeforeActivation);
        return lastOutput;
    }

    /**
     * ∂A
     * --
     * ∂Z
     */
    public Vector activationDerivativeByOutputBeforeActivationDerivative() {
        return activation.derivative(lastOutput);
    }

//    /**
//     * ∂C
//     * --
//     * ∂A
//     */
//    public double errorDerivativeByActivationDerivative(final double expectedOutput) { // dC / dA
//        return cost.calculateDerivativeCost(lastOutput, expectedOutput);
//    }

    /**
     * ∂C
     * --
     * ∂A
     */
    public Vector errorDerivativeByActivationDerivative(final Vector lastOutput, final Vector expectedOutput) {
        return cost.calculateDerivativeCost(lastOutput, expectedOutput);
    }

    /**
     * ∂z
     * --
     * ∂W
     */
    public Vector outputBeforeActivationDerivativeByWeightsDerivative() {
        return lastInput;
    }

    @Override
    public void backPropagationOutputLayer(Vector expectedOutputs) {
        neurons.forEach(neuron -> {
                    int neuronIndex = neurons.indexOf(neuron);
                    double costDerivative = cost.calculateDerivativeCost(lastOutput.getValue(neuronIndex), expectedOutputs.getValue(neuronIndex));
                    double activationDerivative = activation.derivative(lastOutputBeforeActivation.getValue(neuronIndex));
                    double nodeValue = costDerivative * activationDerivative;
                    neuron.setNodeValue(nodeValue);
                });
    }

    @Override
    public void backPropagationHiddenLayer(List<Neuron> oldNodeValues) {
        for (int newNodeIndex = 0; newNodeIndex < numberOfOutputs; newNodeIndex++) {
            double newNodeValue = 0;
            for (Neuron oldNodeValue : oldNodeValues) {
                // Partial derivative of the weighted input with respect to the input
                double weightedInputDerivative = oldNodeValue.getWeights().getValue(newNodeIndex);
                newNodeValue += (weightedInputDerivative * oldNodeValue.getNodeValue());
            }

            newNodeValue *= activation.derivative(lastOutputBeforeActivation.getValue(newNodeIndex));
            neurons.get(newNodeIndex).setNodeValue(newNodeValue);
        }
    }

    @Override
    public void updateGradients() {
        for (Neuron neuron : neurons) {
            double nodeValue = neuron.getNodeValue();
            for (int nodeIn = 0; nodeIn < numberOfInputs; nodeIn++) {
                double derivativeCostWrtWeight = lastInput.getValue(nodeIn) * nodeValue;

                // The costGradientW array stores these partial derivatives for each weight.
                // Note: the derivative is being added to the array here because ultimately we want
                // to calculate the average gradient across all the data in the training batch
                neuron.getCostGradientW().setValue(nodeIn, derivativeCostWrtWeight + neuron.getCostGradientW().getValue(nodeIn));
            }
        }
    }

    @Override
    public void applyGradients() {
        for (Neuron neuron : neurons) {
            for (int i = 0; i < neuron.getWeights().getSize(); i++) {
                double currentWeight = neuron.getWeights().getValue(i);
                neuron.getWeights().setValue(i, currentWeight - (neuron.getCostGradientW().getValue(i) * (learningRatio)));
                neuron.getCostGradientW().setValue(i, 0.0);
            }
        }
    }

    @Override
    public List<Neuron> getNeurons() {
        return neurons;
    }
}
