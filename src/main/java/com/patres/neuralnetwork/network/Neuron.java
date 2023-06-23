package com.patres.neuralnetwork.network;

import com.patres.neuralnetwork.math.Vector;
import com.patres.neuralnetwork.network.activation.ActivationFunction;
import com.patres.neuralnetwork.network.cost.CostFunction;

public class Neuron {

    private Vector weights;
    private final double bias;
    private final ActivationFunction activation;
    private final CostFunction cost;

    private Vector lastInput; // X
    private double lastOutputBeforeActivation; // Z
    private double lastOutput;


    public Neuron(final int numberOfInputs, final ActivationFunction activation, final CostFunction cost) {
        this.weights = Vector.gaussianOf(numberOfInputs);
        this.bias = 0.0;
        this.activation = activation;
        this.cost = cost;
    }

    public Neuron(final Vector weights, final double bias, final ActivationFunction activation, final CostFunction cost) {
        this.weights = weights;
        this.bias = bias;
        this.activation = activation;
        this.cost = cost;
    }


    public double forward(Vector inputs) {
        lastInput = inputs;
        lastOutputBeforeActivation = inputs.dotProduct(weights) + bias;
        lastOutput = activation.activate(lastOutputBeforeActivation);
        return lastOutput;
    }


    /**
     * ∂C
     * --
     * ∂A
     */
    public double errorDerivativeByActivationDerivative(final double expectedOutput) { // dC / dA
        return cost.calculateDerivativeCost(lastOutput, expectedOutput);
    }

    /**
     * ∂A
     * --
     * ∂Z
     */
    public double activationDerivativeByOutputBeforeActivationDerivative() {
        return activation.derivative(lastOutput);
    }

    /**
     * ∂z
     * --
     * ∂W
     */
    public Vector outputBeforeActivationDerivativeByWeightsDerivative() {
        return lastInput;
    }

    /**
     * ∂z
     * --
     * ∂B
     */
    public double outputBeforeActivationDerivativeByBiasDerivative() {
        return 1.0;
    }
    public Vector outputBeforeActivationDerivativeByLastWeightsDerivative() { // dZ / dX
        return weights;
    }

    /**
     * ∂C   ∂z   ∂A   ∂C
     * -- = -- * -- * --
     * ∂W   ∂W   ∂Z   ∂A
     */
    public Vector errorDerivativeByWeightsDerivative(final double expectedOutput) { // ∂C/∂W
        return outputBeforeActivationDerivativeByWeightsDerivative() // ∂Z/∂W
                .multiply(activationDerivativeByOutputBeforeActivationDerivative()) // ∂A/∂Z
                .multiply(errorDerivativeByActivationDerivative(expectedOutput));  // ∂C/∂A  // TODO w hiden tylko expectedOutput
    }

//    public Vector errorDerivativeByLastWeightsDerivative(final double expectedOutput) { // dC / dW
//        return activationDerivativeByOutputBeforeActivationDerivative() // dA / dZ
//                .multiply(errorDerivativeByActivationDerivative(expectedOutput));  // dC / dA
//    }

    public Vector errorDerivativeByLastWeightsDerivative(final double expectedOutput) { // dC / dW
        return outputBeforeActivationDerivativeByLastWeightsDerivative() // dZ / dX
                .multiply(activationDerivativeByOutputBeforeActivationDerivative()) // dA / dZ
                .multiply(errorDerivativeByActivationDerivative(expectedOutput));  // dC / dA
    }

    public Vector learn(final double expectedOutput, final double learningRate) {
        Vector dLdX = errorDerivativeByLastWeightsDerivative(expectedOutput);
        Vector weightRatioToChange = errorDerivativeByWeightsDerivative(expectedOutput).multiply(learningRate);
        weights = weights.subtract(weightRatioToChange);
        return dLdX;
    }

}
