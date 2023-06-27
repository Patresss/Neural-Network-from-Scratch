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
    private double nodeValue;
    private Vector costGradientW;


    public Neuron(final int numberOfInputs, final ActivationFunction activation, final CostFunction cost) {
        this.weights = Vector.gaussianOf(numberOfInputs);
        this.bias = 0.0;
        this.activation = activation;
        this.cost = cost;
        this.costGradientW = Vector.gaussianOf(numberOfInputs);;
    }

    public Neuron(final Vector weights, final double bias, final ActivationFunction activation, final CostFunction cost) {
        this.weights = weights;
        this.bias = bias;
        this.activation = activation;
        this.cost = cost;
    }


//    public double forward(Vector inputs) {
//        lastOutputBeforeActivation = inputs.dotProduct(weights) + bias;
//        lastOutput = activation.activate(lastOutputBeforeActivation);
//        return lastOutput;
//    }


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
        return activation.derivative(lastOutputBeforeActivation); // TODO bylo lastOutput
    }

    /**
     * ∂Z
     * --
     * ∂A (z poprzedniej wwartwy)
     */
    public double weightedInputDerivativeByActivationDerivative() {
        return activation.derivative(lastOutputBeforeActivation); // TODO bylo lastOutput
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

    public double calculateNodeValue(final double expectedOutput) { // ∂C / ∂W
        return activationDerivativeByOutputBeforeActivationDerivative() // ∂A / ∂Z
                * errorDerivativeByActivationDerivative(expectedOutput);  // ∂C / ∂A
    }

    public Vector learnOutput(final double expectedOutput, final double learningRate) {
        Vector nodeValue = weights
                .multiply(activationDerivativeByOutputBeforeActivationDerivative()) // ∂A/∂Z
                .multiply(errorDerivativeByActivationDerivative(expectedOutput));  // ∂C/∂A

        // ∂C/∂W
        Vector weightRatioToChange = outputBeforeActivationDerivativeByWeightsDerivative() // ∂Z/∂W
                .multiply(activationDerivativeByOutputBeforeActivationDerivative()) // ∂A/∂Z
                .multiply(errorDerivativeByActivationDerivative(expectedOutput));  // ∂C/∂A

        weights = weights.subtract(weightRatioToChange.multiply(learningRate));
        return nodeValue;
    }

    public Vector learnHidden(final double nodeValueWeigthedPreviousLayer, final double learningRate) {

        // ∂C/∂W
        Vector weightRatioToChange =
                outputBeforeActivationDerivativeByWeightsDerivative() // ∂Z/∂W
                .multiply(nodeValueWeigthedPreviousLayer)
                .multiply(activation.derivative(lastOutputBeforeActivation));

        weights = weights.subtract(weightRatioToChange.multiply(learningRate));

        return weightRatioToChange;
    }


//    public Vector learn(final double expectedOutput, final double learningRate) {
//        nodeValue = calculateNodeValue(expectedOutput);
//
//        //        // Partial derivative of the weighted input with respect to the input
//        Vector output = weights.multiply(nodeValue);
//
//        Vector weightRatioToChange = lastInput.multiply(nodeValue * learningRate);
//        weights = weights.subtract(weightRatioToChange);
//        return output;
//    }

//    public Vector learnOutput(final double expectedOutput, final double learningRate) {
//        Vector output = weights.multiply(
//                cost.calculateCost(lastOutput, expectedOutput) *
//                activation.derivative(lastOutput));
//        // iteruj po inputach? TODO
//        nodeValue = calculateNodeValue(expectedOutput);
//
////        Vector output = weights.multiply(nodeValue);
//
//
//        Vector weightRatioToChange = lastInput.multiply(nodeValue * learningRate);
//        weights = weights.subtract(weightRatioToChange);
//        return output;
//    }

    public Vector learnOutputDlaWszystkich(final double dLdO, final double learningRate) {

        // iteracja po j
        Vector output = weights
                .multiply(dLdO * activation.derivative(lastOutput));

        Vector weightRatioToChange = lastInput
                .multiply(dLdO * activation.derivative(lastOutput))
                .multiply(learningRate);
        weights = weights.subtract(weightRatioToChange);
        return output;
    }


//
//    public double learnHidden(final double nodeValueWithWeightFromOldLayer, final double learningRate) {
//
//        Vector weightRatioToChange =     lastInput.multiply(
//                nodeValueWithWeightFromOldLayer *
//                        activation.derivative(lastOutput));
//
//        // nodeValueWithWeight -> weightedInputDerivative * oldNodeValues[oldNodeIndex]
//
//        double sumValues = nodeValueWithWeightFromOldLayer;
//        double nodeValueInner = activation.derivative(lastOutput) * sumValues;
//
////        Vector weightRatioToChange = lastInput.multiply(nodeValueInner).multiply(learningRate);
//        weights = weights.subtract(weightRatioToChange.multiply(learningRate));
//        return nodeValueInner;
//    }

    public Vector getWeights() {
        return weights;
    }

    public double getBias() {
        return bias;
    }
    public double getNodeValue() {
        return nodeValue;
    }

    public void setNodeValue(double nodeValue) {
        this.nodeValue = nodeValue;
    }

    public Vector getCostGradientW() {
        return costGradientW;
    }

    public void setCostGradientW(Vector costGradientW) {
        this.costGradientW = costGradientW;
    }
}
