package com.patres.neuralnetwork;

import com.patres.neuralnetwork.math.Vector;
import com.patres.neuralnetwork.network.Neuron;
import com.patres.neuralnetwork.network.activation.ActivationFunction;
import com.patres.neuralnetwork.network.cost.CostFunction;

import java.util.List;

import static java.util.stream.IntStream.range;

public class FullyConnectedLayer extends Layer {

    private final List<Neuron> neurons;
    private final List<Neuron> neuronsCnn;
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
        this.neuronsCnn = range(0, numberOfInputs)
                .mapToObj(neuron -> new Neuron(numberOfOutputs, activation, cost))
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
        this.neuronsCnn = neurons;
        this.numberOfInputs = numberOfInputs;
        this.learningRatio = learningRatio;
        this.activation = ActivationFunction.RE_LU;
        this.cost = CostFunction.SQUARE;

    }

    @Override
    public Vector forward(Vector inputs) {
        double[] output = neurons.stream()
                .mapToDouble(neuron -> neuron.forward(inputs))
                .toArray();
        return Vector.of(output);
    }

//    public Vector fullyConnectedForwardPass(Vector inputs) {
//
//        lastInput = inputs;
//
//        Vector z = Vector.emptyOf(numberOfOutputs);
//        Vector out = Vector.emptyOf(numberOfOutputs);
//
//        for (int j = 0; j < numberOfOutputs; j++) {
//            for (int i = 0; i < numberOfInputs; i++) {
////                z[j] += input[i]*_weights[i][j];
//
//                z.setValue(j, z.getValue(j) + (inputs.getValue(i) * neurons.get(j).getWeights().getValue(j)));
//            }
//        }
//
//        lastOutputBeforeActivation = z;
//
//        for (int i = 0; i < numberOfInputs; i++) {
//            for (int j = 0; j < numberOfOutputs; j++) {
//                out.setValue(j, activation.activate(z.getValue(j)));
//            }
//        }
//        lastOutput = out;
//        return out;
//    }


    public Vector fullyConnectedForwardPass(Vector inputs) {
        lastInput = inputs;

        lastOutputBeforeActivation = Vector.of(
                neurons.stream()
                        .mapToDouble(neuron -> inputs.dotProduct(neuron.getWeights()) + neuron.getBias())
                        .toArray()
        );
        lastOutput = activation.activate(lastOutputBeforeActivation);
        return lastOutput;
    }
//    public Vector forward(Vector inputs) {
//        Vector scaledInput = inputs.multiply((1.0 / (256.0 * 100.0))); // TODO nie wiem dlaczego ale dziala
//        lastInput = scaledInput;
//
//        lastOutputBeforeActivation = Vector.of(
//                neuronsCnn.stream()
//                        .mapToDouble(neuron -> inputs.dotProduct(neuron.getWeights()) + neuron.getBias())
//                        .toArray()
//        );
//        lastOutput = activation.activate(lastOutputBeforeActivation);
//        return lastOutput;
//    }

//    public void calculateOutputLayerNodeValues(Vector expectedOutputs) {
//         errorDerivativeByActivationDerivative(lastOutput, expectedOutputs) // ∂C / ∂A
//                .multiply(activationDerivativeByOutputBeforeActivationDerivative()); // ∂A / ∂Z
//        neurons.stream()
//                .forEach(neuron -> {
//                     double nodeValue = errorDerivativeByActivationDerivative(lastOutput, expectedOutputs) // ∂C / ∂A
//                             .multiply(activationDerivativeByOutputBeforeActivationDerivative()); // ∂A / ∂Z
//                    neuron.setNodeValue(nodeValue);
//                });
//
//        for (int i = 0; i < layerLearnData.nodeValues.Length; i++)
//        {
//            // Evaluate partial derivatives for current node: cost/activation & activation/weightedInput
//            double costDerivative = cost.CostDerivative(layerLearnData.activations[i], expectedOutputs[i]);
//            double activationDerivative = activation.Derivative(layerLearnData.weightedInputs, i);
//            layerLearnData.nodeValues[i] = costDerivative * activationDerivative;
//        }
//    }


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
    public Vector backPropagation(Vector expectedOutputs) {
        List<Vector> list = neurons.stream()
                .map(neuron -> {
                    return neuron.learnHidden(expectedOutputs.getValue(neurons.indexOf(neuron)), learningRatio);
                })
                .toList();
        Vector reduce = list.stream()
                .reduce(Vector.emptyOf(numberOfInputs), Vector::add);
        return reduce;
    }

    @Override
    public void backPropagationOutputLayer(Vector expectedOutputs) {
        neurons.stream()
                .forEach(neuron -> {
                    int neuronIndex = neurons.indexOf(neuron);
                    double costDerivative = cost.calculateDerivativeCost(lastOutput.getValue(neuronIndex), expectedOutputs.getValue(neuronIndex));
//                    double activationDerivative = lastOutputBeforeActivation.getValue(neuronIndex) * (1.0 - lastOutputBeforeActivation.getValue(neuronIndex));
                    double activationDerivative = activation.derivative(lastOutputBeforeActivation.getValue(neuronIndex));
                    double nodeValue = costDerivative * activationDerivative;
                    neuron.setNodeValue(nodeValue);
                });
    }

    @Override
    public void backPropagationHiddenLayer(List<Neuron> oldNodeValues) {
        for (int newNodeIndex = 0; newNodeIndex < numberOfOutputs; newNodeIndex++) {
            double newNodeValue = 0;
            for (int oldNodeIndex = 0; oldNodeIndex < oldNodeValues.size(); oldNodeIndex++) {
                // Partial derivative of the weighted input with respect to the input
                double weightedInputDerivative = oldNodeValues.get(oldNodeIndex).getWeights().getValue(newNodeIndex);
                newNodeValue += (weightedInputDerivative * oldNodeValues.get(oldNodeIndex).getNodeValue());
            }

//            newNodeValue *= (lastOutputBeforeActivation.getValue(newNodeIndex) * (1.0 - lastOutputBeforeActivation.getValue(newNodeIndex)));
            newNodeValue *= activation.derivative(lastOutputBeforeActivation.getValue(newNodeIndex));
            neurons.get(newNodeIndex).setNodeValue(newNodeValue);
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
    public void updateGradients() {
        for (Neuron neuron : neurons) {
            double nodeValue = neuron.getNodeValue();
            for (int nodeIn = 0; nodeIn < numberOfInputs; nodeIn++) {
                double derivativeCostWrtWeight = lastInput.getValue(nodeIn) * nodeValue;
                neuron.getCostGradientW().setValue(nodeIn, derivativeCostWrtWeight + neuron.getCostGradientW().getValue(nodeIn));
            }
        }
    }



    @Override
    public Vector backPropagationOutput(Vector expectedOutputs) {
        List<Vector> list = neurons.stream()
                .map(neuron -> {
                    return neuron.learnOutput(expectedOutputs.getValue(neurons.indexOf(neuron)), learningRatio);
                })
                .toList();
        Vector reduce = list.stream()
                .reduce(Vector.emptyOf(numberOfInputs), Vector::add);
        return reduce;
    }

    @Override
    public Vector backPropagationHidden(Vector nodeValueWithWeightFromOldLayer) {
        List<Vector> list = neurons.stream()
                .map(neuron -> {
                    return neuron.learnHidden(nodeValueWithWeightFromOldLayer.getValue(neurons.indexOf(neuron)), learningRatio);
                })
                .toList();
        Vector reduce = list.stream()
                .reduce(Vector.emptyOf(numberOfInputs), Vector::add);
        return reduce;
    }


    public Vector backPropagationONew(Vector expectedOutputs) {
        List<Vector> list = neurons.stream()
                .map(neuron -> {
                    return neuron.learnOutput(expectedOutputs.getValue(neurons.indexOf(neuron)), learningRatio);
                })
                .toList();
        Vector reduce = list.stream()
                .reduce(Vector.emptyOf(numberOfInputs), Vector::add);
        return reduce;
    }


    @Override
    public Vector backPropagationO(Vector expectedOutputs) {
        List<Vector> list = neurons.stream()
                .map(neuron -> {
                    return neuron.learnOutput(expectedOutputs.getValue(neurons.indexOf(neuron)), learningRatio);
                })
                .toList();
        Vector reduce = list.stream()
                .reduce(Vector.emptyOf(numberOfInputs), Vector::add);
        return reduce;
    }

    @Override
    public Vector backPropagationDlaWszystkich(Vector expectedOutputs) {
        List<Vector> list = neurons.stream()
                .map(neuron -> {
                    return neuron.learnOutputDlaWszystkich(expectedOutputs.getValue(neurons.indexOf(neuron)), learningRatio);
                })
                .toList();
        Vector reduce = list.stream()
                .reduce(Vector.emptyOf(numberOfInputs), Vector::add);
        return reduce;
    }

    @Override
    public Vector backPropagationNew(Vector dLdO) {
        Vector dLdX = Vector.emptyOf(lastInput.getSize());
        double dOdz;
        double dzdw;
        double dLdw;
        double dzdx;

        for (int k = 0; k < numberOfInputs; k++) {
            double dLdX_sum = 0;
            for (int j = 0; j < numberOfOutputs; j++) {
                dOdz = activation.derivative(lastOutputBeforeActivation.getValue(j));
                dzdw = lastInput.getValue(k);
                dzdx = neuronsCnn.get(k).getWeights().getValue(j); // tu mam błąd 0 biore wagi z innych indekxow
                dLdw = dLdO.getValue(j) * dOdz * dzdw;

                neuronsCnn.get(k).getWeights().setValue(j, neuronsCnn.get(k).getWeights().getValue(j) - (dLdw * learningRatio));
                dLdX_sum += dLdO.getValue(j) * dOdz * dzdx;
            }

            dLdX.setValue(k, dLdX_sum);
        }
        return dLdX;
    }


    @Override
    public List<Neuron> getNeurons() {
        return neurons;
    }
}
