package com.patres.neuralnetwork.network;

import com.patres.neuralnetwork.FullyConnectedLayer;
import com.patres.neuralnetwork.data.data.Input;
import com.patres.neuralnetwork.math.Vector;
import com.patres.neuralnetwork.network.activation.ActivationFunction;
import com.patres.neuralnetwork.network.cost.CostFunction;
import org.junit.jupiter.api.Test;

import java.util.List;

class NeuralNetworkTest {

    //https://www.javatpoint.com/pytorch-backpropagation-process-in-deep-neural-network
//    Input values
//    X1=0.05
//    X2=0.10
//
//    Initial weight
//    W1=0.15     w5=0.40
//    W2=0.20     w6=0.45
//    W3=0.25     w7=0.50
//    W4=0.30     w8=0.55
//
//    Bias Values
//    b1=0.35     b2=0.60
//
//    Target Values
//    T1=0.01
//    T2=0.99

    @Test
    public void testt() {
        System.out.println("TTTTTTTTTTTTT");

        final NeuralNetwork neuralNetwork = new NeuralNetwork(List.of(
                new FullyConnectedLayer(
                        2,
                        0.5,
                        List.of(new Neuron(Vector.of(0.15, 0.20),
                                0.35,
                                ActivationFunction.SIGMOID,
                                CostFunction.JAVA_POINT_SQUARE),
                                new Neuron(Vector.of(0.25, 0.30),
                                        0.35,
                                        ActivationFunction.SIGMOID,
                                        CostFunction.JAVA_POINT_SQUARE))),
                new FullyConnectedLayer(
                        2,
                        0.5,
                        List.of(new Neuron(Vector.of(0.40, 0.45),
                                        0.60,
                                        ActivationFunction.SIGMOID,
                                        CostFunction.JAVA_POINT_SQUARE),
                                new Neuron(Vector.of(0.50, 0.55),
                                        0.60,
                                        ActivationFunction.SIGMOID,
                                        CostFunction.JAVA_POINT_SQUARE)))
        ));

        Input input = new Input(Vector.of(0.05, 0.10), Vector.of(0.01, 0.99));
        neuralNetwork.train(input);
        System.out.println(neuralNetwork);
    }

}