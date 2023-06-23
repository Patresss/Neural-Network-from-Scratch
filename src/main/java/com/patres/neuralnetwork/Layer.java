package com.patres.neuralnetwork;

import com.patres.neuralnetwork.math.Matrix;
import com.patres.neuralnetwork.math.Vector;
import com.patres.neuralnetwork.network.Neuron;
import com.patres.neuralnetwork.network.activation.Activation;

import java.util.List;

import static java.util.stream.IntStream.range;

public abstract class Layer {

    public abstract Vector forward(Vector inputs);

    public abstract Vector backPropagation(Vector expectedOutputs);
}
