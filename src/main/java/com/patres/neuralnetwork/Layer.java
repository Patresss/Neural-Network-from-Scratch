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

    public abstract void backPropagationOutputLayer(Vector expectedOutputs);


    public abstract void applyGradients();

    public abstract void updateGradients();

    public abstract void backPropagationHiddenLayer(List<Neuron> oldNodeValues);

    public abstract Vector backPropagationOutput(Vector expectedOutputs);

    public abstract Vector backPropagationHidden(Vector expectedOutputs);

    public abstract Vector backPropagationO(Vector expectedOutputs);

    public abstract Vector backPropagationDlaWszystkich(Vector expectedOutputs);

    public abstract Vector backPropagationNew(Vector expectedOutputs);

    public abstract Vector fullyConnectedForwardPass(Vector output);

    public abstract List<Neuron> getNeurons();
}
