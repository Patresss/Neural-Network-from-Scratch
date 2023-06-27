package com.patres.neuralnetwork;

import com.patres.neuralnetwork.math.Vector;
import com.patres.neuralnetwork.network.Neuron;

import java.util.List;

public abstract class Layer {

    public abstract Vector forward(Vector inputs);

    public abstract void backPropagationOutputLayer(Vector expectedOutputs);

    public abstract void applyGradients();

    public abstract void updateGradients();

    public abstract void backPropagationHiddenLayer(List<Neuron> oldNodeValues);

    public abstract List<Neuron> getNeurons();
}
