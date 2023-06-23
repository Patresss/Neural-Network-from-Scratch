package com.patres.neuralnetwork.network.activation;

import com.patres.neuralnetwork.math.Matrix;
import com.patres.neuralnetwork.math.Vector;

import java.util.stream.IntStream;

import static java.util.stream.IntStream.range;

public abstract class Activation {

    public Vector forward(Vector inputs) {
        Vector output = Vector.emptyOf(inputs.getSize());
        range(0, inputs.getSize()).forEach(index -> {
                double value = inputs.getValue(index);
                output.setValue(index, activate(value));
        });
        return output;
    }

    public Vector derivative(Vector inputs) {
        Vector output = Vector.emptyOf(inputs.getSize());
        range(0, inputs.getSize()).forEach(index -> {
            double value = inputs.getValue(index);
            output.setValue(index, derivative(value));
        });
        return output;
    }

    public abstract double activate(double input);

    public abstract double derivative(double input);

}