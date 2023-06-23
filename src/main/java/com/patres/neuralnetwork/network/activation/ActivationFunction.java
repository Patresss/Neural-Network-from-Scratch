package com.patres.neuralnetwork.network.activation;

import com.patres.neuralnetwork.math.Vector;

import java.util.function.Function;

import static java.lang.Math.exp;
import static java.util.stream.IntStream.range;

public enum ActivationFunction {

    RE_LU(
            input -> Math.max(0.0, input),
            input -> input <= 0 ? 0.0 : 1.0),
    SIGMOID(
            input -> 1.0 / (1.0 + exp(-input)),
            input -> {
                double activation = 1.0 / (1.0 + exp(-input));
                return activation * (1.0 - activation);
//                double activation = 1.0 / (1.0 + (1.0/ exp(input)));
//                return input * (1.0 - input);
            });

    private final Function<Double, Double> activation;
    private final Function<Double, Double> derivative;

    ActivationFunction(Function<Double, Double> activation, Function<Double, Double> derivative) {
        this.activation = activation;
        this.derivative = derivative;
    }

    public Vector activate(Vector inputs) {
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

    public double activate(double input) {
        return activation.apply(input);
    }

    public double derivative(double input) {
        return derivative.apply(input);
    }

}