package com.patres.neuralnetwork.network.activation;

public class ActivationReLU extends Activation {

    private static final double LEAK = 0.01;

    @Override
    public double activate(double input) {
        return Math.max(0, input);
    }

    @Override
    public double derivative(double input) {
        return input <= 0 ? LEAK : 1;
    }

}