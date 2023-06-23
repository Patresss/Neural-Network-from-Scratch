package com.patres.neuralnetwork.network.cost;

public abstract class Cost {
    public abstract double calculateCost(double output, double expectedOutput);

    public abstract double derivative(double output, double expectedOutput);
}
