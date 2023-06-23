package com.patres.neuralnetwork.network.cost;

import static java.lang.Math.pow;

public class SquareCost extends Cost {

    @Override
    public double calculateCost(double output, double expectedOutput) {
        return pow(output - expectedOutput, 2);
    }

    @Override
    public double derivative(double output, double expectedOutput) {
        return 2*(output - expectedOutput);
    }
}
