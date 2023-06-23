package com.patres.neuralnetwork.network.cost;

import com.patres.neuralnetwork.math.Vector;

import static java.lang.Math.pow;

public enum CostFunction {

    SQUARE(
            (output, expectedOutput) -> pow(output - expectedOutput, 2.0),
            (output, expectedOutput) -> 2.0 * (output - expectedOutput)
    ),
    JAVA_POINT_SQUARE(
            (output, expectedOutput) -> 0.5 * pow(output - expectedOutput, 2.0),
            (output, expectedOutput) -> (output - expectedOutput)
    );;

    private final CostProvider costCalculator;
    private final CostProvider derivativeCostCalculator;

    CostFunction(CostProvider costCalculator, CostProvider derivativeCostCalculator) {
        this.costCalculator = costCalculator;
        this.derivativeCostCalculator = derivativeCostCalculator;
    }

    @FunctionalInterface
    interface CostProvider {
        double calculate(double output, double expectedOutput);
    }

    public Vector calculateCost(Vector output, Vector expectedOutput) {
        return output.createNewVector((index) -> calculateCost(output.getValue(index), expectedOutput.getValue(index)));
    }

    public Vector calculateDerivativeCost(Vector output, Vector expectedOutput) {
        return output.createNewVector((index) -> calculateDerivativeCost(output.getValue(index), expectedOutput.getValue(index)));
    }

    public double calculateCost(double output, double expectedOutput) {
        return costCalculator.calculate(output, expectedOutput);
    }

    public double calculateDerivativeCost(double output, double expectedOutput) {
        return derivativeCostCalculator.calculate(output, expectedOutput);
    }
}