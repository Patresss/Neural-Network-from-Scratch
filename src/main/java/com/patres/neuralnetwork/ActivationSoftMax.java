package com.patres.neuralnetwork;

import com.patres.neuralnetwork.math.Matrix;

import static java.util.stream.IntStream.range;

public class ActivationSoftMax {

    /**
     * exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
     * probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
     * self.output = probabilities
     */
    public Matrix forward(Matrix inputs) {
        Matrix expValues = calculateExpValues(inputs);
        return calculateProbabilities(expValues);
    }

    public Matrix calculateExpValues(Matrix inputs) {
        Matrix output = Matrix.emptyOf(inputs.getNumberOfRows(), inputs.getNumberOfColumns());
        range(0, inputs.getNumberOfRows()).forEach(row -> {
            double maxOfRow = inputs.getMaxOfRow(row);
            range(0, inputs.getNumberOfColumns()).forEach(column -> {
                double value = inputs.getValue(row, column);
                double expValue = Math.exp(value - maxOfRow);
                output.setValue(row, column, expValue);
            });
        });
        return output;
    }

    public Matrix calculateProbabilities(Matrix inputs) {
        Matrix output = Matrix.emptyOf(inputs.getNumberOfRows(), inputs.getNumberOfColumns());
        range(0, inputs.getNumberOfRows()).forEach(row -> {
            double sumOfRow = inputs.calculateSumOfRow(row);
            range(0, inputs.getNumberOfColumns()).forEach(column -> {
                double value = inputs.getValue(row, column) / sumOfRow;
                output.setValue(row, column, value);
            });
        });
        return output;
    }

}