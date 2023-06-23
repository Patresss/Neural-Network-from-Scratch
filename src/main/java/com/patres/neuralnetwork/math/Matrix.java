package com.patres.neuralnetwork.math;

import java.util.Arrays;
import java.util.Random;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import static java.util.stream.Collectors.joining;
import static java.util.stream.IntStream.range;

public class Matrix {

    private final static Random random = new Random(0);

    private final double[][] data;
    private final int numberOfRows;
    private final int numberOfColumns;

    private Matrix(final double[][] data) {
        this.data = data;
        this.numberOfRows = data.length;
        this.numberOfColumns = data.length > 0 ? data[0].length : 0;
    }

    private Matrix(int numberOfRows, int numberOfColumns) {
        this.data = new double[numberOfRows][numberOfColumns];
        this.numberOfRows = numberOfRows;
        this.numberOfColumns = numberOfColumns;
    }

    public static Matrix of(final double[][] data) {
        return new Matrix(data);
    }

    public static Matrix emptyOf(int numberOfRows, int numberOfColumns) {
        return new Matrix(numberOfRows, numberOfColumns);
    }

    public static Matrix gaussianOf(int numberOfRows, int numberOfColumns) {
        final Matrix matrix = new Matrix(numberOfRows, numberOfColumns);
        for (int rowIndex = 0; rowIndex < numberOfRows; rowIndex++) {
            for (int columnIndex = 0; columnIndex < numberOfColumns; columnIndex++) {
                double value = 0.1 * random.nextGaussian();
                matrix.setValue(rowIndex, columnIndex, value);
            }
        }
        return matrix;
    }


    public void setValue(int rowIndex, int columnIndex, double value) {
        data[rowIndex][columnIndex] = value;
    }

    public double getValue(int rowIndex, int columnIndex) {
        return data[rowIndex][columnIndex];
    }

    public int getNumberOfRows() {
        return numberOfRows;
    }

    public int getNumberOfColumns() {
        return numberOfColumns;
    }

    public Matrix dotProduct(final Matrix matrix) {
        final Matrix output = Matrix.emptyOf(numberOfRows, matrix.getNumberOfColumns());
        for (int i = 0; i < output.getNumberOfRows(); i++) {
            for (int j = 0; j < output.getNumberOfColumns(); j++) {
                double value = 0;
                for (int k = 0; k < numberOfColumns; k++) {
                    value += getValue(i, k) * matrix.getValue(k, j);
                }
                output.setValue(i, j, value);
            }
        }
        return output;
    }

    public Matrix add(final Vector vector) {
        final Matrix output = Matrix.emptyOf(numberOfRows, numberOfColumns);
        for (int i = 0; i < numberOfRows; i++) {
            for (int j = 0; j < numberOfColumns; j++) {
                final double value = getValue(i, j) + vector.getValue(j);
                output.setValue(i, j, value);
            }
        }
        return output;
    }

    @Override
    public String toString() {
        return range(0, numberOfRows)
                .mapToObj(this::toStringRow)
                .collect(joining(System.lineSeparator()));
    }

    private String toStringRow(final int indexOfRow) {
        return range(0, numberOfColumns)
                .mapToObj(column -> String.format("%5f", getValue(indexOfRow, column)))
                .collect(joining(" "));
    }

    public void print() {
        print(numberOfRows);
    }

    public void print(int maxRows) {
        range(0, Math.min(maxRows, numberOfRows)).forEach(row -> {
            range(0, numberOfColumns).forEach(column -> {
                double value = getValue(row, column);
                System.out.printf("%5f ", value);
            });
            System.out.println();
        });
    }

    public double getMaxOfRow(final int rowIndex) {
        return Arrays.stream(data[rowIndex]).max().orElse(0.0);
    }

    public double calculateSumOfRow(final int rowIndex) {
        return Arrays.stream(data[rowIndex]).sum();
    }

    public Vector toVector() {
        double[] vector = new double[numberOfRows * numberOfColumns];
        int vectorIndex = 0;
        for (int i = 0; i < numberOfRows; i++) {
            for (int j = 0; j < numberOfColumns; j++) {
                vector[vectorIndex++] = data[i][j];
            }
        }
        return Vector.of(vector);
    }

}
