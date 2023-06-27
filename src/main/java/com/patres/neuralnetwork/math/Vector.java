package com.patres.neuralnetwork.math;

import java.util.Random;
import java.util.function.Function;

import static java.util.stream.Collectors.joining;
import static java.util.stream.IntStream.range;

public class Vector {

    private final static Random random = new Random(0);

    private final double[] data;

    private Vector(final int size) {
        this.data = new double[size];
    }

    private Vector(final double[] data) {
        this.data = data;
    }

    public static Vector emptyOf(final int size) {
        return new Vector(size);
    }

    public static Vector of(final double... data) {
        return new Vector(data);
    }

    public static Vector gaussianOf(final int size) {
        final double[] data = new double[size];
        for (int i = 0; i < size; i++) {
            data[i] = 0.1 * random.nextGaussian();
        }
        return new Vector(data);
    }

    public double getValue(final int index) {
        return data[index];
    }

    public void setValue(final int index, final double value) {
        data[index] = value;
    }

    public double dotProduct(final Vector vector) {
        double result = 0.0;
        for (int i = 0; i < getSize(); i++) {
            result += getValue(i) * vector.getValue(i);
        }
        return result;
    }

    public Vector multiply(final double scalar) {
        return createNewVector((index) -> getValue(index) * scalar);
    }

    public Vector multiply(final Vector vector) {
        return createNewVector((index) -> getValue(index) * vector.getValue(index));
    }

    public Vector add(final Vector vector) {
        return createNewVector((index) -> getValue(index) + vector.getValue(index));
    }

    public double sumValues() {
        double result = 0.0;
        for (int i = 0; i < getSize(); i++) {
            result += getValue(i);
        }
        return result;
    }

    public Vector add(final double value) {
        return createNewVector((index) -> getValue(index) + value);
    }

    public Vector subtract(final Vector vector) {
        return createNewVector((index) -> getValue(index) - vector.getValue(index));
    }

    public Vector pow(final double exponent) {
        return createNewVector((index) -> Math.pow(getValue(index), exponent));
    }

    public Vector pow() {
        return pow(2.0);
    }

    public Vector createNewVector(Function<Integer, Double> valueSupplier) {
        final Vector result = Vector.emptyOf(getSize());
        for (int i = 0; i < getSize(); i++) {
            result.setValue(i, valueSupplier.apply(i));
        }
        return result;
    }


    public int getSize() {
        return data.length;
    }

    public int getMaxIndex(){
        double max = 0;
        int index = 0;
        for(int i = 0; i < data.length; i++){
            if(data[i] >= max){
                max = data[i];
                index = i;
            }
        }
        return index;
    }

    @Override
    public String toString() {
        return range(0, getSize())
                .mapToObj(column -> String.format("%5f", getValue(column)))
                .collect(joining(" "));
    }
}
