package com.patres.neuralnetwork.data.data;

import com.patres.neuralnetwork.math.Matrix;

public class ImageInput extends Input {

    private final Matrix dataMatrix;

    public Matrix getDataAsMatrix() {
        return dataMatrix;
    }

    public ImageInput(Matrix dataMatrix, int label) {
        super(dataMatrix.toVector(), label);
        this.dataMatrix = dataMatrix;
    }

}
