package com.patres.neuralnetwork.data.data;

import com.patres.neuralnetwork.math.Vector;

public class Input {

    private final Vector data;
    private final Vector labels;
    private Integer label;


    public Input(Vector data, int label) {
        this.data = data;
        this.label = label;
        final Vector labels = Vector.emptyOf(data.getSize());
        labels.setValue(label, 1);
        this.labels = labels;
    }

    public Input(Vector data, Vector labels) {
        this.data = data;
        this.labels = labels;
    }


    public Vector getData() {
        return data;
    }


    public Vector getLabels() {
        return labels;
    }

    public Integer getLabel() {
        return label;
    }

    public String getLabelAsString() {
        return label != null ? label.toString() : labels.toString();
    }


    @Override
    public String toString() {
        return getLabelAsString() + System.lineSeparator() +
                data;
    }
}
