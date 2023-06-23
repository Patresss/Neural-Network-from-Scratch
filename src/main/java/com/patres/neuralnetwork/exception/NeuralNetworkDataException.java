package com.patres.neuralnetwork.exception;

public class NeuralNetworkDataException extends RuntimeException {


    public NeuralNetworkDataException() {
    }

    public NeuralNetworkDataException(String message) {
        super(message);
    }

    public NeuralNetworkDataException(String message, Throwable cause) {
        super(message, cause);
    }
}
