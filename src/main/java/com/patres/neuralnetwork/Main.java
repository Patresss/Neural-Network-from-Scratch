package com.patres.neuralnetwork;

import com.patres.neuralnetwork.data.DigitRecognizer;

import static com.patres.neuralnetwork.utils.FileResourcesUtils.getFileFromResource;

public class Main {

    public static void main(String[] args)  {

        DigitRecognizer digitRecognizer = new DigitRecognizer(
                getFileFromResource("image/mnist_train.csv"),
                getFileFromResource("image/mnist_test.csv"),
                28,
                28);
        digitRecognizer.run();


//        //create a dataset object to hold features
//        Dataset dataset = new Dataset(100, 3);
//
//        LayerDense dense1 = new LayerDense(2, 3);
//        ActivationReLU activation1 = new ActivationReLU();
//
//
//        LayerDense dense2= new LayerDense(3, 3);
//        ActivationSoftMax activation2 = new ActivationSoftMax();
//
//        Matrix dense1Output = dense1.forward(dataset.getDataAsMatrix());
//        Matrix activation1Output = activation1.forward(dense1Output);
//
//        Matrix dense2Output = dense2.forward(activation1Output);
//        Matrix activation2Output = activation2.forward(dense2Output);
//
//        activation2Output.print(5);
//
//        Plot plt = Plot.create();
//        plt.plot()
//                .add(Arrays.asList(1.3, 2))
//                .label("label")
//                .linestyle("--");
//        plt.xlabel("xlabel");
//        plt.ylabel("ylabel");
//        plt.text(0.5, 0.2, "text");
//        plt.title("Title!");
//        plt.legend();
//        plt.show();


    }


}
