package com.patres.neuralnetwork;

import com.patres.neuralnetwork.math.Matrix;

import java.util.Random;

public class Dataset {
        private double[][] data;
        private int[] classLabels;

        private final Random random = new Random(0);

        public Dataset(int pointsPerClass, int numberOfClasses) {
            prepareData(pointsPerClass, numberOfClasses);
        }


    /**
     * https://cs231n.github.io/neural-networks-case-study/
     *
     * N = 100 # number of points per class
     * D = 2 # dimensionality
     * K = 3 # number of classes
     * X = np.zeros((N*K,D)) # data matrix (each row = single example)
     * y = np.zeros(N*K, dtype='uint8') # class labels
     * for j in range(K):
     *   ix = range(N*j,N*(j+1))
     *   r = np.linspace(0.0,1,N) # radius
     *   t = np.linspace(j*4,(j+1)*4,N) + np.random.randn(N)*0.2 # theta
     *   X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
     *   y[ix] = j
     * # lets visualize the data:
     * plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
     * plt.show(
     */
    private void prepareData(int pointsPerClass, int numberOfClasses) {
            data = new double[pointsPerClass * numberOfClasses][2];
            classLabels = new int[pointsPerClass * numberOfClasses];
            int ix = 0;
            for (int class_number = 0; class_number < numberOfClasses; class_number++) {
                double r = 0;
                double t = class_number * 4;
                while (r <= 1 && t <= (class_number + 1) * 4) {
                    double random_t = t + random.nextInt(pointsPerClass) * 0.2;
                    data[ix][0] = r * Math.sin(random_t * 2.5);
                    data[ix][1] = r * Math.cos(random_t * 2.5);
                    classLabels[ix] = class_number;
                    r += 1.0 / (pointsPerClass - 1);
                    t += 4.0 / (pointsPerClass - 1);
                    ix++;
                }
            }
        }


    public double[][] getData() {
        return data;
    }

    public Matrix getDataAsMatrix() {
        return Matrix.of(data);
    }

    public int[] getClassLabels() {
        return classLabels;
    }
}