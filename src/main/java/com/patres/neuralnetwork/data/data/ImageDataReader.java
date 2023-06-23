package com.patres.neuralnetwork.data.data;

import com.patres.neuralnetwork.math.Matrix;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.List;

public class ImageDataReader {

    private final File file;
    private final int rows;
    private final int columns;

    public ImageDataReader(File file, int rows, int columns) {
        this.file = file;
        this.rows = rows;
        this.columns = columns;
    }

    public List<ImageInput> readData(){
        List<ImageInput> imageInputs = new ArrayList<>();

        try (BufferedReader dataReader = new BufferedReader(new FileReader(file))){

            String line;

            while((line = dataReader.readLine()) != null){
                String[] lineItems = line.split(",");

                Matrix data = Matrix.emptyOf(rows, columns);
                int label = Integer.parseInt(lineItems[0]);

                int i = 1;

                for(int row = 0; row < rows; row++){
                    for(int col = 0; col < columns; col++){
                        data.setValue(row, col, Integer.parseInt(lineItems[i]));
                        i++;
                    }
                }
                imageInputs.add(new ImageInput(data, label));
            }

        } catch (Exception e){
            throw new IllegalArgumentException("File not found " + file.getAbsolutePath());
        }

        return imageInputs;

    }

}
