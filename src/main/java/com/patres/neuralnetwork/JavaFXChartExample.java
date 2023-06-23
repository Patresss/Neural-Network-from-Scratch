package com.patres.neuralnetwork;

import javafx.application.Application;
import javafx.scene.Scene;
import javafx.scene.chart.LineChart;
import javafx.scene.chart.NumberAxis;
import javafx.scene.chart.XYChart;
import javafx.stage.Stage;

public class JavaFXChartExample extends Application {

    @Override
    public void start(Stage stage) {
        // Tworzenie osi
        final NumberAxis xAxis = new NumberAxis();
        final NumberAxis yAxis = new NumberAxis();
        xAxis.setLabel("X");
        yAxis.setLabel("Y");

        // Tworzenie wykresu
        final LineChart<Number, Number> lineChart = new LineChart<>(xAxis, yAxis);
        lineChart.setTitle("Wykres");
        
        // Tworzenie serii danych
        XYChart.Series<Number, Number> series = new XYChart.Series<>();
        series.setName("Dane");
        series.getData().add(new XYChart.Data<>(1, 2));
        series.getData().add(new XYChart.Data<>(2, 3));
        series.getData().add(new XYChart.Data<>(3, 1));
        
        // Dodawanie serii danych do wykresu
        lineChart.getData().add(series);
        
        // Tworzenie sceny i dodawanie wykresu
        Scene scene = new Scene(lineChart, 800, 600);
        
        // Ustawianie sceny na etapie
        stage.setScene(scene);
        
        // Wyświetlanie etapu
        stage.show();
    }

    public static void main(String[] args) {
        launch(args);
    }
}