package com.patres.neuralnetwork.utils;

import java.util.Collection;
import java.util.function.BiConsumer;

public class StreamUtils {

    public static <T> void forEach(Collection<T> collection,
                                   BiConsumer<T, Integer> consumer) {
        int index = 0;
        for (T object : collection){
            consumer.accept(object, index++);
        }
    }
}
