from neuron import Neuron
from math import pow


class OneLayerNet:

    def __init__(self, inputs_count, output_neurons_count): # здесь количество входных и выходых нейронов
        self.__inputs_count = inputs_count
        self.__neurons = []
        for j in range(output_neurons_count):
            self.__neurons.append(Neuron(inputs_count))


    def train(self, vector, learning_rate):

        for j in range(len(self.__neurons)): # переборка обучающих векторов
            self.__neurons[j].calc_y(vector.get_x()) # из вектора берем обучающий  вектор ,
            # calc_y вычисляет вес в нейроне

        weights_deltas = [[0] * (len(vector.get_x()) + 1)] * len(self.__neurons) # вычисляем вес для дельт
        loss = 0
        # переборка всех нейронов
        for j in range(len(self.__neurons)):
            sigma = (vector.get_d()[j] - self.__neurons[j].get_y()) \
                    * self.__neurons[j].derivative() # считаем сигму для каждого нейрона
            weights_deltas[j][0] = learning_rate * sigma # нулевой строке массива дельт веса
            # присваиваем значение скорости обучения на сигму нейрона
            wlen = len(self.__neurons[j].get_weights())# кол-во веса у контретного нейрона
            for i in range(wlen): #проход по всему весу каждой строки
                weights_deltas[j][i] = learning_rate * sigma * vector.get_x()[i]
                # считаем новый вес для каждого элемента массива
                # строки умножаются на первое значение обучающего вектора
            self.__neurons[j].correct_weights(weights_deltas[j])
            # для каждого нейрона корректируется вес по средствам прибавления получившейся дельты веса
            # вычисляем сумму  ошибок для каждого нейрона
            loss += pow(vector.get_d()[j] - self.__neurons[j].get_y(), 2)
            # квадрат разности между желаемым и действительлным





        # Возвращение половины ошибок
        return 0.5 * loss

    def test(self, vector):
        y = [0] * len(self.__neurons)
        for j in range(len(self.__neurons)):
            self.__neurons[j].calc_y(vector.get_x()) # из вектора берем обучающий  вектор
            # calc_y вычисляет вес в нейроне
            y[j] = self.__neurons[j].get_y()
        return y




















