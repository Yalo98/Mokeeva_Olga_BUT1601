import numpy as np
# класс Vector возвращает параметры х и d в виде массива, внутри __init__ делая его массивом нужного вида
class Vector:
    def __init__(self, x, d):
        # x.shape- это размер массива
        if len(x.shape) > 2:
            # np.asanyarray преобразует данные в массив
            self.__x = list(np.asanyarray(x).reshape(-2))
            # reshape- изменяет форму массива
        else:
            self.__x = x
        self.__d = d

    def get_x(self):
        return self.__x

    def get_d(self):
        return self.__d
