from datagenerator import DataGenerator
import os


class DataReader:

    def __init__(self, data_dir, scale_size, shuffle=False, input_channels=1, nb_classes=5):

        self.__n_classes = nb_classes
        self.__shuffle = shuffle
        self.__input_channels = input_channels
        self.__scale_size = scale_size
        self.__generator = None
        self.read_data(data_dir)

    def get_generator(self):
        return self.__generator # возвращает генерацию

    def read_data(self, data_dir):
        # сначала посмотр всех файлов  1 раз, больше метод Walk не находят, переменная I становится больше
        # только после этого  добавляются файлы и метки,
        # data_dir-каталог в котором  работаем
        patterns = [] # берем шаблоны
        labels = [] # берем метки

        i = -1
        for root, dirs, files in os.walk(data_dir): # Метод Python walk () генерирует имена файлов
            # в дереве каталогов, путем обхода дерева сверху вниз или снизу вверх
            if i < 0:
                i = i + 1
            else:
                [patterns.append(root + '/' + file) for file in files] # Метод append () принимает один
                # элемент в качестве входного параметра и добавляет его в конец списка,
                # то есть добавляется по входному файлу и его метке
                [labels.append(i) for file in files]
                i = i + 1

        self.__generator = DataGenerator(patterns, labels, self.__scale_size, self.__shuffle,
                                         self.__input_channels, self.__n_classes)
