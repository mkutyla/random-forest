import src.utilities.quality as qt


def run():
    quality = qt.Quality([0, 1], [0, 0, 0, 1, 1, 1], [0, 0, 1, 1, 0, 1])
    quality.get_confusion_matrix()
    quality.print_confusion_matrix()
    print(quality.get_class_measures(str(1)))


if __name__ == '__main__':
    run()
