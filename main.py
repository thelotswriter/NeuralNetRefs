"""PURPOSE OF THIS PROJECT
    This is a collection of relatively simple pytorch networks.
    This fulfills three main goals:

        1)  Stay in the head-space of deep learning while contemplating
        future bigger, more interesting projects
        2)  Build confidence in my abilities with PyTorch.
        3)  Provide a reference point for future projects.
        4)  Demonstrate coding techniques.

    In the interest of time, I do not intend to get too involved, instead
    focusing on playing with parameters and obtaining reasonable results."""

import mlp
import conv2dClassifier


# Runs one of the models based on the model_num
import styleTransfer


def select_model(model_num=0, verbose=False, load=False, train=True):
    if model_num == 0:
        mlp.run(verbose=verbose, load=load, train_net=train)
    elif model_num == 1:
        conv2dClassifier.run(verbose=verbose, load=load, train_net=train)
    elif model_num == 2:
        styleTransfer.run()
    else:
        print('Invalid selection.')


if __name__ == '__main__':
    # Loop, selecting and running nets until done
    while True:
        # Select a net or quit
        model_num = input('Enter the number to run the corresponding model:\n'
                          '0 - Multilayer Perceptron Classifier\n'
                          '1 - Convolutional Classifier\n'
                          '2 - Style Transfer\n'
                          'Quit - Close program\n')
        if 'quit' in model_num.lower():
            break
        model_num = int(model_num)
        # Check for booleans if relevant.
        # Should extra info be printed?
        # Should a previous model be loaded?
        # Should the model be trained?
        if model_num != 2:
            verbose = input('Verbose? (Y/N)\n')
            verbose = ("y" in verbose.lower())
            load = input('Load model? (Y/N)\n')
            load = ("y" in load.lower())
            train = input('Train net? (Y/N)\n')
            train = ("y" in train.lower())
            select_model(model_num=model_num, verbose=verbose, load=load, train=train)
        else:
            select_model(model_num=model_num)
    print('Enjoy your day!')
