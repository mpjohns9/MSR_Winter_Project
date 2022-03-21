""" This file controls the baseball grip classification package.

Running main.py allows a user to specify arguments to customize
training a model and running classification in real-time.

To train and save a model run: python3 main.py --train "[SAVED MODEL PATH]"
The epochs argument allows a user to specify the number of training
epochs (default is 1000).

To run the prediction file using your laptop webcam, use the argument
--predict "[SAVED MODEL PATH]"

Direct any questions to marshalljohnson2022@u.northwestern.edu
"""

import argparse
import train_model
from predict import predict_grip

def main():
    """ The main function. """
    network = train_model.Network(args.train)
    if not (args.train or args.predict):
        print("Neither train nor predict arguments declared. Exiting...")
        quit()

    if args.train:
        if args.epochs:
            network.train(epochs=args.epochs)
        else:
            network.train() 

    if args.predict:
        predict_grip(args.predict)
    

if __name__ == "__main__":
    # initialize terminal arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=str, required=False)
    parser.add_argument('--epochs', type=int, required=False)
    parser.add_argument('--predict', type=str, required=False)
    args = parser.parse_args()
    main()