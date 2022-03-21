import argparse
import train_model

def main():
    """ The main function. """
    network = train_model.Network(args.train)
    if not (args.train or args.predict):
        print("Neither train nor predict arguments declared. Exiting...")
        quit()

    if args.epochs and args.train:
        network.train(epochs=args.epochs)
    else:
        network.train()    
    

if __name__ == "__main__":
    # initialize terminal arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=str, required=False)
    parser.add_argument('--epochs', type=int, required=False)
    parser.add_argument('--predict', type=str, required=False)
    args = parser.parse_args()
    main()


    """
    TASKS:

    """