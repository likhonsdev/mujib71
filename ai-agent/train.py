
import argparse
import time

def train(epochs):
    for epoch in range(epochs):
        print("Epoch %d/%d training..." % (epoch+1, epochs))
        time.sleep(2)
        print("Epoch %d complete, accuracy: %.2f" % (epoch+1, 0.8 + epoch*0.01))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1)
    args = parser.parse_args()
    train(args.epochs)
