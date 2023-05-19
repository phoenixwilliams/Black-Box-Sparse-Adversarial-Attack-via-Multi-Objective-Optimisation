# from ProposedMethod.QueryEfficient.Scratch import Attack
from MOAA.MOAA import Attack
from Cifar10Models import Cifar10Model # Can be changes to ImageNetModels
from LossFunctions import UnTargeted, Targeted
import numpy as np
import argparse
import os


if __name__ == "__main__":
    """
    Non-Targeted
    pc = 0.1
    pm = 0.4
    
    Targeted:
    pc = 0.1
    pm = 0.2
    """
    np.random.seed(0)

    pc = 0.1
    pm = 0.4

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="0 or 1", type=int)
    parser.add_argument("--start", type=int)
    parser.add_argument("--save_directory", type=int)
    args = parser.parse_args()

    x_test, y_test, y_target = get_images_and_labels() # replace this with your own method of getting the images and
    # labels.
    model = Cifar10Model(args.model) # replace this with you own model, assumed to return probabilities on __call__(image)

    #loss = Targeted(model, y_test[i], y_target[i], to_pytorch=True)
    loss = UnTargeted(model, y_test[i], to_pytorch=True) # to_pytorch is True only is the model is a pytorch model
    params = {
        "x": x_test, # Image is assume to be numpy array of shape height * width * 3
        "eps": 24, # number of changed pixels
        "iterations": 1000 // 2, # model query budget / population size
        "pc": pc, # crossover parameter
        "pm": pm, # mutation parameter
        "pop_size": 2, # population size
        "zero_probability": 0.3,
        "include_dist": True, # Set false to not consider minimizing perturbation size
        "max_dist": 1e-5, # l2 distance from the original image you are willing to end the attack
        "p_size": 2.0, # Perturbation values have {-p_size, p_size, 0}. Change this if you want smaller perturbations.
        "tournament_size": 2, #Number of parents compared to generate new solutions, cannot be larger than the population
        "save_directory": args.save_directory
    }
    attack = Attack(params)
    attack.attack(loss)
