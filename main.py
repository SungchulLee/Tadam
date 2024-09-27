import torch
import torch.nn as nn
import torch.optim as optim
from my_own_optim import Tadam 

from global_name_space import ARGS
from load_data import load_data
from model import Net
from train import train, compute_accuracy
from utils import show_batch_or_ten_images_with_label_and_predict

def save_model(model):
    torch.save(model.state_dict(), ARGS.path)

def load_model():
    model = Net().to(ARGS.device)
    model.load_state_dict(torch.load(ARGS.path))
    return model

def main():
    trainloader, testloader = load_data()

    model = Net().to(ARGS.device)
    loss_ftn = nn.CrossEntropyLoss()

    # To use Tadam, you need to specify two additional parameters beyond those used by Adam.
    # The first is 'total_steps', which defines the total number of Tadam gradient descent iterations.
    # The second is 'gamma', which controls the adjustment of the trust region's size.
    # If the optimizer's performance is poor (i.e., \bar{\rho}_n < \gamma), the trust region is shrunk.
    # If the optimizer's performance is excellent (i.e., \bar{\rho}_n > 1 - \gamma), the trust region is expanded.
    # If the optimizer's performance is moderate (i.e., \gamma <= \bar{\rho}_n <= 1 - \gamma), the trust region remains unchanged.
    total_steps = ARGS.epochs * (60_000 // ARGS.batch_size)
    gamma = 0.25

    # option 1: Use Tadam in my_own_optim.py
    # optimizer = Tadam(model.parameters(), total_steps=total_steps, gamma=0.25, lr=ARGS.lr)
    # option 2: Use Tadam /Users/sungchul/opt/anaconda3/lib/python3.9/site-packages/torch/optim/tadam.py
    optimizer = optim.Tadam(model.parameters(), total_steps=total_steps, gamma=gamma, lr=ARGS.lr)
    
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=ARGS.gamma)

    # Display sample images and predictions before training
    show_batch_or_ten_images_with_label_and_predict(testloader, model)

    # Train the model
    train(model, trainloader, loss_ftn, optimizer, scheduler)

    # Display sample images and predictions after training
    show_batch_or_ten_images_with_label_and_predict(testloader, model)

    # Save and load the trained model
    save_model(model)
    model = load_model()

    # Compute and display accuracy on test data
    compute_accuracy(model, testloader)

if __name__ == "__main__":
    main()