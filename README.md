# Tic-Tac-Toe AI with Neural Network and Genetic Algorithm
### 📌 Project Overview
This project implements a Reinforcement Learning approach to train a Neural Network (NN) to play Tic-Tac-Toe. The network is trained using a Genetic Algorithm (GA), not backpropagation. As an opponent and trainer, the Minimax algorithm is used with varying difficulty levels.

This work was developed for the Artificial Intelligence course at [PUCRS](https://portal.pucrs.br/), under the supervision of Professor Silvia Moraes.

## 🧠 How It Works
### 🕹 Minimax Difficulty Modes
The Minimax algorithm simulates an intelligent opponent. During training, it is configured with three levels:

Easy: 25% Minimax, 75% random moves

Medium: 50% Minimax, 50% random

Hard: 100% Minimax

Each chromosome (i.e., a set of NN weights) plays 6 games, 2 at each difficulty level.

### 🧬 Genetic Algorithm
Population: 10 chromosomes, each encoding 237 real values (weights and biases).

Fitness Function:

Win: +10

Draw: +1

Loss: -5

Invalid move: -20

Selection:

Elitism: top 2 passed unchanged

Tournament: crossover between top 2 or top 4 (if population score > -1000)

Crossover: one-point crossover

Mutation: 1% normally; 50% if all chromosomes have worst score (-1200)

### 🤖 Neural Network
A simple feedforward network (MLP) with the following architecture:

Input Layer: 9 neurons (board cells: 1 = own move, -1 = opponent, 0 = empty)

Hidden Layer: 12 neurons

Output Layer: 9 neurons (probabilities for each board cell)

Total weights: 237

### 🎯 Training Goal
Training continues until a chromosome achieves a perfect score (60), typically reached in ~60 generations. The trained network can then play against human players.

### 🖥️ Features
Front-end interface for Tic-Tac-Toe

#### Modes:

  Human vs Minimax
  
  Minimax trains the Neural Network
  
  Human vs Trained Neural Network
  
  Real-time tracking of NN evolution
  
  Accuracy evaluation post-training

## 📊 Results
The GA successfully evolves the NN to defeat the Minimax algorithm consistently, with adjustments to mutation rate necessary to stabilize learning.

## 🎥 Demo Video
[YouTube Presentation](https://www.youtube.com/watch?v=ryqGGci4mgo)

## 🚀 How to Run
Follow the steps below to set up and run the project on your local machine:

1. Clone the Repository

        git clone https://github.com/GuilhermeGMV/T2IA
        cd T2IA

2. Set Up the Environment
Make sure you have Python 3.7+ installed. Then install the required dependencies:

        pip install numpy pandas
   
4. Run the Project
To start training the neural network and interact with the game:

        python main.py
   
This will execute the training using the Genetic Algorithm and display the game interface as defined in your front end.
