# NEAT_HeroJump

NEAT_HeroJump is a 2D game environment built in Python for experimenting with the NEAT algorithm, allowing neural networks to evolve and learn how to play the game.

## About the Project

- **Purpose**: Created to explore self-learning AI and test if a neural network could learn the game mechanics.
- **Background**: My first major NEAT project in Python, developed when I was 15 years old (10th grade).
- **Development Time**: Built over about 3 weeks, working 2-3 hours daily.
- **Libraries Used**: `neat`, `matplotlib`, and `pickle`.
- **Technical Details**: Custom collision handling and camera movement implemented from scratch.
- **Code Structure**: The project contains around 600 lines of code.

## Game Goal

- **Objective**: Calculate the optimal angle (between -1 and 1, later converted to degrees) to jump from one pillar to the next.
- **Neural Network Inputs**: For each jump, the neural network receives:
  - Current Y position
  - Next pillar’s Y position
  - Horizontal distance to the next pillar (X-axis)
- **Training Results**: After around 2,000 epochs, the AI learned to jump accurately, demonstrating the power of the NEAT algorithm.

This project was an excellent way to learn and apply the NEAT algorithm in Python.

## Gameplay Screenshots

![Gameplay Screenshot 1](screenshots/screen1.jpg?raw=true)
![Gameplay Screenshot 2](screenshots/screen2.jpg?raw=true)
![Gameplay Screenshot 3](screenshots/screen3.jpg?raw=true)
![Gameplay Screenshot 4](screenshots/screen4.jpg?raw=true)
