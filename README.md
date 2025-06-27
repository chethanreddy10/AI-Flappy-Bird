# Flappy Bird AI with NEAT

An AI implementation that learns to play Flappy Bird using the NEAT (NeuroEvolution of Augmenting Topologies) algorithm. Watch as neural networks evolve over generations to master the art of navigating through pipes!

## 🎮 Overview

This project demonstrates machine learning in action by training an AI to play Flappy Bird. The AI uses neuroevolution to develop strategies for timing jumps perfectly to avoid obstacles and achieve high scores.



## 🚀 Features

- **AI-Powered Gameplay**: Neural networks learn to play without human intervention
- **Real-time Evolution**: Watch birds improve across generations
- **Visual Feedback**: See live statistics including generation, score, and population
- **Automatic Saving**: Best-performing models are saved automatically
- **Customizable Parameters**: Easily adjust game physics and AI settings

## 📁 Project Structure

```
flappy_bird/
├── main.py          # Main execution and NEAT setup
├── bird.py          # Bird class implementation
├── pipe.py          # Pipe class implementation
├── base.py          # Base (ground) class implementation
└── config-feedforward.txt  # NEAT algorithm configuration
```

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/chethanreddy10/AI-Flappy-Bird.git
   cd AI-Flappy-Bird
   ```

2. **Install dependencies**
   ```bash
   pip install pygame neat-python
   ```

## 🎯 Usage

Run the AI training:
```bash
python main.py
```

The game window will open showing:
- Multiple birds attempting to navigate pipes
- Current generation number
- Live score counter
- Number of birds still alive
- Optional debug lines showing AI decision-making

## 🧠 How It Works

### NEAT Algorithm
The AI uses NEAT (NeuroEvolution of Augmenting Topologies) to evolve neural networks:

1. **Initial Population**: Random neural networks control each bird
2. **Fitness Evaluation**: Birds earn points for:
   - Staying alive (+0.1 per frame)
   - Successfully passing pipes (+5 per pipe)
3. **Selection**: Best-performing birds are chosen for reproduction
4. **Evolution**: Neural networks mutate and crossover to create new generations
5. **Repeat**: Process continues for up to 50 generations

### Game Mechanics
- Birds automatically move forward
- Neural networks decide when to jump
- Collision detection with pipes and ground
- Physics simulation for realistic movement

## 🎨 Classes Overview

### Bird Class (`bird.py`)
- Handles bird physics and animation
- Manages jumping mechanics and rotation
- Provides collision detection

### Pipe Class (`pipe.py`)
- Creates and manages pipe obstacles
- Handles collision detection with birds
- Controls pipe movement and gap positioning

### Base Class (`base.py`)
- Implements scrolling ground effect
- Provides visual foundation for the game

## ⚙️ Customization

### Game Settings (`config.py`)
```python
WIN_WIDTH = 500      # Window width
WIN_HEIGHT = 800     # Window height
DRAW_LINES = True    # Show AI debug lines
```

### NEAT Parameters (`config-feedforward.txt`)
Adjust evolution settings:
- Population size
- Mutation rates
- Fitness thresholds
- Network topology constraints

### Physics Tweaks
Modify in respective class files:
- Bird jump strength
- Pipe gap size
- Movement speeds
- Gravity effects

## 📊 Training Process

1. **Generation 1**: Random behavior, most birds crash immediately
2. **Early Generations**: Gradual improvement in basic survival
3. **Mid Training**: Birds learn to time jumps with pipe openings
4. **Advanced Generations**: Sophisticated strategies emerge
5. **Mastery**: Consistent high scores and efficient navigation

## 🏆 Success Metrics

- **Survival Time**: How long birds stay alive
- **Pipes Passed**: Number of obstacles successfully navigated
- **Score Achievement**: Target score of 50+ triggers model saving
- **Generation Efficiency**: Fewer generations needed for improvement

## 📈 Output and Results

### Console Output
- Generation statistics
- Population fitness metrics
- Best performer scores
- Evolution progress tracking

### Saved Models
- Best genome saved as `best.pickle` when score > 50(you can play with if you want high)
- Can be loaded for future gameplay or analysis

## 🔧 Troubleshooting

**Game runs too fast/slow**: Adjust frame rate in main loop
**Poor AI performance**: Modify NEAT parameters or increase population size
**Display issues**: Check pygame installation and graphics drivers

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request



## 🙏 Acknowledgments

- NEAT algorithm by Kenneth Stanley
- Pygame community for game development tools
- Flappy Bird original concept by Dong Nguyen


**Happy Training!** 🐦🤖
