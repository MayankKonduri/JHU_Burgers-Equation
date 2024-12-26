# Physics-Informed Neural Networks (PINNs) for Shock Waves in Burgers' Equation

## Project Overview
This repository implements **Physics-Informed Neural Networks (PINNs)** to solve Burgers' equation, capturing nonlinear shock wave phenomena. Unlike traditional numerical methods, PINNs use deep learning to embed physical constraints directly into the loss function, enabling accurate solutions for challenging scenarios like steep gradients and discontinuities. This project focuses on efficient and precise solutions to Burgers' equation with potential applications in fluid dynamics and related fields.

## Features
- **PINN Architecture**: Implements a fully connected neural network with physics constraints.
- **Custom Loss Function**: Combines boundary conditions and Burgers' equation residuals.
- **Efficient Training**: Optimized using Adam with a smooth loss curve for convergence.
- **Visualization**: Generates contour plots to visualize shock wave propagation over time.

## Citation
Samaniego, E., Anitescu, C., Goswami, S., et al. (2020). "An energy approach to the solution of partial differential equations in computational mechanics via machine learning: Concepts, implementation and applications." *Computer Methods in Applied Mechanics and Engineering*, 362.

## Requirements
- Python 3.x
- TensorFlow
- NumPy
- Matplotlib

## Usage
1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/Burgers-PINNs.git
