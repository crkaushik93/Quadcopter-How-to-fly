# Quadcopter-How-to-fly

In this project, you will design an agent that can fly a quadcopter, and then train it using a reinforcement learning algorithm of your choice! 
Try to apply the techniques you have learnt in this module to find out what works best, but also feel free to come up with innovative ideas and test them.

We'll use a specific noise process that has some desired properties, called the Ornstein–Uhlenbeck process. 
It essentially generates random samples from a Gaussian (Normal) distribution, but each sample affects the next one such that two consecutive samples 
are more likely to be closer together than further apart. 
In this sense, the process in Markovian in nature.

Remember that we want to use this process to add some noise to our actions, in order to encourage exploratory behavior. And since our actions translate to force and 
torque being applied to a quadcopter, we want consecutive actions to not vary wildly. Otherwise, we may not actually get anywhere! Imagine flicking a controller up-down, 
left-right randomly!

Besides the temporally correlated nature of samples, the other nice thing about the OU process is that it tends to settle down close to the specified mean over time. 
When used to generate noise, we can specify a mean of zero, and that will have the effect of reducing exploration as we make progress on learning the task.

import numpy as np
import copy

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, mu, theta, sigma):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state
        
 
 
# Project Overview:
The Quadcopter or Quadrotor Helicopter is becoming an increasingly popular aircraft for both personal and professional use. 
Its maneuverability lends itself to many applications, from last-mile delivery to cinematography, from acrobatics to search-and-rescue.

Most quadcopters have 4 motors to provide thrust, although some other models with 6 or 8 motors are also sometimes referred to as quadcopters. 
Multiple points of thrust with the center of gravity in the middle improves stability and enables a variety of flying behaviors.

But it also comes at a price–the high complexity of controlling such an aircraft makes it almost impossible to manually control each individual motor's thrust. 
So, most commercial quadcopters try to simplify the flying controls by accepting a single thrust magnitude and yaw/pitch/roll controls, making it much more intuitive and fun.

The next step in this evolution is to enable quadcopters to autonomously achieve desired control behaviors such as takeoff and landing. You could design these controls with a classic approach (say, by implementing PID controllers). Or, you can use reinforcement learning to build agents that can learn these behaviors on their own. 
This is what you are going to do in this project!

