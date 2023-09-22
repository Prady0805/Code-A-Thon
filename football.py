# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 08:41:57 2023

@author: Pradyumna
"""

import pygame as pg
import numpy as np
import sys

def ball_animation():
    
    global ball_speed_x, ball_speed_y, opponent_score, opp_misses
     # Update the game state
    ball.x += ball_speed_x
    ball.y += ball_speed_y

    # Handle ball collisions
    if ball.top <= 0 or ball.bottom >= screen_height:
        ball_speed_y *= -1

    if ball.left <= 0 or ball.right >= screen_width:
        ball_speed_x *= -1

    if ball.colliderect(opponent):
        ball_speed_x *= -1    
               
	# Opponent Score
    if ball.colliderect(player):
        ball_restart()
        opponent_score += 1
        
    #MISSES
    if ball.left <= 0:
        ball_restart()
        opp_misses += 1


def player_animation():
    
    player.y += player_speed
    
    if player.top <= 0:
        player.top = 0
    if player.bottom >= screen_height:
        player.bottom = screen_height

#OPTIONAL        
def opp_ai():
    
    if opponent.top < ball.y:
        opponent.y += opponent_speed
    if opponent.bottom > ball.y:
        opponent.y -= opponent_speed

    if opponent.top <= 0:
        opponent.top = 0
    if opponent.bottom >= screen_height:
        opponent.bottom = screen_height

def ball_restart():
    
    global ball_speed_x, ball_speed_y
    
    ball.center = (screen_width/2, screen_height/2)
    ball_speed_x *= np.random.choice((1,-1))
    ball_speed_y *= np.random.choice((1,-1))

#SETUP!
#main screen
pg.init();
clock = pg.time.Clock();

#creating a screen with height and width
screen_width = 1300;
screen_height = 800;
screen = pg.display.set_mode((screen_width, screen_height));
pg.display.set_caption("FOOTBALL PONG");
background_image = pg.image.load('futbol.jpg')
background_image = pg.transform.scale(background_image, (screen_width, screen_height))

#COLORS
light_grey = (200,200,200)
red = (255, 0, 0)
blue = (0, 0, 255)
black = (0,0,0)
bg_color = pg.Color('grey12')

#Game Rectangles
ball = pg.Rect(screen_width / 2 - 15, screen_height / 2 - 15, 30, 30)
player = pg.Rect(screen_width - 10, screen_height / 2 - 125, 10,250)
opponent = pg.Rect(10, screen_height / 2 - 70, 30,120)

#SPEED OF BALL
ball_speed_x = 30 * np.random.choice((1,-1));
ball_speed_y = 30 * np.random.choice((1,-1));
player_speed = 0;
opponent_speed = 30 

#SCORE
player_score = 0
opponent_score = 0
opp_misses = 0
basic_font = pg.font.Font('freesansbold.ttf', 32)

# Q-learning parameters
learning_rate = 0.1
discount_factor = 0.9
epsilon = 0.1

# Reward for hitting the ball and scoring a point
reward_hit = 10
reward_scored = 100

# Q-table initialization
num_states = 4  # Number of states (positions: ball y, ball x, player y, player x)
num_actions = 2  # Number of actions (up, down)

# Q-table initialization with appropriate dimensions
num_bins = 10
num_states = num_bins * num_bins * num_bins * num_bins
q_table = np.zeros((num_states, num_actions))

# Function to discretize the state based on ball and player positions
def discretize_state(ball_y, ball_x, player_y, player_x):
    # Discretize the continuous state into a discrete state
    # For simplicity, we'll just divide each dimension into a few discrete bins

    # Convert the coordinates to integers
    ball_y = int(ball_y)
    ball_x = int(ball_x)
    player_y = int(player_y)
    player_x = int(player_x)

    bins = [np.linspace(0, screen_height, num_bins + 1).astype(int),
            np.linspace(0, screen_width, num_bins + 1).astype(int),
            np.linspace(0, screen_height, num_bins + 1).astype(int),
            np.linspace(0, screen_height, num_bins + 1).astype(int)]

    # Use np.digitize with right=True to avoid the "object too deep" error
    state = [np.digitize([ball_y], bins[0], right=True)[0],
             np.digitize([ball_x], bins[1], right=True)[0],
             np.digitize([player_y], bins[2], right=True)[0],
             np.digitize([player_x], bins[3], right=True)[0]]

    return state


# Function to hash a state-action pair to an index in the Q-table
def hash_state_action(state, action):
    # Ensure state indices are within the valid range
    state_indices = np.clip(state, 0, num_bins - 1)

    # Calculate the index using ravel_multi_index
    return np.ravel_multi_index(state_indices, (num_bins, num_bins, num_bins, num_bins)) * num_actions + action


# Function to get the discrete state index
def get_state_index(state):
    # Calculate the index of the state in the Q-table
    return state[0] * num_bins * num_bins * num_bins + \
           state[1] * num_bins * num_bins + \
           state[2] * num_bins + \
           state[3]


# Function to update opponent's position based on ball position
def update_opponent_position():
    if opponent.centery < ball.centery:
        opponent.y += opponent_speed
    else:
        opponent.y -= opponent_speed

# ... Rest of the existing game loop code ...

# Inside the main loop
while True:
    for event in pg.event.get():
        if event.type == pg.QUIT:
            pg.quit()
            sys.exit()

    # Discretize the current state
    current_state = discretize_state(ball.y, ball.x, player.y, player.x)
    current_state_index = hash_state_action(current_state, 0) 

    # Update opponent's position
    update_opponent_position()

    # Choose an action using epsilon-greedy policy
    if np.random.uniform(0, 1) < epsilon:
        action = np.random.choice(num_actions)  # Random action
    else:
        action = np.argmax(q_table[current_state, :])  # Greedy action

    # Apply the action
    if action == 0:
        player.y -= player_speed
    elif action == 1:
        player.y += player_speed

   
    if ball.colliderect(player):
        reward = reward_scored
    elif ball.colliderect(opponent):
        reward = reward_hit
    elif ball.left <= 0:
        reward = 0
    else:
        reward = 0

    # Update Q-value using Q-learning
    new_state = discretize_state(ball.y, ball.x, player.y, player.x)
    new_state_index = hash_state_action(new_state, action)

    #new_state_index = min(new_state_index, num_states - 1)
    #if new_state_index >= num_states:
        # If the new_state_index exceeds the Q-table size, clip it to the maximum index
     #   new_state_index = num_states - 1
     
    if 0 <= new_state_index < num_states:
        q_table[current_state_index, action] += learning_rate * \
            (reward + discount_factor * np.max(q_table[new_state_index]) - q_table[current_state_index, action])

    #GAME LOGIC
    ball_animation()
    player_animation()
    
        
    screen.fill(bg_color)
    screen.blit(background_image, (0, 0))
    pg.draw.rect(screen, black, player)
    pg.draw.rect(screen, red, opponent)
    pg.draw.ellipse(screen, light_grey, ball)
    pg.draw.aaline(screen, blue, (screen_width / 2, 0),(screen_width / 2, screen_height))
            
    opponent_text = basic_font.render(f'SCORE: {opponent_score}',False,light_grey)
    misses_text = basic_font.render(f'MISSES: {opp_misses}',False,light_grey)
    screen.blit(opponent_text,(800,50))
    screen.blit(misses_text, (800, 90))
    
    
    pg.display.flip();
    clock.tick(60);

