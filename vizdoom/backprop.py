#!/usr/bin/python3

from __future__ import print_function
from vizdoom import *
import sys
import os
import threading
import math

from random import choice
from time import sleep
from matplotlib import pyplot as plt

sys.path.append('../../deep_feedback_learning')

import numpy as np
import cv2
import deep_feedback_learning

# Create DoomGame instance. It will run the game and communicate with you.
game = DoomGame()

# Now it's time for configuration!
# load_config could be used to load configuration instead of doing it here with code.
# If load_config is used in-code configuration will also work - most recent changes will add to previous ones.
# game.load_config("../../scenarios/basic.cfg")

# Sets path to additional resources wad file which is basically your scenario wad.
# If not specified default maps will be used and it's pretty much useless... unless you want to play good old Doom.
game.set_doom_scenario_path("./basic.wad")

# Sets map to start (scenario .wad files can contain many maps).
game.set_doom_map("map01")

# Sets resolution. Default is 320X240
game.set_screen_resolution(ScreenResolution.RES_160X120)

# create masks for left and right visual fields - note that these only cover the upper half of the image
# this is to help prevent the tracking getting confused by the floor pattern
width = 160
widthNet = 160
height = 120
heightNet = 120
preprocess_input_images = lambda x: x / 255. - 0.5
imgCentre = np.array([int(width / 2), int(height / 2)])

# Sets the screen buffer format. Not used here but now you can change it. Defalut is CRCGCB.
game.set_screen_format(ScreenFormat.RGB24)

# Enables depth buffer.
game.set_depth_buffer_enabled(True)

# Enables labeling of in game objects labeling.
game.set_labels_buffer_enabled(True)

# Enables buffer with top down map of the current episode/level.
game.set_automap_buffer_enabled(True)

# Sets other rendering options
game.set_render_hud(False)
game.set_render_minimal_hud(False)  # If hud is enabled
game.set_render_crosshair(True)
game.set_render_weapon(False)
game.set_render_decals(False)
game.set_render_particles(False)
game.set_render_effects_sprites(False)
game.set_render_messages(False)
game.set_render_corpses(False)

# Adds buttons that will be allowed. 
# game.add_available_button(Button.MOVE_LEFT)
# game.add_available_button(Button.MOVE_RIGHT)
game.add_available_button(Button.MOVE_LEFT_RIGHT_DELTA, 50)
game.add_available_button(Button.ATTACK)
game.add_available_button(Button.TURN_LEFT_RIGHT_DELTA)

# Adds game variables that will be included in state.
game.add_available_game_variable(GameVariable.AMMO2)

# Causes episodes to finish after 200 tics (actions)
game.set_episode_timeout(500)

# Makes episodes start after 10 tics (~after raising the weapon)
game.set_episode_start_time(10)

# Makes the window appear (turned on by default)
game.set_window_visible(True)

# Turns on the sound. (turned off by default)
game.set_sound_enabled(True)

# Sets the livin reward (for each move) to -1
game.set_living_reward(-1)

# Sets ViZDoom mode (PLAYER, ASYNC_PLAYER, SPECTATOR, ASYNC_SPECTATOR, PLAYER mode is default)
game.set_mode(Mode.PLAYER)

# Enables engine output to console.
#game.set_console_enabled(True)

nFiltersInput = 0
nFiltersHidden = 0
minT = 2
maxT = 10
nHidden0 = 2
nHidden1 = 2

learningRate = 0
#0.001

net = deep_feedback_learning.DeepFeedbackLearning(widthNet*heightNet,[nHidden0*nHidden0, nHidden1*nHidden1], 1, nFiltersInput, nFiltersHidden, minT,maxT)
net.getLayer(0).setConvolution(widthNet,heightNet)
net.getLayer(1).setConvolution(nHidden0,nHidden0)
net.getLayer(2).setConvolution(nHidden1,nHidden1)
net.initWeights(1,0,deep_feedback_learning.Neuron.MAX_OUTPUT_RANDOM)
net.setAlgorithm(deep_feedback_learning.DeepFeedbackLearning.backprop)
net.setLearningRate(learningRate)
net.setUseDerivative(0)
net.setMomentum(0.5)
net.setBias(0)
net.setLearningRateDiscountFactor(1)
net.getLayer(0).setActivationFunction(deep_feedback_learning.Neuron.TANH)
net.getLayer(1).setActivationFunction(deep_feedback_learning.Neuron.TANH)
epoch = 200
reflexGain = 1.0
netGain = 10.

# Initialize the game. Further configuration won't take any effect from now on.
game.init()

# Run this many episodes
episodes = 1000

# Sets time that will pause the engine after each action (in seconds)
# Without this everything would go too fast for you to keep track of what's happening.
sleep_time = 1.0 / DEFAULT_TICRATE # = 0.028

delta2 = 0
dontshoot = 1

inp = np.zeros(widthNet*heightNet)

sharpen = np.array((
	[0, 1, 0],
	[1, 4, 1],
	[0, 1, 0]), dtype="int")

edge = np.array((
	[0, 1, 0],
	[1, -4, 1],
	[0, 1, 0]), dtype="int")






plt.ion()
plt.show()
ln1 = False
ln2 = [False,False,False,False]


def getMaxColourPos(img, colour):
    img = np.array(img, dtype='float64')
    width = int(img.shape[1])
    diff = np.ones(img.shape)
    diff[:,:,0] = colour[0]
    diff[:,:,1] = colour[1]
    diff[:,:,2] = colour[2]
    diff = np.absolute(np.add(diff, (-1*img)))
    diff = np.sum(diff, axis=2)

    indx = np.argmin(diff)
    indx0 = int(indx / width)
    indx1 = indx % width
    pts = np.asarray(np.where((np.mean(diff) - diff) > 150))
    if (pts.shape[1]>0):
        bottomLeft = np.array([np.amin(pts[1]), np.amin(pts[0])])
        topRight = np.array([np.amax(pts[1]), np.amax(pts[0])])
    else:
        bottomLeft = []
        topRight = []

    return np.array([indx1, indx0]), bottomLeft, topRight, np.mean(diff) - diff[indx0,indx1]


def getWeights2D(neuron):
    n_neurons = net.getLayer(0).getNneurons()
    n_inputs = net.getLayer(0).getNeuron(neuron).getNinputs()
    weights = np.zeros(n_inputs)
    for i in range(n_inputs):
        if net.getLayer(0).getNeuron(neuron).getMask(i):
            weights[i] = net.getLayer(0).getNeuron(neuron).getAvgWeight(i)
        else:
            weights[i] = np.nan
    return weights.reshape(heightNet,widthNet)

def getWeights1D(layer,neuron):
    n_neurons = net.getLayer(layer).getNneurons()
    n_inputs = net.getLayer(layer).getNeuron(neuron).getNinputs()
    weights = np.zeros(n_inputs)
    for i in range(n_inputs):
        weights[i] = net.getLayer(layer).getNeuron(neuron).getAvgWeight(i)
    return weights

def plotWeights():
    global ln1
    global ln2

    while True:

        if ln1:
            ln1.remove()
        plt.figure(1)
        w1 = getWeights2D(0)
        for i in range(1,net.getLayer(0).getNneurons()):
            w2 = getWeights2D(i)
            w1 = np.where(np.isnan(w2),w1,w2)
        ln1 = plt.imshow(w1,cmap='gray')
        plt.draw()
        plt.pause(0.1)

        for j in range(1,net.getNumHidLayers()+1):
            if ln2[j]:
                ln2[j].remove()
            plt.figure(j+1)
            w1 = np.zeros( (net.getLayer(j).getNneurons(),net.getLayer(j).getNeuron(0).getNinputs()) )
            for i in range(net.getLayer(j).getNneurons()):
                w1[i,:] = getWeights1D(j,i)
            ln2[j] = plt.imshow(w1,cmap='gray')
            plt.draw()
            plt.pause(1.0)


t1 = threading.Thread(target=plotWeights)
t1.start()

try:
    checkpointFile = open("Models/checkpoint")
    try:
        modelName = checkpointFile.read().splitlines()
        if(net.loadModel(modelName[0])):
            print("loaded from Model file: ", modelName[0])
        else:
            print("FAILED loading from Model file: ", modelName[0])
    except:
        print ("Checkpoint file contains no valid model")
    finally:
        checkpointFile.close()
except Exception:
    print ("No checkpoint found...")

imgRect = np.zeros((width,height,3))
imgRect1 = np.zeros((widthNet,heightNet))
simpleInputs = np.zeros((width, height))

for i in range(episodes):

    # Starts a new episode. It is not needed right after init() but it doesn't cost much. At least the loop is nicer.
    game.new_episode()

    tc = 0
    output = 0
    while not game.is_episode_finished():

        # Gets the state
        state = game.get_state()

        # Which consists of:
        n = state.number
        vars = state.game_variables
        screen_buf = state.screen_buffer
        depth_buf = state.depth_buffer
        labels_buf = state.labels_buffer
        automap_buf = state.automap_buffer
        labels = state.labels

        midlinex = int(width / 2)
        midliney = int(height * 0.75)
        crcb = screen_buf

        simpleInputs = np.array(np.sum(crcb, axis=2) / 3)

        shoot = 0
        if (dontshoot > 1):
            dontshoot = dontshoot - 1
        else:
            if (tc > 30):
                shoot = 1
                dontshoot = 5

        centre, bottomLeft, topRight, colourStrength = getMaxColourPos(crcb, [255, 0, 0])
        colourSteer = imgCentre[0]
        imgRect[...] = 0.

        if (len(bottomLeft) > 0 and len(topRight) > 0 and ((topRight[0] - bottomLeft[0]) < width / 3) and (
                    (topRight[1] - bottomLeft[1]) < height / 2)):
            colourSteer = bottomLeft[0] + int(0.5 * (topRight[0] - bottomLeft[0]))
        cv2.arrowedLine(imgRect, (colourSteer, imgCentre[1] + 10), (colourSteer, imgCentre[1]),
                        color=(255, 255, 255), thickness=2)
        imgRect1 = cv2.resize(np.array(np.sum(imgRect, axis=2)), (widthNet, heightNet))

        if (tc > 2):
            delta = (float(colourSteer) - float(imgCentre[0])) / float(width)
            net.setLearningRate(1E-3)

        else:
            delta = 0
            net.setLearningRate(0.)

        err = np.ones(nHidden0*nHidden0)*(delta)
        net.doStep(imgRect1.flatten(),err[:1])

        output = net.getOutput(0)
        print(n,delta,output)

        diff_theta = reflexGain * delta

        action = [ 0., shoot, diff_theta + netGain*output]
        r = game.make_action(action)
        tc = tc + 1

#        if sleep_time > 0:
#            sleep(sleep_time)

    # Check how the episode went.
    tc = 0

    #30 fps
    sleep(.03)

    if not os.path.exists("Models"):
        os.makedirs("Models")
    net.saveModel("Models/BP-" + str(i) + ".txt")

    file = open("Models/checkpoint", 'w')
    file.write("Models/BP-" + str(i) + ".txt")
    file.close()

# It will be done automatically anyway but sometimes you need to do it in the middle of the program...
game.close()
