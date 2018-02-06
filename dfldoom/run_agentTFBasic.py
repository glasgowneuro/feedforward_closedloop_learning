from __future__ import print_function
import numpy as np
import cv2
import tensorflow as tf
import sys
import time
import os
sys.path.append('./agent')
sys.path.append('./deep_feedback_learning')
from agent.doom_simulator import DoomSimulator
from agent.agent import Agent
from deep_feedback_learning import DeepFeedbackLearning
import threading
from time import sleep
from matplotlib import pyplot as plt

width = 160
widthIn = 160
height = 120
heightIn = 120

preprocess_input_images = lambda x: x / 255. - 0.5

sharpen = np.array((
	[0, 1, 0],
	[1, 4, 1],
	[0, 1, 0]), dtype="int")

edge = np.array((
	[0, 1, 0],
	[1, -4, 1],
	[0, 1, 0]), dtype="int")

def getColourImbalance(img, colour):
    if(img.shape[0]) != 3:
        print("Error in getColourImbalance: wrong number of image channels: ", img.shape)
        return 0.

    width = int(img.shape[2]/2)
    height = int(img.shape[1]/2)
    print ("width: ", width, "height", height)
    avgLeft = np.average(img[:,:,:width], axis=1)
    avgLeft = np.average(avgLeft, axis=1)
#    avgLeft = np.dot(avgLeft, colour)
    avgRight = np.average(img[:,:,width:], axis=1)
    avgRight = np.average(avgRight, axis=1)
#    avgRight = np.dot(avgRight, colour)
    avgTop = np.average(img[:, :height, :], axis=1)
    avgTop = np.average(avgTop, axis=1)
    #    avgTop = np.dot(avgTop, colour)
    avgBottom = np.average(img[:, height:, :], axis=1)
    avgBottom = np.average(avgBottom, axis=1)
    #    avgBottom = np.dot(avgBottom, colour)
    print("avgLeft: ", avgLeft, " avgRight: ", avgRight, "avgTop", avgTop, "avgBottom", avgBottom)
    return 1.

def getMaxColourPos(img, colour):
    img = np.array(img, dtype='float64')
    width = int(img.shape[1])
    diff = np.ones(img.shape)
    diff[:,:,0] = colour[0]
    diff[:,:,1] = colour[1]
    diff[:,:,2] = colour[2]
    diff = np.absolute(np.add(diff, (-1*img)))
    cv2.imwrite("/home/paul/tmp/Images/Positive/diff-" + ".jpg", diff)
    diff = np.sum(diff, axis=2)
    cv2.imwrite("/home/paul/tmp/Images/Positive/diffGrey-" + ".jpg", diff)

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


def savePosImage(curr_step, centre, x1, y1, x2, y2, _img, myFile, width, height):
    print ("img shape: ", img2.shape)
    myFile.write("/home/paul/tmp/Images/" + str(curr_step) + ".jpg"
                 + " 1"
                 + " " + str(x1) + " " + str(y1)
                 + " " + str(x2) + " " + str(y2) + "\n")
    img = np.zeros(_img.shape,dtype=np.uint8)
    outImage = Image.fromarray(img)
    outImage.save("/home/paul/tmp/Images/Positive/" + str(curr_step) + ".jpg")

def saveNegImage(curr_step, img2, myFile, width, height):
    myFile.write("/home/paul/tmp/Images/" + str(curr_step) + ".jpg\n")
#    img2 = np.rollaxis(img2, 0, 3)
    img = Image.fromarray(img2)
    img.save("/home/paul/tmp/Images/Negative/" + str(curr_step) + ".jpg")
#    np.save('/home/paul/tmp/Images/Negative/' + str(curr_step), img2)

def main():
    ## Simulator
    simulator_args = {}
    simulator_args['config'] = 'config/config.cfg'
    simulator_args['resolution'] = (width,height)
    simulator_args['frame_skip'] = 1
    simulator_args['color_mode'] = 'RGB24'
    simulator_args['game_args'] = "+name ICO +colorset 7"

    ## Agent
    agent_args = {}

    # preprocessing
    preprocess_input_images = lambda x: x / 255. - 0.5
    agent_args['preprocess_input_images'] = lambda x: x / 255. - 0.5
    agent_args['preprocess_input_measurements'] = lambda x: x / 100. - 0.5
    agent_args['num_future_steps'] = 6
    pred_scale_coeffs = np.expand_dims(
        (np.expand_dims(np.array([8., 40., 1.]), 1) * np.ones((1, agent_args['num_future_steps']))).flatten(), 0)
    agent_args['meas_for_net_init'] = range(3)
    agent_args['meas_for_manual_init'] = range(3, 16)
    agent_args['resolution'] = (width,height)
    # just use grayscale for nnet inputs
    agent_args['num_channels'] = 1


    # net parameters
    agent_args['net_type'] = "fc"
    agent_args['conv_params'] = np.array([(16, 5, 4), (32, 3, 2), (64, 3, 2), (128, 3, 2)],
                                         dtype=[('out_channels', int), ('kernel', int), ('stride', int)])
    agent_args['fc_img_params'] = np.array([(128,)], dtype=[('out_dims', int)])
    agent_args['fc_meas_params'] = np.array([(128,), (128,), (128,)], dtype=[('out_dims', int)])
    agent_args['fc_joint_params'] = np.array([(256,), (256,), (-1,)], dtype=[('out_dims', int)])
    agent_args['target_dim'] = agent_args['num_future_steps'] * len(agent_args['meas_for_net_init'])
    agent_args['n_actions'] = 7

    # experiment arguments
    agent_args['test_objective_params'] = (np.array([5, 11, 17]), np.array([1., 1., 1.]))
    agent_args['history_length'] = 3
    agent_args['history_length_ico'] = 3
    historyLen = agent_args['history_length']
    print ("HistoryLen: ", historyLen)

    print('starting simulator')
    simulator = DoomSimulator(simulator_args)
    num_channels = simulator.num_channels

    print('started simulator')

    agent_args['state_imgs_shape'] = (
    historyLen * num_channels, simulator.resolution[1], simulator.resolution[0])

    agent_args['n_ffnet_input'] = (agent_args['resolution'][0]*agent_args['resolution'][1])
    agent_args['n_ffnet_hidden'] = np.array([50,5])
    agent_args['n_ffnet_output'] = 1
    agent_args['n_ffnet_act'] = 7
    agent_args['n_ffnet_meas'] = simulator.num_meas
    agent_args['learning_rate'] = 1E-3

    modelDir = os.path.join(os.path.expanduser("~"), "Dev/GameAI/vizdoom_cig2017/icodoom/ICO1/Models")

    if 'meas_for_net_init' in agent_args:
        agent_args['meas_for_net'] = []
        for ns in range(historyLen):
            agent_args['meas_for_net'] += [i + simulator.num_meas * ns for i in agent_args['meas_for_net_init']]
        agent_args['meas_for_net'] = np.array(agent_args['meas_for_net'])
    else:
        agent_args['meas_for_net'] = np.arange(historyLen * simulator.num_meas)
    if len(agent_args['meas_for_manual_init']) > 0:
        agent_args['meas_for_manual'] = np.array([i + simulator.num_meas * (historyLen - 1) for i in
                                                  agent_args[
                                                      'meas_for_manual_init']])  # current timestep is the last in the stack
    else:
        agent_args['meas_for_manual'] = []

    agent_args['state_meas_shape'] = (len(agent_args['meas_for_net']),)

#    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
#    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))

#    agent = Agent(sess, agent_args)
#    agent.load('/home/paul/Dev/GameAI/vizdoom_cig2017/icolearner/ICO1/checkpoints/ICO-8600')
#    print("model loaded..")

    #    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
#    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))

    img_buffer = np.zeros(
        (historyLen, simulator.resolution[1], simulator.resolution[0], num_channels), dtype='uint8')

    meas_buffer = np.zeros((historyLen, simulator.num_meas))
    act_buffer = np.zeros((historyLen, 7))
    act_buffer_ico = np.zeros((agent_args['history_length_ico'], 7))
    curr_step = 0
    old_step = -1
    term = False

    print ("state_meas_shape: ", meas_buffer.shape, " == ", agent_args['state_meas_shape'])
    print ("act_buffer_shape: ", act_buffer.shape)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options,log_device_placement=False))
    ag = Agent(sess, agent_args)

    if (os.path.isfile("checkpoints1/checkpoint")):
        ag.load('/home/paul/Dev/GameAI/vizdoom_cig2017/icodoom/ICO1/checkpoints1/')
        print("model loaded..")
    else:
        print ("No model file, initialising...")


    diff_y = 0
    diff_x = 0
    diff_z = 0
    diff_theta = 0
    epoch = 200
    radialFlowLeft = 30.
    radialFlowRight = 30.
    radialFlowInertia = 0.4
    radialGain = 4.
    rotationGain = 50.
    errorThresh = 10.
    updatePtsFreq = 50
    skipImage = 1
    skipImageICO = 5
    reflexGain = 0.1
    netGain = 0. #10.
    oldHealth = 0.

    # create masks for left and right visual fields - note that these only cover the upper half of the image
    # this is to help prevent the tracking getting confused by the floor pattern
    half_height = round(height/2)
    half_width = round(width/2)

    maskLeft = np.zeros([height, width], np.uint8)
    maskLeft[half_height:, :half_width] = 1.
    maskRight = np.zeros([height, width], np.uint8)
    maskRight[half_height:, half_width:] = 1.

    lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    feature_params = dict(maxCorners=500, qualityLevel=0.03, minDistance=7, blockSize=7)

    imgCentre = np.array([int(simulator_args['resolution'][0] / 2), int(simulator_args['resolution'][1] /2)])
    print ("Image centre: ", imgCentre)
    simpleInputs = np.zeros((width, height))
    input_buff = np.zeros((1,width*height))
    target_buff = np.zeros((1,1))
    meas_buff = np.zeros((1,simulator.num_meas))
    netOut = 0.
    netErr = np.zeros((width,height))
    delta = 0.
    delta2 = 0
    dontshoot = 1
    deltaZeroCtr = 1
    curr_act = np.zeros(7).tolist()

    reflexOn = False
    iter = 0
    episodes = 1000
    simulator._game.init()

    for i in range(episodes):
#        print ("Episode ", i)
        tc = 0
        simulator._game.new_episode()

        while (tc < 500):
            screen_buf, meas, rwrd, term = simulator.step(curr_act)
            if (screen_buf is None):
                break

            midlinex = int(width / 2)
            midliney = int(height * 0.75)
            crcb = screen_buf
            screen_left = screen_buf[0:midliney, 0:midlinex, 2]
            screen_right = screen_buf[0:midliney, midlinex:width, 2]
            screen_left = cv2.filter2D(screen_left, -1, sharpen)
            screen_right = cv2.filter2D(screen_right, -1, sharpen)
            simpleInputs = preprocess_input_images(np.array(np.sum(crcb, axis=2) / 3))

#            simpleInputs = cv2.filter2D(simpleInputs, -1, edge)
#            simpleInputs = simpleInputs - np.mean(simpleInputs)
            screen_diff = screen_left - np.fliplr(screen_right)
            screen_diff = cv2.resize(screen_diff, (width, height))
            # cv2.imwrite('/tmp/left.png',screen_left)
            # cv2.imwrite('/tmp/right.png',screen_right)
#            cv2.imwrite("/home/paul/tmp/Images/diff-" + str(iter) + ".png", screen_diff)
#            cv2.imwrite("/home/paul/tmp/Images/raw-" + str(iter) + ".png", crcb)


            lavg = np.average(screen_left)
            ravg = np.average(screen_right)

            shoot = 0
            if (dontshoot > 1):
                dontshoot = dontshoot - 1
            else:
                if (tc > 30):
                    shoot = 1
                    dontshoot = 5

            centre, bottomLeft, topRight, colourStrength = getMaxColourPos(crcb, [255, 0, 0])
            colourSteer = imgCentre[0]
            imgRect = np.zeros(crcb.shape)

            if (len(bottomLeft) > 0 and len(topRight) > 0 and ((topRight[0] - bottomLeft[0]) < width / 3) and (
                (topRight[1] - bottomLeft[1]) < height / 2)):
                colourSteer = bottomLeft[0] + int(0.5 * (topRight[0] - bottomLeft[0]))

            cv2.arrowedLine(imgRect, (colourSteer, imgCentre[1] + 10), (colourSteer, imgCentre[1]),
                            color=(255, 255, 255), thickness=2)
#            cv2.imwrite("/home/paul/tmp/Images/simple-" + str(iter) + ".jpg", simpleInputs)
#            cv2.imwrite("/home/paul/tmp/Images/rect-" + str(iter) + ".jpg", imgRect)
#            cv2.imwrite("/home/paul/tmp/Images/" + str(iter) + ".jpg", crcb)

            #            cv2.imwrite("/home/paul/tmp/Images/Positive/arrow-" + str(iter) + ".jpg", imgRect)
#            cv2.imwrite("/home/paul/tmp/Images/Positive/" + str(iter) + ".jpg", crcb)

#            blue = cv2.filter2D(blue, -1, edge)
#            cv2.imwrite("/home/paul/tmp/Images/Positive/blue-" + str(curr_step) + ".jpg", blue)
            meas_buff[0,:] = meas
            imgRect = np.array(np.sum(imgRect, axis=2) / 3)
#            input_buff[0,:] = np.ndarray.flatten(imgRect)
            input_buff[0,:] = np.ndarray.flatten(imgRect)



#            input_buff[0,:] = np.random.normal(0.0, 0.01, size=width*height)
#            print("mean: ", np.mean(input_buff[0,:]), " var: ", np.var(input_buff[0,:]))

            if (tc > 2):
                delta = (float(colourSteer) - float(imgCentre[0])) / float(width)

            else:
                delta = 0

            target_buff[...] = delta + netOut
            target_buff[...] = -0.5

            ag.act_fcnet(input_buff, meas, target_buff)
            netOut = np.ndarray.flatten(ag.ext_fcnet_output)[0].flatten()[0]

            netErr[:,:] = 0.
            diff_theta = reflexGain * delta
#            print(tc, diff_theta, netGain*netOut, target_buff[0,0], delta)
            print(tc, reflexGain * delta, netOut)

            curr_act = np.zeros(7).tolist()
            curr_act[0] = 0
            curr_act[1] = 0
            curr_act[2] = 1
            curr_act[3] = 0. #curr_act[3] + diff_z
            curr_act[3] = 0.
            curr_act[4] = 0
            curr_act[5] = 0
            curr_act[6] = diff_theta + netGain*netOut


            iter += 1

            if (curr_step % epoch == 0):
                ag.save('/home/paul/Dev/GameAI/vizdoom_cig2017/icodoom/ICO1/checkpoints1/BPBasic', curr_step)
            curr_step += 1
            # 30 fps
#            time.sleep(0.03)

            tc += 1

    simulator.close_game()


if __name__ == '__main__':
    main()
