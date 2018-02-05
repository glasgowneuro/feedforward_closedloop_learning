from __future__ import print_function
import numpy as np
import cv2
import tensorflow as tf
import sys
import os
sys.path.append('./agent')
sys.path.append('./deep_feedback_learning')
from agent.doom_simulator import DoomSimulator
from agent.agent import Agent

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
    height = int(img.shape[0])
#    img[:,10,10] = [0,0,255]
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
#    print("COLOUR: ", [indx1, indx0])

#    cv2.imwrite("/home/paul/tmp/Images/Positive/rect-" + ".jpg", img)

#    print ("Colour diff: ", np.mean(diff) - diff[indx0,indx1])
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
    simulator_args['resolution'] = (widthIn,heightIn)
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
#    agent_args['net_type'] = "conv"
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
    agent_args['learning_rate'] = 1E-4

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

    if (os.path.isfile("checkpoints/checkpoint")):
        ag.load('/home/paul/Dev/GameAI/vizdoom_cig2017/icodoom/ICO1/checkpoints/')
        print("model loaded..")
    else:
        print ("No model file, initialising...")


    diff_y = 0
    diff_x = 0
    diff_z = 0
    diff_theta = 0
    iter = 1
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
    reflexGain = 1E-4
    flowGain = 0.
    netGain = 10.
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
    rawInputs = np.zeros((height, width))
    cheatInputs = np.zeros((width, height))
    input_buff = np.zeros((1,width*height))
    target_buff = np.zeros((1,1))
    meas_buff = np.zeros((1,simulator.num_meas))
    netOut = 0.
    netErr = np.zeros((width,height))
    delta = 0.
    shoot = 0

    reflexOn = False
    iter = 0

    while not term:
        if curr_step < historyLen:
            curr_act = np.zeros(7).tolist()
            img, meas, rwrd, term = simulator.step(curr_act)
            print("Image: ", img.shape, " max: ", np.amax(img), " min: ", np.amin(img))

            if curr_step == 0:
                p0Left = cv2.goodFeaturesToTrack(img[:,:,0], mask=maskLeft, **feature_params)
                p0Right = cv2.goodFeaturesToTrack(img[:,:,0], mask=maskRight, **feature_params)

            img_buffer[curr_step % historyLen] = img
            meas_buffer[curr_step % historyLen] = meas
            act_buffer[curr_step % historyLen] = curr_act[:7]

        else:
            img1 = img_buffer[(curr_step-2) % historyLen,:,:,:]
            img2 = img_buffer[(curr_step-1) % historyLen,:,:,:]
            state = simulator._game.get_state()

            stateImg = state.screen_buffer

            if(curr_step % updatePtsFreq == 0):
                p0Left = cv2.goodFeaturesToTrack(img[:,:,0], mask=maskLeft, **feature_params)
                p0Right = cv2.goodFeaturesToTrack(img[:,:,0], mask=maskRight, **feature_params)

            p1Left, st, err = cv2.calcOpticalFlowPyrLK(img1[:,:,0], img2[:,:,0], p0Left, None, **lk_params)
            p1Right, st, err = cv2.calcOpticalFlowPyrLK(img1[:,:,0], img2[:,:,0], p0Right, None, **lk_params)
            flowLeft = (p1Left - p0Left)[:,0,:]
            flowRight = (p1Right - p0Right)[:,0,:]
            radialFlowTmpLeft = 0
            radialFlowTmpRight = 0

            for i in range(0, len(p0Left)):
                radialFlowTmpLeft += ((p0Left[i,0,:] - imgCentre)).dot(flowLeft[i,:]) / float(len(p0Left))
            for i in range(0, len(p0Right)):
                radialFlowTmpRight += ((p0Right[i,0,:] - imgCentre)).dot(flowRight[i,:]) / float(len(p0Right))

            rotation = act_buffer[(curr_step - 1) % historyLen][6]
            forward = act_buffer[(curr_step - 1) % historyLen][3]
            # keep separate radial errors for left and right fields
            radialFlowLeft = radialFlowLeft + radialFlowInertia * (radialFlowTmpLeft - radialFlowLeft)
            radialFlowRight = radialFlowRight + radialFlowInertia * (radialFlowTmpRight - radialFlowRight)
            expectFlowLeft = radialGain * forward + (rotationGain * rotation if rotation < 0. else 0.)
            expectFlowRight = radialGain * forward - (rotationGain * rotation if rotation > 0. else 0.)

            flowErrorLeft = forward * (expectFlowLeft - radialFlowLeft) / (1. + rotationGain * np.abs(rotation))
            flowErrorRight = forward * (expectFlowRight - radialFlowRight) / (1. + rotationGain * np.abs(rotation))
            flowErrorLeft = flowErrorLeft if flowErrorLeft > 0. else 0.
            flowErrorRight = flowErrorRight if flowErrorRight > 0. else 0.
            icoSteer = 0.

            if curr_step > 100:
                health = meas[1]

                if (health<0.1):
                    reflexOn = False
                    iter = 0


                # Don't run any networks when the player is dead!
                if (health < 101. and health > 0.):

                    icoInSteer = flowGain * ((flowErrorRight - errorThresh) if (flowErrorRight - errorThresh) > 0. else 0. -
                    flowGain * (flowErrorLeft - errorThresh) if (flowErrorLeft - errorThresh) > 0. else 0. )

                    centre, bottomLeft, topRight, colourStrength = getMaxColourPos(stateImg, [255, 0, 0])
                    colourSteer = imgCentre[0]
                    cheatInputs = stateImg*1.

                    if(len(bottomLeft)>0 and len(topRight)>0 and ((topRight[0] - bottomLeft[0]) < width/3) and ((topRight[1] - bottomLeft[1]) < height/2)):
                        colourSteer = bottomLeft[0] + int(0.5 * (topRight[0] - bottomLeft[0]))
#                        cv2.imwrite("/home/paul/tmp/Backup/rect-" + str(curr_step) + ".jpg", cheatInputs)

                    cv2.arrowedLine(cheatInputs, (colourSteer, imgCentre[1]+10), (colourSteer, imgCentre[1]), color=(255,255,255), thickness=2)

                    rawInputs = np.array(np.sum(stateImg, axis=2) / 3)
                    cheatInputs = np.array(np.sum(cheatInputs, axis=2) / 3)
#                    cv2.imwrite("/home/paul/tmp/Backup/cheat-" + str(curr_step) + ".jpg", cheatInputs)

                    input_buff[0,:] = np.ndarray.flatten(cheatInputs)
                    input_buff = input_buff - np.mean(input_buff)
                    input_buff = input_buff / np.sqrt(np.var(input_buff))

                    # we want the reflex to be delayed wrt to the image input, so that the image is. Otherwise the learning can
                    # never reduce the error to zero no matter how good the controller.
                    if (iter>2):
                        delta = (float(colourSteer) - float(imgCentre[0]))/float(width)
                    else:
                        delta = 0

                    if(iter>2):
                        if(np.abs(delta) < 0.01):
                            shoot = 1

                    target_buff[...] = delta + netOut
#                    target_buff[...] = delta

#                    target_buff[...] = 0.2
                    meas_buff[0,:] = meas

                    ag.act(input_buff, meas, target_buff)
                    if(ag.net_type == 'conv'):
                        netOut = np.ndarray.flatten(ag.ext_covnet_output)[0].flatten()[0]
                    elif(ag.net_type == 'fc'):
                        netOut = np.ndarray.flatten(ag.ext_fcnet_output)[0].flatten()[0]

                    print (" *** ", delta, delta + netOut, netGain*netOut, ag.learning_rate)

                    diff_theta = 0.6 * max(min((icoInSteer), 5.), -5.)

                    netErr[:,:] = 0.
                    diff_theta = diff_theta + reflexGain * colourStrength * delta

                    curr_act = np.zeros(7).tolist()
                    curr_act[0] = 0
                    curr_act[1] = 0
                    curr_act[2] = 0 #shoot
                    curr_act[3] = curr_act[3] + diff_z
                    curr_act[4] = 0
                    curr_act[5] = 0.
                    curr_act[6] = diff_theta + netGain*netOut

                    iter += 1


            if (curr_step % epoch == 0):
                ag.save('/home/paul/Dev/GameAI/vizdoom_cig2017/icodoom/ICO1/checkpoints/BP', curr_step)

            img, meas, rwrd, term = simulator.step(curr_act)
            if (not (meas is None)) and meas[0] > 30.:
                meas[0] = 30.

            if not term:
                img_buffer[curr_step % historyLen] = img
                meas_buffer[curr_step % historyLen] = meas
                act_buffer[curr_step % historyLen] = curr_act[:7]
        curr_step += 1


    simulator.close_game()

if __name__ == '__main__':
    main()
