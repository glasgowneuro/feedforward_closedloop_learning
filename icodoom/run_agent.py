from __future__ import print_function
import numpy as np
import cv2
#import tensorflow as tf
import sys
sys.path.append('./agent')
sys.path.append('./deep_feedback_learning')
from agent.doom_simulator import DoomSimulator
#from agent.agent import Agent
from deep_feedback_learning import DeepFeedbackLearning
import threading
from matplotlib import pyplot as plt

width = 320
widthIn = 320
height = 240
heightIn = 240
nFiltersInput = 3
nFiltersHidden = 3
nHidden = [16, 10, 10]
# nFiltersHidden = 0 means that the layer is linear without filters
minT = 3
maxT = 15

deepBP = DeepFeedbackLearning(width * height, [nHidden[0], nHidden[1], nHidden[2]], 1, nFiltersInput, nFiltersHidden, minT, maxT)
# init the weights
deepBP.getLayer(0).setConvolution(width, height)
deepBP.initWeights(1E-6, 1)
deepBP.setBias(1)
deepBP.setAlgorithm(DeepFeedbackLearning.ico)
deepBP.setLearningRate(1E-4)
deepBP.seedRandom(89)
deepBP.setUseDerivative(1)

preprocess_input_images = lambda x: x / 255. - 0.5

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
ln = False

def getWeights(neuron):
    n_neurons = deepBP.getLayer(0).getNneurons()
    n_inputs = deepBP.getLayer(0).getNeuron(neuron).getNinputs()
    weights = np.zeros(n_inputs)
    for i in range(n_inputs):
        if deepBP.getLayer(0).getNeuron(neuron).getMask(i):
            weights[i] = deepBP.getLayer(0).getNeuron(neuron).getAvgWeight(i)
        else:
            weights[i] = np.nan
    return weights.reshape(height,width)

def plotWeights():
    global ln

    while True:
        if ln:
            ln.remove()
        w1 = getWeights(0)
        for i in range(1,deepBP.getLayer(0).getNneurons()):
            w2 = getWeights(i)
            w1 = np.where(np.isnan(w2),w1,w2)
        ln = plt.imshow(w1,cmap='gray')
        plt.draw()
        print("*** UPDATE PLOT ***")
        plt.pause(10)

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
    print("COLOUR: ", [indx1, indx0])

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

    agent_args['n_ffnet_inputs'] = 2*(agent_args['resolution'][0]*agent_args['resolution'][1])
    agent_args['n_ffnet_hidden'] = np.array([50,5])
    agent_args['n_ffnet_outputs'] = 1
    agent_args['n_ffnet_act'] = 7
    agent_args['n_ffnet_meas'] = simulator.num_meas
    agent_args['learning_rate'] = 1E-4

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

#    ag = Agent(agent_args)

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
    reflexGain = 0.01
    oldHealth = 0.

    # create masks for left and right visual fields - note that these only cover the upper half of the image
    # this is to help prevent the tracking getting confused by the floor pattern
    half_height = round(height/2)
    half_width = round(width/2)

    maskLeft = np.zeros([height, width], np.uint8)
    maskLeft[half_height:, :half_width] = 1.
    maskRight = np.zeros([height, width], np.uint8)
    maskRight[half_height:, half_width:] = 1.

    # for ICO, the errors are the same dimensionality as the first hidden layer
    netErr = np.zeros(nHidden[0])

#    deepIcoEfference = Deep_ICO(simulator_args['resolution'][0] * simulator_args['resolution'][1] + 7, 10, 1)
    nh = np.asarray([36,36])
#    deepIcoEfference = Deep_ICO_Conv(1, [1], 1, Deep_ICO_Conv.conv)
#    deepIcoEfference = Deep_ICO_Conv(simulator_args['resolution'][0] * simulator_args['resolution'][1] + 7,
#                                     nh, simulator_args['resolution'][0] * simulator_args['resolution'][1], Deep_ICO_Conv.conv)
#    deepIcoEfference.setLearningRate(0.01)
#    deepIcoEfference.setAlgorithm(Deep_ICO.backprop)
#    print ("Model type: ", "ff" if deepIcoEfference.getModelType() == 0 else "conv")

#    deepIcoEfference.initWeights(1 / (np.sqrt(float(simulator_args['resolution'][0] * simulator_args['resolution'][1] + 7))))
#    deepIcoEfference.initWeights(0.0)
    outputImage = np.zeros(simulator_args['resolution'][0] * simulator_args['resolution'][1])
    imageDiff = np.zeros(simulator_args['resolution'][0] * simulator_args['resolution'][1])
    outputArray = np.zeros(1) #deepIcoEfference.getNoutputs())

    lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    feature_params = dict(maxCorners=500, qualityLevel=0.03, minDistance=7, blockSize=7)

    imgCentre = np.array([simulator_args['resolution'][0] / 2, simulator_args['resolution'][1] /2])
    print ("Image centre: ", imgCentre)
    simpleInputs1 = np.zeros((width, height))
    simpleInputs2 = np.zeros((width, height))
    input_buff = np.zeros((1,width*height))
    target_buff = np.zeros((1,1))


    t = threading.Thread(target=plotWeights)
    t.start()

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
            greyImg1 = np.sum(img1, axis=0)
            greyImg2 = cv2.resize(stateImg, (width,height))
            greyImg2 = np.array(np.sum(greyImg2, axis=2)/3, dtype='uint8')

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

                # Don't run any networks when the player is dead!
                if (health < 101. and health > 0.):
                    #print (curr_step)

                    icoInLeft = (flowErrorLeft - errorThresh) if (flowErrorLeft - errorThresh) > 0. else 0. / reflexGain
                    icoInRight = (flowErrorRight - errorThresh) if (flowErrorRight - errorThresh) > 0. else 0. / reflexGain
                    icoInSteer = ((flowErrorRight - errorThresh) if (flowErrorRight - errorThresh) > 0. else 0. / reflexGain -
                    (flowErrorLeft - errorThresh) if (flowErrorLeft - errorThresh) > 0. else 0. / reflexGain)

                    centre1, bottomLeft1, topRight1, colourStrength1 = getMaxColourPos(img1, [255, 0, 0])
                    centre2, bottomLeft2, topRight2, colourStrength2 = getMaxColourPos(img2, [255, 0, 0])
                    colourSteer = centre2[0]
                    # get the setpoint in the -.9/+.9 range
                    simpleInputs1[:,:] = 0.1*np.random.rand(width, height)
                    simpleInputs2[:,:] = 0.1*np.random.rand(width, height)
                    sp = 1.8*(colourSteer - imgCentre[0]) / width
                    print ("ColourSteer: ", colourSteer, " ColourStrength: ", colourStrength2)

                    if(colourStrength2 > 150.):
                        #print ("ColourSteer: ", colourSteer, " ColourStrength: ", colourStrength)
                        #inputs[colourSteer,:] = colourStrength / 300.
                        simpleInputs2[bottomLeft2[0]:topRight2[0], bottomLeft2[1]:topRight2[1]] = 1.
                        #print(bottomLeft[0], bottomLeft[1], topRight[0], topRight[1], np.sum(inputs))
                    else:
                        colourStrength2 = 0.
                        sp =0
                    if (colourStrength1 > 150.):
                        simpleInputs1[bottomLeft1[0]:topRight1[0], bottomLeft1[1]:topRight1[1]] = 1.

                    netErr[:] = 0.
                    #deepBP.doStep(np.ndarray.flatten(inputs), np.ndarray.flatten(netErr))
                    #icoSteer = deepBP.getOutput(0)
                    #delta = sp - icoSteer
                    delta = 0.06 * colourStrength2 * (colourSteer - imgCentre[0])/width
                    #delta = 0.6 * max(min((icoInSteer), 5.), -5.)
                    #delta = 1. - icoSteer

                    #input_buff[0,:] = preprocess_input_images(np.ndarray.flatten(img2[2,:,:]))
                    #input_buff[0,:] = np.ndarray.flatten(inputs)
                    #input_buff[0,:] = np.concatenate([np.ndarray.flatten(greyImg1), np.ndarray.flatten(greyImg2)])
                    greyImg2 = cv2.filter2D(greyImg2, -1, edge)
                    input_buff[0,:] = np.ndarray.flatten(preprocess_input_images(greyImg2))
                    target_buff[0,0] = delta
                    if (False):
                        deepBP.setLearningRate(0.)
                        #net_output = np.ndarray.flatten(agent.test_ffnet(input_buff))[0]
                    #else:
                        #net_output = np.ndarray.flatten(agent.learn_ffnet(input_buff, target_buff))[0]

                    netErr[:] = delta
                    deepBP.doStep(preprocess_input_images(greyImg2.flatten()), netErr.flatten())

                    icoSteer = deepBP.getOutput(0)
                    print (" ** ", curr_step, icoSteer, " ", delta, " ", colourStrength2)

                    diff_theta = 0.6 * max(min((icoInSteer), 5.), -5.)

                    diff_theta = diff_theta + 0.01 * colourStrength2 * (colourSteer - imgCentre[0])/width
                    diff_theta = diff_theta + 10. * icoSteer
                    #diff_theta = diff_theta + 20. * net_output
                    curr_act = np.zeros(7).tolist()
                    curr_act[0] = 0
                    curr_act[1] = 0
                    curr_act[2] = 0
                    curr_act[3] = curr_act[3] + diff_z
                    curr_act[3] = 0.
                    curr_act[4] = 0
                    curr_act[5] = 0
                    curr_act[6] = curr_act[6] + diff_theta
                    oldHealth = health


            img, meas, rwrd, term = simulator.step(curr_act)
            if (not (meas is None)) and meas[0] > 30.:
                meas[0] = 30.

            if not term:
                img_buffer[curr_step % historyLen] = img
                meas_buffer[curr_step % historyLen] = meas
                act_buffer[curr_step % historyLen] = curr_act[:7]

            #if curr_step % epoch == 0:
            #    agent.save('/home/paul/Dev/GameAI/vizdoom_cig2017/icolearner/ICO1/checkpoints', curr_step)

#                np.save('/home/paul/tmp/icoSteer-' + str(curr_step), icoSteer.weights)
#                np.save('/home/paul/tmp/imageDiff-' + str(curr_step), imageDiff)
    #            np.save('/home/paul/tmp/icoDetect-' + str(curr_step), icoDetect.weights)

        #            icoSteer.saveInputs(curr_step)
        curr_step += 1


    simulator.close_game()
#    ag.save('/home/paul/Dev/GameAI/vizdoom_cig2017/icolearner/ICO1/checkpoints/' + 'hack-' + str(iter))


if __name__ == '__main__':
    main()