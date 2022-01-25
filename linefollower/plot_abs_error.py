#!/usr/bin/python3
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


# Creates a scrolling data display: column col of the tsvfile
class RealtimePlotWindow:

    def __init__(self,tsvfile,col):
        self.col = col
        self.tsvfile = tsvfile
        # create a plot window
        self.fig, self.ax = plt.subplots()
        self.ax.set_ylim(-0.02, 0.02)
        # that's our plotbuffer
        self.plotbuffer = np.zeros(2000)
        # create an empty line
        self.line, = self.ax.plot(self.plotbuffer)
        # start the animation
        self.ani = animation.FuncAnimation(self.fig, self.update, interval=100)

    # updates the plot
    def update(self, data):
        try:
            data = np.loadtxt(self.tsvfile);
            self.plotbuffer = np.append(data[-2000:,self.col],np.zeros(2000))
            self.plotbuffer = self.plotbuffer[:2000]
            self.line.set_ydata(self.plotbuffer)
        except:
            pass
        return self.line,


# Create an instance of an animated scrolling window
realtimePlotWindow = RealtimePlotWindow("llog.tsv",2)

# show the plot and start the animation
plt.show()

print("finished")
