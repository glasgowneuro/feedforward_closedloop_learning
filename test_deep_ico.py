import deep_ico
import numpy as np

def testDEEPICO():
    with open('test.csv', 'wb') as csvfile:
        csvfile.close()
        
    with open('test.csv', 'ab') as csvfile:
        net = deep_ico.Deep_ICO(2, 2, 1)
        inp = np.zeros(2)
        err = np.zeros(1)
        for i in range(100):
            if (i > 10) :
                inp[0] = 1
            else :
                inp[0] = 0
            if ((i > 20) and (i<90)) :
                err[0] = 1
            else :
                err[0] = 0
            net.doStep(inp,err)
            output = net.getOutput(0)
            print(output)
            np.savetxt(csvfile,np.hstack((inp,err,output)),delimiter="\t",newline="\t")
            crlf="\n"
            csvfile.write(crlf.encode())

if __name__ == '__main__':
    testDEEPICO()
