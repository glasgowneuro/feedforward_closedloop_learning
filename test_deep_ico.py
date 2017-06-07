import deep_ico
import numpy

def testDEEPICO():
    with open('test.csv', 'wb') as csvfile:
        csvfile.close()
        
    with open('test.csv', 'ab') as csvfile:
        net = deep_ico.Deep_ICO(2, 2, 1)
        inp = numpy.zeros(2)
        err = numpy.zeros(1)
        for i in range(100):
            if (i > 10) :
                inp[0] = 1
            if ((i > 20) and (i<90)) :
                err[0] = 1
            else :
                err[0] = 0
            net.step(inp,err)
            output = net.getOutput()
            print(output)
            np.savetxt(csvfile,np.hstack((inp,err,output)),delimiter="\t",newline="\t")
            crlf="\n"
            csvfile.write(crlf.encode())

if __name__ == '__main__':
    testDEEPICO()
