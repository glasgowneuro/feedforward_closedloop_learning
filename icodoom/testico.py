import numpy as np
from agent.agent import Agent
from agent.firfilter import Filterbank





def main():
    filterbank = Filterbank(num_filters=2, ntaps=5, tau=1)
    print (filterbank.bank)



if __name__ == '__main__':
    main()


