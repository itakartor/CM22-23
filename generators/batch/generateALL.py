from batch_complete import completeG
from batch_ggraph import ggGraph
from batch_goto import *
from batch_netgen import *
from batch_rmfgen import rmfGen

def generateAllGraphs():
    completeG()
    ggGraph()
    rmfGen()

if __name__ == '__main__':
    generateAllGraphs()