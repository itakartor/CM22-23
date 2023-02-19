import subprocess # Used to run commands from python

def ggGraph():
    for i in range(1,11):
        print(f"Generatore net: {i}")
        input_file= open("../binary/ggraph/param/ggraph"+str(i)+".par")
        output_file=open("../binary/ggraph/ggraph"+str(i)+".dmx",'w')
        subprocess.call(["./src/ggraph"],stdin=input_file,stdout=output_file ,cwd="../binary/ggraph")