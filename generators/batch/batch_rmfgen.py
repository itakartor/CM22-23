import subprocess # Used to run commands from python

def rmfGen():
    for a in (4,8):
        for b in (4,8,16):
            for c in (100,500):
                print(f"Generatore net: {a} {b} {c}")
                subprocess.call(["./src/genrmf", "-out", str(a)+"-"+str(b)+"-"+str(c), "-"+str(a) , str(a) , "-"+str(b) , str(b)  , "-"+str(c)+"1" , "0" , "-"+str(c)+"2" , str(c) , "-cost" , "100" , "-dem" , "1000" , "-seed" , "3141592" ],cwd="../binary/rmfgen")
