import subprocess # Used to run commands from python


def rmfGen():

    for a in (4,8):
        for b in (4,8,16):
            for c in (100,500):
                print(f"Generatore RMFGEN net: {a} {b} {c}")
                subprocess.call(["./src/genrmf","-out",f"../../dmx/rmf_{str(a)}_{str(b)}_{str(c)}.dmx", "-a" , str(a) , "-b" , str(b)  , "-c1" , "0" , "-c2" , str(c) , "-cost" , "100" , "-dem" , "1000" , "-seed" , "3141592" ],  cwd="../binary/rmfgen")
