import subprocess # Used to run commands from python

def completeG():
    for i in range(1,7):
        print(f"Generatore Completenet: {i}")
        input_file= open("../binary/complete/param/parcomp"+str(i)+".net")
        subprocess.call(["./src/complete", f"../../dmx/complete{i}.dmx"],stdin=input_file,cwd="../binary/complete")