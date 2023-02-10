import subprocess # Used to run commands from python

for i in range(1,7):
    print(f"Generatore net: {i}")
    input_file= open("../binary/complete/param/parcomp"+str(i)+".net")
    subprocess.call(["./src/complete", f"complete{i}.dmx"],stdin=input_file,cwd="../binary/complete")