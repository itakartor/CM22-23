import subprocess # Used to run commands from python

# n = 2^8 nodes
for i in (8,16,32):
    for j in range(1,6):
        print(f"Generatore net: 8 - {i} - {j}")
        input_file= open("../binary/netgen/param/net8_"+str(i)+"_"+str(j)+".par")
        output_file=open("../binary/netgen/dmx/net8_"+str(i)+"_"+str(j)+".dmx",'w')
        subprocess.call(["./src/netgen"],stdin=input_file,stdout=output_file ,cwd="../binary/netgen")

# n = 2^10 nodes
for i in (8,32,64):
    for j in range(1,6):
        print(f"Generatore net: 10 - {i} - {j} ")
        input_file= open("../binary/netgen/param/net10_"+str(i)+"_"+str(j)+".par")
        output_file=open("../binary/netgen/dmx/net10_"+str(i)+"_"+str(j)+".dmx",'w')
        subprocess.call(["./src/netgen"],stdin=input_file,stdout=output_file ,cwd="../binary/netgen")

# n = 2^12 nodes
for i in (8,64,256):
    for j in range(1,6):
        print(f"Generatore net: 12 - {i} -{j} ")
        input_file= open("../binary/netgen/param/net12_"+str(i)+"_"+str(j)+".par")
        output_file=open("../binary/netgen/dmx/net12_"+str(i)+"_"+str(j)+".dmx",'w')
        subprocess.call(["./src/netgen"],stdin=input_file,stdout=output_file ,cwd="../binary/netgen")

# n = 2^14 nodes
for i in (8,64):
    for j in range(1,6):
        print(f"Generatore net: 14 - {i} - {j}")
        input_file= open("../binary/netgen/param/net14_"+str(i)+"_"+str(j)+".par")
        output_file=open("../binary/netgen/dmx/net14_"+str(i)+"_"+str(j)+".dmx",'w')
        subprocess.call(["./src/netgen"],stdin=input_file,stdout=output_file ,cwd="../binary/netgen")

# n = 2^16 nodes

for j in range(1,6):
    print(f"Generatore net: 16 - 8 - {j}")
    input_file= open("../binary/netgen/param/net16_8_"+str(j)+".par")
    output_file=open("../binary/netgen/dmx/net16_8_"+str(j)+".dmx",'w')
    subprocess.call(["./src/netgen"],stdin=input_file,stdout=output_file ,cwd="../binary/netgen")