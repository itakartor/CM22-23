import subprocess # Used to run commands from python

#8
def goto8():
    for i in (8,16,32):
        for j in (1,2,3,4,5):
            print(f"Generatore net: {i}-{j}")
            input_file= open("../binary/goto/param/pargoto8_"+str(i)+"_"+str(j)+".net")
            output_file=open("../dmx/goto8_"+str(i)+"_"+str(j)+".dmx",'w')
            subprocess.call(["./src/goto"],stdin=input_file,stdout=output_file ,cwd="../binary/goto")
#10
def goto10():
    for i in (8,32,64):
        for j in (1,2,3,4,5):
            print(f"Generatore net: {i}-{j}")
            input_file= open("../binary/goto/param/pargoto10_"+str(i)+"_"+str(j)+".net")
            output_file=open("../dmx/goto10_"+str(i)+"_"+str(j)+".dmx",'w')
            subprocess.call(["./src/goto"],stdin=input_file,stdout=output_file ,cwd="../binary/goto")
        
# 12 - 14*
def goto12_14():
    for i in (12,14):
        for j in (8,64,256):
            for h in (1,2,3,4,5):
                print(f"Generatore net: {i}-{j}")
                input_file= open("../binary/goto/param/pargoto"+str(i)+"_"+str(j)+"_"+str(h)+".net")
                output_file=open("../dmx/goto"+str(i)+"_"+str(j)+"_"+str(h)+".dmx",'w')
                subprocess.call(["./src/goto"],stdin=input_file,stdout=output_file ,cwd="../binary/goto")

def gotoAll():
    goto8()
    goto10()
    goto12_14()
'''
# 16*
for i in (8,64,1024):
    for j in (1,2,3,4,5):
        print(f"Generatore net: {i}-{j}")
        input_file= open("../binary/goto/param/pargoto16_"+str(i)+"_"+str(j)+".net")
        output_file=open("../dmx/goto16_"+str(i)+"_"+str(j)+".dmx",'w')
        subprocess.call(["./src/goto"],stdin=input_file,stdout=output_file ,cwd="../binary/goto")
'''