"""This file is used fopr creating a build script in python and using it to build+run a C program.
NOTE: This assumes you have gcc installed"""

import os # Used to determine if machine is windows or unix/macOS
import subprocess # Used to run commands from python
import tarfile # importing the "tarfile" module

def createDir(filename,where):  
    # open file
    file = tarfile.open(filename)
    # extracting file
    file.extractall(where)
    file.close()

def make(generator, path):
    """Function to run make c makefile
    NOTE: Fortran77compliler needed and some correction in make files"""
    print(f"Run make for: {generator} From source file: {path}\n")
    subprocess.run(["make"], cwd=path, shell=True)

def compile(filename, binary_filename):
    """Function to compile the provided file in gcc"""
    # Below is equivalent to running: gcc -o hello_world hello_world.c
    print(f"Creating binary: {binary_filename} From source file: {filename}\n")
    subprocess.run(["gcc", "-o", binary_filename, filename])

def run_binary(binary_filename):
    """Runs the provided binary"""
    print(f"Running binary: {binary_filename}\n")
    subprocess.run([binary_filename])


GENERATORS_TGZ_PATH='generators'
GENERATORS_FOLDER='generators/binary'
def main():
    generators_list = [file for file in os.listdir(GENERATORS_TGZ_PATH) 
         if os.path.isfile(os.path.join(GENERATORS_TGZ_PATH, file))]

    print("Files in '", GENERATORS_TGZ_PATH, "' :",generators_list)
    # exist or not.
    if not os.path.exists(GENERATORS_FOLDER):
        # if the demo_folder directory is not present 
        # then create it.
        print("Create Generators binary sources folders:",GENERATORS_FOLDER)
        os.makedirs(GENERATORS_FOLDER)
    for generator in generators_list:
        path=os.path.join(GENERATORS_TGZ_PATH,generator)
        gen_name=os.path.splitext(generator)[0]
        if not os.path.exists(os.path.join(GENERATORS_FOLDER,gen_name)):
            print(f"Uncompress and create {generator} folder")
            createDir(path,GENERATORS_FOLDER)
    generators_bin = os.listdir(GENERATORS_FOLDER)
    print("Directory in '", GENERATORS_FOLDER, "' :",generators_bin)
    for generator in generators_bin:
        make(generator,os.path.join(GENERATORS_FOLDER,generator,"src"))
    
if __name__ == "__main__":
    main()