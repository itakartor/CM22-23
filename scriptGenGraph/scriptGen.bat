# This file is write with WLS in Windows system
# This script is usefull for to generate all test of a graph generator
# We have to compile the generator with command make in the directory src
# Then we can copy or move the executable in the directory with script
# In the end we have script with near somethings like:
# - param directory
# - src directory
# - script
# - executable of the generator
# Now we can execute the script with ./script
# I used only 2 gen for now 
mkdir ./output;
mkdir ./output/complete;
mkdir ./output/netgen;
for i in {1..6};do ./complete/complete ./output/complete/complete$i.dmx < ./complete/param/parcomp$i.net ;done

for i in 8 16 32 ; do for j in {1..5}; do ./netgen/src/netgen  < ./netgen/param/net8_${i}_${j}.par > ./output/netgen/net8_${i}_${j}.dmx; done; done

# n = 2^10 nodes

for i in 8 32 64 ; do for j in {1..5}; do ./netgen/src/netgen  < ./netgen/param/net10_${i}_${j}.par > ./output/netgen/net10_${i}_${j}.dmx; done; done

# n = 2^12 nodes

for i in 8 64 256 ; do for j in {1..5}; do ./netgen/src/netgen  < ./netgen/param/net12_${i}_${j}.par > ./output/netgen/net12_${i}_${j}.dmx; done; done

# n = 2^14 nodes

for i in 8 64 ;do for j in {1..5}; do ./netgen/src/netgen  < ./netgen/param/net14_${i}_${j}.par > ./output/netgen/net14_${i}_${j}.dmx; done; done

# n = 2^16 nodes

for i in 8 ; do for j in {1..5}; do ./netgen/src/netgen  < ./netgen/param/net16_${i}_${j}.par > ./output/netgen/net16_${i}_${j}.dmx; done; done