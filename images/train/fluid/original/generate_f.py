#!/usr/bin/env python
from subprocess import Popen
print("Generate")

n=1
for i in range(12):
    for j in range(4):
        p=Popen(['patchy2d','source.conf','-s','1000000','--new_number_of_particles','{}'.format(n),'-n','f{}'.format(n)])
        n+=1
    p.wait()
