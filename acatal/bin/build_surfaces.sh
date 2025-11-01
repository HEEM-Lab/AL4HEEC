get_shear.py POSCAR 0 0 6 1

for i in $(seq 0 5); do add_vacuum_space.py POSCAR_shear_$i 10 6 3; done
