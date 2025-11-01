#/bin/bash

#######################################
# Created on Fri Nov 10 22:55:13 2023 #
# @author: ZHANG Jun                  #
#######################################


#run vasp
echo "Starting time: $(date)"                                                  >> out.log
echo '============='                                                           >> out.log
echo 'INCAR'                                                                   >> out.log
echo '============='                                                           >> out.log
echo                                                                           >> out.log
cat INCAR                                                                      >> out.log
echo '============='                                                           >> out.log
echo 'POSCAR'                                                                  >> out.log
echo '============='                                                           >> out.log
cat POSCAR                                                                     >> out.log
echo '============='                                                           >> out.log
echo                                                                           >> out.log

mpirun vasp_std                                                                >> out.log

echo '============='                                                           >> out.log
echo CONTCAR                                                                   >> out.log
echo '============='                                                           >> out.log
cat CONTCAR                                                                    >> out.log
echo '============='                                                           >> out.log
tar -cjvf DOSCAR_PROCAR_vasprun.tar.bz2  DOSCAR PROCAR vasprun.xml             >> out.log
rm DOSCAR PROCAR vasprun.xml                                                   >> out.log
echo "Ending time: $(date)"
