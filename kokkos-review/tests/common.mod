units           lj
atom_style      atomic
lattice         fcc 0.8442
region          box block 0 5 0 5 0 5
create_box      1 box
create_atoms    1 box
mass            1 1.0
velocity        all create 1.0 87287 loop geom
neighbor        0.3 bin
neigh_modify    delay 0 every 1 check no
