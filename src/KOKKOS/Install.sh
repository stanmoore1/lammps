# Install/unInstall package files in LAMMPS
# mode = 0/1/2 for uninstall/install/update

mode=$1

# enforce using portable C locale
LC_ALL=C
export LC_ALL

# arg1 = file, arg2 = file it depends on

action () {
  if (test $mode = 0) then
    rm -f ../$1
  elif (! cmp -s $1 ../$1) then
    if (test -z "$2" || test -e ../$2) then
      cp $1 ..
      if (test $mode = 2) then
        echo "  updating src/$1"
      fi
    fi
  elif (test -n "$2") then
    if (test ! -e ../$2) then
      rm -f ../$1
    fi
  fi
}

# force rebuild of files with LMP_KOKKOS switch

KOKKOS_INSTALLED=0
if (test -e ../Makefile.package) then
  KOKKOS_INSTALLED=`grep DLMP_KOKKOS ../Makefile.package | wc -l`
fi

if (test $mode = 1) then
  if (test $KOKKOS_INSTALLED = 0) then
    touch ../accelerator_kokkos.h
  fi
elif (test $mode = 0) then
  if (test $KOKKOS_INSTALLED = 1) then
    touch ../accelerator_kokkos.h
  fi
fi

# list of files with optional dependencies

action atom_kokkos.cpp
action atom_kokkos.h
action atom_map_kokkos.cpp
action atom_vec_atomic_kokkos.cpp
action atom_vec_atomic_kokkos.h
action atom_vec_kokkos.cpp
action atom_vec_kokkos.h
action comm_kokkos.cpp
action comm_kokkos.h
action comm_tiled_kokkos.cpp
action comm_tiled_kokkos.h
action compute_temp_kokkos.cpp
action compute_temp_kokkos.h
action domain_kokkos.cpp
action domain_kokkos.h
action fix_nve_kokkos.cpp
action fix_nve_kokkos.h
action fix_property_atom_kokkos.cpp
action fix_property_atom_kokkos.h
action kokkos.cpp
action kokkos.h
action kokkos_base.h
action kokkos_few.h
action kokkos_type.h
action meam_kokkos.h meam.h
action meam_dens_final_kokkos.h meam_dens_final.cpp
action meam_dens_init_kokkos.h meam_dens_init.cpp
action meam_force_kokkos.h meam_force.cpp
action meam_funcs_kokkos.h meam_funcs.cpp
action meam_impl_kokkos.h meam_impl.cpp
action meam_setup_done_kokkos.h meam_setup_done.cpp
action memory_kokkos.h
action mliap_data_kokkos.cpp mliap_data.cpp
action mliap_data_kokkos.h mliap_data.h
action mliap_descriptor_kokkos.h mliap_descriptor.h
action mliap_descriptor_so3_kokkos.cpp mliap_descriptor_so3.cpp
action mliap_descriptor_so3_kokkos.h mliap_descriptor_so3.h
action mliap_model_linear_kokkos.cpp mliap_model_linear.cpp
action mliap_model_linear_kokkos.h mliap_model_linear.h
action mliap_model_python_kokkos.cpp mliap_model_linear.cpp
action mliap_model_python_kokkos.h mliap_model_linear.h
action mliap_model_kokkos.h mliap_model.h
action mliap_so3_kokkos.cpp mliap_so3.cpp
action mliap_so3_kokkos.h mliap_so3.h
action modify_kokkos.cpp
action modify_kokkos.h
action neigh_bond_kokkos.cpp
action neigh_bond_kokkos.h
action neigh_list_kokkos.cpp
action neigh_list_kokkos.h
action neighbor_kokkos.cpp
action neighbor_kokkos.h
action npair_halffull_kokkos.cpp
action npair_halffull_kokkos.h
action npair_trim_kokkos.cpp
action npair_trim_kokkos.h
action npair_kokkos.cpp
action npair_kokkos.h
action npair_halffull_kokkos.h
action npair_halffull_kokkos.cpp
action nbin_kokkos.cpp
action nbin_kokkos.h
action pair_hybrid_kokkos.cpp
action pair_hybrid_kokkos.h
action pair_hybrid_overlay_kokkos.cpp
action pair_hybrid_overlay_kokkos.h
action pair_kokkos.h
action pair_lj_cut_kokkos.cpp
action pair_lj_cut_kokkos.h
action pair_snap_kokkos.cpp pair_snap.cpp
action pair_snap_kokkos.h pair_snap.h
action pair_snap_kokkos_impl.h pair_snap.cpp
action pair_tersoff_kokkos.cpp pair_tersoff.cpp
action pair_tersoff_kokkos.h pair_tersoff.h
action pair_zbl_kokkos.cpp
action pair_zbl_kokkos.h
action sna_kokkos.h sna.h
action sna_kokkos_impl.h sna.cpp
action third_order_kokkos.cpp dynamical_matrix.cpp
action third_order_kokkos.h dynamical_matrix.h
action transpose_helper_kokkos.h
action verlet_kokkos.cpp
action verlet_kokkos.h

# Install cython pyx file only if non-KOKKOS version is present
action mliap_model_python_couple_kokkos.pyx mliap_model_python_couple.pyx

# edit 2 Makefile.package files to include/exclude package info

if (test $1 = 1) then

  if (test -e ../Makefile.package) then
    sed -i -e 's/[^ \t]*kokkos[^ \t]* //g' ../Makefile.package
    sed -i -e 's/[^ \t]*KOKKOS[^ \t]* //g' ../Makefile.package
    sed -i -e 's|^PKG_INC =[ \t]*|&-DLMP_KOKKOS |' ../Makefile.package
#    sed -i -e 's|^PKG_PATH =[ \t]*|&-L..\/..\/lib\/kokkos\/core\/src |' ../Makefile.package
    sed -i -e 's|^PKG_CPP_DEPENDS =[ \t]*|&$(KOKKOS_CPP_DEPENDS) |' ../Makefile.package
    sed -i -e 's|^PKG_LIB =[ \t]*|&$(KOKKOS_LIBS) |' ../Makefile.package
    sed -i -e 's|^PKG_LINK_DEPENDS =[ \t]*|&$(KOKKOS_LINK_DEPENDS) |' ../Makefile.package
    sed -i -e 's|^PKG_SYSINC =[ \t]*|&$(KOKKOS_CPPFLAGS) $(KOKKOS_CXXFLAGS) |' ../Makefile.package
    sed -i -e 's|^PKG_SYSLIB =[ \t]*|&$(KOKKOS_LDFLAGS) |' ../Makefile.package
#    sed -i -e 's|^PKG_SYSPATH =[ \t]*|&$(kokkos_SYSPATH) |' ../Makefile.package
  fi

  if (test -e ../Makefile.package.settings) then
    sed -i -e '/CXX\ =\ \$(CC)/d' ../Makefile.package.settings
    sed -i -e '/^[ \t]*include.*kokkos.*$/d' ../Makefile.package.settings
    # multiline form needed for BSD sed on Macs
    sed -i -e '4 i \
CXX = $(CC)
' ../Makefile.package.settings
    sed -i -e '5 i \
include ..\/..\/lib\/kokkos\/Makefile.kokkos
' ../Makefile.package.settings
  fi

  #  comb/omp triggers a persistent bug in nvcc. deleting it.
  rm -f ../*_comb_omp.*

elif (test $1 = 2) then

  #  comb/omp triggers a persistent bug in nvcc. deleting it.
  rm -f ../*_comb_omp.*

elif (test $1 = 0) then

  if (test -e ../Makefile.package) then
    sed -i -e 's/[^ \t]*kokkos[^ \t]* //g' ../Makefile.package
    sed -i -e 's/[^ \t]*KOKKOS[^ \t]* //g' ../Makefile.package
  fi

  if (test -e ../Makefile.package.settings) then
    sed -i -e '/CXX\ =\ \$(CC)/d' ../Makefile.package.settings
    sed -i -e '/^[ \t]*include.*kokkos.*$/d' ../Makefile.package.settings
  fi

fi

# Python cython stuff. Only need to convert/remove sources.
# Package settings were already done in ML-IAP package Install.sh script.

if (test $1 = 1) then
  if (type cythonize > /dev/null 2>&1 && test -e ../python_impl.cpp) then
    cythonize -3 ../mliap_model_python_couple_kokkos.pyx
  fi

elif (test $1 = 0) then
  rm -f ../mliap_model_python_couple_kokkos.cpp ../mliap_model_python_couple_kokkos.h

elif (test $1 = 2) then
  if (type cythonize > /dev/null 2>&1 && test -e ../python_impl.cpp) then
    cythonize -3 ../mliap_model_python_couple_kokkos.pyx
  else
    rm -f ../mliap_model_python_couple_kokkos.cpp ../mliap_model_python_couple_kokkos.h
  fi
fi
