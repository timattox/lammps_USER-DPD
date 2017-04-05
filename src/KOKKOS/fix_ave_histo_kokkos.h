/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef FIX_CLASS

FixStyle(ave/histo/kk,FixAveHistoKokkos)

#else

#ifndef LMP_FIX_AVE_HISTO_KOKKOS_H
#define LMP_FIX_AVE_HISTO_KOKKOS_H

#include <stdio.h>
#include "fix_ave_histo.h"

namespace LAMMPS_NS {

class FixAveHistoKokkos : public Fix {
 public:
  FixAveHistoKokkos(class LAMMPS *, int, char **);
  virtual ~FixAveHistoKokkos();

 protected:
  void bin_vector(int, double *, int);
  void bin_atoms(double *, int);
};

}

#endif
#endif

/* ERROR/WARNING messages:
 */
