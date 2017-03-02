/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing authors:
   James Larentzos (ARL) and Timothy I. Mattox (Engility Corporation)
------------------------------------------------------------------------- */

#include "nbin_ssa_kokkos.h"
#include "neighbor.h"
#include "atom_kokkos.h"
#include "group.h"
#include "domain.h"
#include "comm.h"
#include "update.h"
#include "error.h"
#include "atom_masks.h"

// #include "memory.h"

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

template<class DeviceType>
NBinSSAKokkos<DeviceType>::NBinSSAKokkos(LAMMPS *lmp) : NBinStandard(lmp)
{
  atoms_per_bin = ghosts_per_gbin = 16;

  d_resize = typename AT::t_int_scalar("NBinSSAKokkos::d_resize");
  d_lbinxlo = typename AT::t_int_scalar("NBinSSAKokkos::d_lbinxlo");
  d_lbinylo = typename AT::t_int_scalar("NBinSSAKokkos::d_lbinylo");
  d_lbinzlo = typename AT::t_int_scalar("NBinSSAKokkos::d_lbinzlo");
  d_lbinxhi = typename AT::t_int_scalar("NBinSSAKokkos::d_lbinxhi");
  d_lbinyhi = typename AT::t_int_scalar("NBinSSAKokkos::d_lbinyhi");
  d_lbinzhi = typename AT::t_int_scalar("NBinSSAKokkos::d_lbinzhi");
#ifndef KOKKOS_USE_CUDA_UVM
  h_resize = Kokkos::create_mirror_view(d_resize);
  h_lbinxlo = Kokkos::create_mirror_view(d_lbinxlo);
  h_lbinylo = Kokkos::create_mirror_view(d_lbinylo);
  h_lbinzlo = Kokkos::create_mirror_view(d_lbinzlo);
  h_lbinxhi = Kokkos::create_mirror_view(d_lbinxhi);
  h_lbinyhi = Kokkos::create_mirror_view(d_lbinyhi);
  h_lbinzhi = Kokkos::create_mirror_view(d_lbinzhi);
#else
  h_resize = d_resize;
  h_lbinxlo = d_lbinxlo;
  h_lbinylo = d_lbinylo;
  h_lbinzlo = d_lbinzlo;
  h_lbinxhi = d_lbinxhi;
  h_lbinyhi = d_lbinyhi;
  h_lbinzhi = d_lbinzhi;
#endif
  h_resize() = 1;

  k_gbincount = DAT::tdual_int_1d("NBinSSAKokkos::gbincount",8);
  gbincount = k_gbincount.view<DeviceType>();
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void NBinSSAKokkos<DeviceType>::bin_atoms_setup(int nall)
{
  if (mbins > (int) k_bins.d_view.dimension_0()) {
    k_bins = DAT::tdual_int_2d("NBinSSAKokkos::bins",mbins,atoms_per_bin);
    bins = k_bins.view<DeviceType>();

    k_bincount = DAT::tdual_int_1d("NBinSSAKokkos::bincount",mbins);
    bincount = k_bincount.view<DeviceType>();
  }

  ghosts_per_gbin = atom->nghost / 7; // estimate needed size

  if (ghosts_per_gbin > (int) k_gbins.d_view.dimension_1()) {
    k_gbins = DAT::tdual_int_2d("NBinSSAKokkos::gbins",8,ghosts_per_gbin);
    gbins = k_gbins.view<DeviceType>();
  }

  // Clear the local bin extent bounding box.
  h_lbinxlo() = mbinx - 1; // Safe to = stencil->sx + 1
  h_lbinylo() = mbiny - 1; // Safe to = stencil->sy + 1
  h_lbinzlo() = mbinz - 1; // Safe to = stencil->sz + 1
  h_lbinxhi() = 0; // Safe to = mbinx - stencil->sx - 1
  h_lbinyhi() = 0; // Safe to = mbiny - stencil->sy - 1
  h_lbinzhi() = 0; // Safe to = mbinz - stencil->sz - 1
  deep_copy(d_lbinxlo, h_lbinxlo);
  deep_copy(d_lbinylo, h_lbinylo);
  deep_copy(d_lbinzlo, h_lbinzlo);
  deep_copy(d_lbinxhi, h_lbinxhi);
  deep_copy(d_lbinyhi, h_lbinyhi);
  deep_copy(d_lbinzhi, h_lbinzhi);
}

/* ----------------------------------------------------------------------
   bin owned and ghost atoms for the Shardlow Splitting Algorithm (SSA)
   local atoms are in distinct bins (binhead[]) from the ghosts
   ghost atoms are "binned" in gairhead_ssa[] instead
     ghosts which are not in an Active Interaction Region (AIR) are skipped
------------------------------------------------------------------------- */

template<class DeviceType>
void NBinSSAKokkos<DeviceType>::bin_atoms()
{
  last_bin = update->ntimestep;

  int i;
  int nlocal = atom->nlocal;
  int nghost = atom->nghost;
  int nall = nlocal + nghost;

  atomKK->sync(ExecutionSpaceFromDevice<DeviceType>::space,X_MASK);
  x = atomKK->k_x.view<DeviceType>();

  sublo_[0] = domain->sublo[0];
  sublo_[1] = domain->sublo[1];
  sublo_[2] = domain->sublo[2];
  subhi_[0] = domain->subhi[0];
  subhi_[1] = domain->subhi[1];
  subhi_[2] = domain->subhi[2];

  bboxlo_[0] = bboxlo[0]; bboxlo_[1] = bboxlo[1]; bboxlo_[2] = bboxlo[2];
  bboxhi_[0] = bboxhi[0]; bboxhi_[1] = bboxhi[1]; bboxhi_[2] = bboxhi[2];

  k_binID = DAT::tdual_int_1d("NBinSSAKokkos::binID",nall);
  binID = k_binID.view<DeviceType>();

  // find each local atom's binID
  {
    atoms_per_bin = 0;
    NPairSSAKokkosBinIDAtomsFunctor<DeviceType> f(*this);
    Kokkos::parallel_reduce(nlocal, f, atoms_per_bin);
  }
  deep_copy(h_lbinxlo, d_lbinxlo);
  deep_copy(h_lbinylo, d_lbinylo);
  deep_copy(h_lbinzlo, d_lbinzlo);
  deep_copy(h_lbinxhi, d_lbinxhi);
  deep_copy(h_lbinyhi, d_lbinyhi);
  deep_copy(h_lbinzhi, d_lbinzhi);

  // find each ghost's binID (AIR number)
  {
    for (int i = 0; i < 8; i++) k_gbincount.h_view(i) = 0;
    k_gbincount.modify<LMPHostType>();
    k_gbincount.sync<DeviceType>();
    DeviceType::fence(); // FIXME?
    ghosts_per_gbin = 0;
    NPairSSAKokkosBinIDGhostsFunctor<DeviceType> f(*this);
    Kokkos::parallel_reduce(Kokkos::RangePolicy<DeviceType>(nlocal,nall), f, ghosts_per_gbin);
  }

  // actually bin the ghost atoms
  {
    if(ghosts_per_gbin > (int) gbins.dimension_1()) {
      k_gbins = DAT::tdual_int_2d("gbins", 8, ghosts_per_gbin);
      gbins = k_gbins.view<DeviceType>();
    }
    for (int i = 0; i < 8; i++) k_gbincount.h_view(i) = 0;
    k_gbincount.modify<LMPHostType>();
    k_gbincount.sync<DeviceType>();
    DeviceType::fence(); // FIXME?

    Kokkos::parallel_for(
#ifdef ALLOW_NON_DETERMINISTIC_SSA
      Kokkos::RangePolicy<DeviceType>(nlocal,nall)
#else
      Kokkos::RangePolicy<Kokkos::Serial>(nlocal,nall)
#endif
      , KOKKOS_LAMBDA (const int i) {
      const int iAIR = binID(i);
      if (iAIR > 0) { // include only ghost atoms in an AIR
        const int ac = Kokkos::atomic_fetch_add(&gbincount[iAIR], (int)1);
        gbins(iAIR, ac) = i;
      }
    });
    DeviceType::fence();
  }
  c_gbins = gbins; // gbins won't change until the next bin_atoms

  // actually bin the local atoms
  {
    if ((mbins > (int) bins.dimension_0()) ||
        (atoms_per_bin > (int) bins.dimension_1())) {
      k_bins = DAT::tdual_int_2d("bins", mbins, atoms_per_bin);
      bins = k_bins.view<DeviceType>();
    }
    MemsetZeroFunctor<DeviceType> f_zero;
    f_zero.ptr = (void*) k_bincount.view<DeviceType>().ptr_on_device();
    Kokkos::parallel_for(mbins, f_zero);
    DeviceType::fence();

    NPairSSAKokkosBinAtomsFunctor<DeviceType> f(*this);
#ifdef ALLOW_NON_DETERMINISTIC_SSA
    Kokkos::parallel_for(nlocal, f);
#else
    Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::Serial>(0, nlocal), f);
#endif
    DeviceType::fence();

  }
  c_bins = bins; // bins won't change until the next bin_atoms

//now dispose of the k_binID array
  k_binID = DAT::tdual_int_1d("NBinSSAKokkos::binID",0);
  binID = k_binID.view<DeviceType>();
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void NBinSSAKokkos<DeviceType>::binAtomsItem(const int &i) const
{
  const int ibin = binID(i);
  const int ac = Kokkos::atomic_fetch_add(&(bincount[ibin]), (int)1);
  bins(ibin, ac) = i;
}

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void NBinSSAKokkos<DeviceType>::binIDAtomsItem(const int &i, int &update) const
{
  int loc[3];
  const int ibin = coord2bin(x(i, 0), x(i, 1), x(i, 2), &(loc[0]));
  binID(i) = ibin;

  // Find the bounding box of the local atoms in the bins
  if (loc[0] < d_lbinxlo()) Kokkos::atomic_fetch_min(&d_lbinxlo(),loc[0]);
  if (loc[0] >= d_lbinxhi()) Kokkos::atomic_fetch_max(&d_lbinxhi(),loc[0] + 1);
  if (loc[1] < d_lbinylo()) Kokkos::atomic_fetch_min(&d_lbinylo(),loc[1]);
  if (loc[1] >= d_lbinyhi()) Kokkos::atomic_fetch_max(&d_lbinyhi(),loc[1] + 1);
  if (loc[2] < d_lbinzlo()) Kokkos::atomic_fetch_min(&d_lbinzlo(),loc[2]);
  if (loc[2] >= d_lbinzhi()) Kokkos::atomic_fetch_max(&d_lbinzhi(),loc[2] + 1);

  const int ac = Kokkos::atomic_fetch_add(&(bincount[ibin]), (int)1);
  if (update <= ac) update = ac + 1;
}

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void NBinSSAKokkos<DeviceType>::binIDGhostsItem(const int &i, int &update) const
{
  const int iAIR = coord2ssaAIR(x(i, 0), x(i, 1), x(i, 2));
  binID(i) = iAIR;
  if (iAIR > 0) { // include only ghost atoms in an AIR
    const int ac = Kokkos::atomic_fetch_add(&gbincount[iAIR], (int)1);
    if (update <= ac) update = ac + 1;
  }
}

namespace LAMMPS_NS {
template class NBinSSAKokkos<LMPDeviceType>;
#ifdef KOKKOS_HAVE_CUDA
template class NBinSSAKokkos<LMPHostType>;
#endif
}