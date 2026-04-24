import numpy as np
import matplotlib.pyplot as plt
import math

nwater = 4096
natoms = nwater * 3
nframes = 101
box_length = 49.66876414
rmax = box_length / 2.0
nbins = 500
traj_file  = "water_traj.xyz"


def read_frames(filename):
    """Return (atom_types_list, coords_array) for each frame in an XYZ file."""
    coords = np.zeros((3 * natoms * nframes,), dtype=np.float64)
    icoord = 0
    with open(filename) as fh:
        iframe = 0
        while iframe < nframes:
            iframe += 1
            print(f"Reading frame {iframe}", flush=True)

            # Read the header
            fh.readline()

            # Read the comment line
            fh.readline()

            # Read the atom types and positions
            for iatom in range(natoms):
                split_line = fh.readline().split()
                coords[icoord + 0] = float(split_line[1])
                coords[icoord + 1] = float(split_line[2])
                coords[icoord + 2] = float(split_line[3])
                icoord += 3

    return coords


def distance(coords, istart, jstart):
    '''Returns distance between particles i and j
       The coordinates of the full system are in coords'''
    dx = abs( coords[istart] - coords[jstart] )
    if dx > box_length / 2.0:
        dx -= box_length

    dy = abs( coords[istart + 1] - coords[jstart + 1] )
    if dy > box_length / 2.0:
        dy -= box_length

    dz = abs( coords[istart + 2] - coords[jstart + 2] )
    if dz > box_length / 2.0:
        dz -= box_length

    dr2 = dx * dx + dy * dy + dz * dz
    return math.sqrt(dr2)


def accumulate(coords):
    oo_counts = np.zeros(nbins, dtype=np.float64)
    oh_counts = np.zeros(nbins, dtype=np.float64)
    hh_counts = np.zeros(nbins, dtype=np.float64)

    for iframe in range(nframes):
        print(f"Accumulating frame {iframe+1}")
        for imol in range(nwater):
            for jmol in range(imol+1, nwater):
                imol_start = (9 * imol) + (9 * nwater * iframe)
                jmol_start = (9 * jmol) + (9 * nwater * iframe)

                # Get the O-O interaction
                dr = distance(coords, imol_start, jmol_start)
                ibin = math.floor( ( dr / rmax ) * nbins )
                if ibin < nbins:
                    oo_counts[ibin] += 2

                # Get the O-H interactions
                dr = distance(coords, imol_start, jmol_start + 3)
                ibin = math.floor( ( dr / rmax ) * nbins )
                if ibin < nbins:
                    oh_counts[ibin] += 1

                dr = distance(coords, imol_start, jmol_start + 6)
                ibin = math.floor( ( dr / rmax ) * nbins )
                if ibin < nbins:
                    oh_counts[ibin] += 1

                dr = distance(coords, imol_start + 3, jmol_start)
                ibin = math.floor( ( dr / rmax ) * nbins )
                if ibin < nbins:
                    oh_counts[ibin] += 1

                dr = distance(coords, imol_start + 6, jmol_start)
                ibin = math.floor( ( dr / rmax ) * nbins )
                if ibin < nbins:
                    oh_counts[ibin] += 1

                # Get the H-H interactions
                dr = distance(coords, imol_start + 3, jmol_start + 3)
                ibin = math.floor( ( dr / rmax ) * nbins )
                if ibin < nbins:
                    hh_counts[ibin] += 2

                dr = distance(coords, imol_start + 3, jmol_start + 6)
                ibin = math.floor( ( dr / rmax ) * nbins )
                if ibin < nbins:
                    hh_counts[ibin] += 2

                dr = distance(coords, imol_start + 6, jmol_start + 3)
                ibin = math.floor( ( dr / rmax ) * nbins )
                if ibin < nbins:
                    hh_counts[ibin] += 2

                dr = distance(coords, imol_start + 6, jmol_start + 6)
                ibin = math.floor( ( dr / rmax ) * nbins )
                if ibin < nbins:
                    hh_counts[ibin] += 2

    return (oo_counts, oh_counts, hh_counts)


def normalize(counts, n_atom1, n_atom2):
    """
    Convert distance counts to g(r).

    g(r) = counts * V / (n_atom1 * n_atom2 * nframes * 4*pi*r^2*dr)
    """
    dr      = rmax / nbins
    r       = np.arange(1, nbins + 1) * dr - dr / 2.0   # bin centres
    shell   = 4.0 * np.pi * r**2 * dr
    g_r     = (counts * box_length**3) / (n_atom1 * nframes * n_atom2 * shell)
    return r, g_r


if __name__ == "__main__":
    counts_oo = np.zeros(nbins)
    counts_oh = np.zeros(nbins)
    counts_hh = np.zeros(nbins)

    coords = read_frames(traj_file)

    counts_oo, counts_oh, counts_hh = accumulate(coords)

    n_o = nwater
    n_h = 2 * nwater
    r, g_oo = normalize(counts_oo, n_o, n_o)
    r, g_oh = normalize(counts_oh, n_o, n_h)
    r, g_hh = normalize(counts_hh, n_h, n_h)

    # Plot the results
    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
    for ax, (label, g) in zip(axes, [("OO", g_oo), ("OH", g_oh), ("HH", g_hh)]):
        ax.plot(r, g, lw=1.0)
        ax.axhline(1.0, color="k", ls="--", lw=0.6, alpha=0.5)
        ax.set_ylabel(f"g$_{{{label}}}$(r)", fontsize=12)
        ax.set_xlim(0, rmax)
        ax.set_ylim(bottom=0)
    axes[-1].set_xlabel("r (Angstrom)", fontsize=12)
    fig.suptitle("Water radial distribution functions", fontsize=13)
    plt.tight_layout()
    plt.savefig("water_rdfs.png", dpi=150)
