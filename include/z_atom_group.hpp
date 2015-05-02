//Copyright (c) 2015 Zachary Kann
//
//Permission is hereby granted, free of charge, to any person obtaining a copy
//of this software and associated documentation files (the "Software"), to deal
//in the Software without restriction, including without limitation the rights
//to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
//copies of the Software, and to permit persons to whom the Software is
//furnished to do so, subject to the following conditions:
//
//The above copyright notice and this permission notice shall be included in all
//copies or substantial portions of the Software.
//
//THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
//SOFTWARE.

// ---
// Author: Zachary Kann

// Contains information about groups of atoms as defined in
// supplied .ndx file. This allows the properties of a subset
// of the atoms to be treated with the same ease as the entire
// set of atoms. This class includes postion, velocity, and mass
// information for individual atoms as well as for molecules
// defined in the .top file.

#include <armadillo>
#include "z_constants.hpp"
#include "z_molecule.hpp"
#include "z_gromacs.hpp"

#ifndef _Z_ATOM_GROUP_HPP_
#define _Z_ATOM_GROUP_HPP_

//See description above
class AtomGroup {
 public:
  // Uses .gro file to make group consisting of all atoms.
  AtomGroup(const std::string& gro, const std::vector<Molecule> molecules)
      : name_("all_atoms") {
    ReadGro(gro, molecules);
  }

  // Used for creating subsets of the all-atoms group.
  AtomGroup(std::string name, std::vector<int> indices,
            const AtomGroup& all_atoms)
      : name_(name), indices_(indices) {
    Init(name, all_atoms.index_to_mass_,  all_atoms.index_to_molecule_,
         all_atoms.index_to_sigma_, all_atoms.index_to_epsilon_);
  }

  // Creates .gro file
  void WriteGro(const std::string& gro, arma::rowvec box,
                std::string description);

  // Called by all constructors to create needed initialize
  // vectors and matrices.
  void Init(std::string name, std::vector<double> index_to_mass,
            std::vector<int> index_to_molecule,
            std::vector<double> index_to_sigma,
            std::vector<double> index_to_epsilon);

  // Deletes a molecule from the group. Sends references to the
  // center of mass position and velocity in case the user wishes
  // to replace it with another molecule.
  void RemoveMolecule(const int molecule_id, arma::rowvec& old_com_position,
                      arma::rowvec& old_com_velocity);

  // Wrapper for the above function which throws away references
  // to position/velocity vectors. Used if replacement is not planned.
  inline void RemoveMolecule(const int molecule_id) {
    arma::rowvec position(DIMS);
    arma::rowvec velocity(DIMS);
    RemoveMolecule(molecule_id, position, velocity);
  }

  // Adds a molecule to the group at the given position.
  // Currently only supports monatomic molecules.
  void AddMolecule(const int molecule_id, const Molecule& mol_to_add,
                   const arma::rowvec& position, const arma::rowvec& velocity);

  // Removes old molecule and adds new one at same position.
  // TODO(Zak) Separate into versions for all_atoms group and smaller groups.
  inline void ReplaceMolecule(const int new_molecule_id,
                              const Molecule& new_molecule,
                              const int mol_to_remove) {
    arma::rowvec position (DIMS);
    arma::rowvec velocity(DIMS);
    RemoveMolecule(mol_to_remove, position, velocity);
    // TODO(Zak): allow adjustement of velocity based
    // on Maxwell-Boltzmann distribution instead of
    // simple replacement.
    AddMolecule(new_molecule_id, new_molecule, position, velocity);
  }

  // Resets center of mass position and velocity of all molecules.
  inline void ZeroCom() {
    com_positions_ = arma::zeros(num_molecules_, DIMS);
    com_velocities_ = arma::zeros(num_molecules_, DIMS);
  }

  // Sets center of mass position and velocity of all molecules.
  void UpdateCom();

  // Mutators
  inline void set_position(const int i, const rvec& position) {
    positions_.row(i) = RvecToRow(position);
  }

  inline void set_velocity(const int i, const rvec& velocity) {
    velocities_.row(i) = RvecToRow(velocity);
  }

  // Accessors
  inline std::string const name() { return name_; }
  std::vector<int>::iterator begin() { return indices_.begin(); }
  std::vector<int>::iterator end() { return indices_.end(); }
  inline int size() const { return group_size_; }
  inline std::vector<int> indices() const { return indices_; }
  inline int indices(int i) const { return indices_[i]; }
  inline double num_molecules() const { return num_molecules_; }
  inline arma::mat positions() const { return positions_; }
  inline arma::mat velocities() const { return velocities_; }
  inline arma::mat com_positions() const { return com_positions_; }
  inline arma::mat com_velocities() const { return com_velocities_; }
  inline arma::rowvec position(int atom) const { return positions_.row(atom); }
  inline arma::rowvec velocity(int atom) const { return velocities_.row(atom); }
  inline double mass(int atom) const { return index_to_mass_[atom]; }
  inline double sigma(int atom) const { return index_to_sigma_[atom]; }
  inline double epsilon(int atom) const { return index_to_epsilon_[atom]; }

  inline arma::rowvec com_position(int molecule) const {
    return com_positions_.row(molecule);
  }

  inline arma::rowvec com_velocity(int molecule) const {
    return com_velocities_.row(molecule);
  }

  inline double position(int atom, int dim) const {
    return positions_(atom, dim);
  }

  inline double velocity(int atom, int dim) const {
    return velocities_(atom, dim);
  }

  inline arma::rowvec velocity_xy(int atom) const {
    return velocities_(atom, arma::span(0,1));
  }

  inline double com_position(int molecule, int dim) const {
    return com_positions_(molecule, dim);
  }

  inline double com_velocity(int molecule, int dim) const {
    return com_velocities_(molecule, dim);
  }

  inline arma::rowvec com_velocity_xy(int molecule) const {
    return com_velocities_(molecule, arma::span(0,1));
  }

  inline double molecule_mass(int molecule) const {
    return molecular_index_to_mass_[molecule];
  }

 private:
  // Collects data from .gro file for all-atoms constructor.
  void ReadGro(const std::string& gro,
               const std::vector<Molecule> molecules);

  // Takes care of the atom-level removal for RemoveMolecule
  void RemoveAtom(const int index);

  // Takes care of the atom-level addition for AddMolecule
  void AddAtom(const Atom atom_to_add, const int molecule_id,
               const arma::rowvec& position, const arma::rowvec& velocity);

  std::string name_;
  std::vector<std::string> atom_names_;
  std::vector<std::string> molecule_names_;
  std::vector<double> index_to_mass_;
  std::vector<int> index_to_molecule_;
  std::vector<int> index_to_molecular_index_;
  std::vector<double> index_to_sigma_;
  std::vector<double> index_to_epsilon_;
  std::vector<double> molecular_index_to_mass_;
  std::vector<int>molecular_index_to_molecule_;
  int num_molecules_;
  arma::mat positions_;
  arma::mat velocities_;
  arma::mat com_positions_;
  arma::mat com_velocities_;
  std::vector<int> indices_;
  int group_size_;
};

#endif

// TODO(Zak): incorporate this function into the class
// extern int clusterCheck(const arma::mat& rii2, const double rminAC2,
//                         arma::irowvec& complete, const int numIons,
//                         const int which, int size);
