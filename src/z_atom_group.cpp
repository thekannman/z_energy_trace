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

// Implementation of AtomGroup class. See include/z_atom_group.hpp for
// more details about the class.

#include "z_atom_group.hpp"
#include <cassert>
#include <boost/lexical_cast.hpp>
#include "z_string.hpp"
#include <boost/format.hpp>

void AtomGroup::Init(std::string name, std::vector<double> index_to_mass,
                     std::vector<int> index_to_molecule,
                     std::vector<double> index_to_sigma,
                     std::vector<double> index_to_epsilon) {
  group_size_ = indices_.size();
  positions_ = arma::zeros(group_size_, DIMS);
  velocities_ = arma::zeros(group_size_, DIMS);
  for (std::vector<int>::iterator i_index = indices_.begin();
       i_index != indices_.end(); ++i_index) {
    index_to_mass_.push_back(index_to_mass[*i_index]);
    index_to_molecule_.push_back(index_to_molecule[*i_index]);
    index_to_sigma_.push_back(index_to_sigma[*i_index]);
    index_to_epsilon_.push_back(index_to_epsilon[*i_index]);
  }
  std::vector<double>::iterator i_mass = index_to_mass_.begin();
  for (std::vector<int>::iterator i_mol = index_to_molecule_.begin();
       i_mol != index_to_molecule_.end(); ++i_mol, ++i_mass) {
    std::vector<int>::iterator mol_index;
    mol_index = std::find(molecular_index_to_molecule_.begin(),
                          molecular_index_to_molecule_.end(), *i_mol);
  if (mol_index == molecular_index_to_molecule_.end()) {
    molecular_index_to_mass_.push_back(0.0);
    index_to_molecular_index_.push_back(molecular_index_to_molecule_.size());
    molecular_index_to_molecule_.push_back(*i_mol);
  } else
    index_to_molecular_index_.push_back(
        std::distance(molecular_index_to_molecule_.begin(),mol_index));
    molecular_index_to_mass_.back() += *i_mass;
  }
  num_molecules_ = molecular_index_to_molecule_.size();
  positions_.set_size(group_size_, DIMS);
  velocities_.set_size(group_size_, DIMS);
  com_positions_.set_size(num_molecules_, DIMS);
  com_velocities_.set_size(num_molecules_, DIMS);
}

void AtomGroup::RemoveMolecule(const int molecule_id,
                               arma::rowvec& old_com_position,
                               arma::rowvec& old_com_velocity) {
  std::vector<int> mark_for_rem;
  for (std::vector<int>::iterator i_mol = index_to_molecule_.begin();
       i_mol != index_to_molecule_.end(); i_mol++) {
    if (*i_mol == molecule_id)
    mark_for_rem.push_back(std::distance(index_to_molecule_.begin(),i_mol));
  }
  int marked_molecular_index = index_to_molecular_index_[mark_for_rem.front()];
  for (std::vector<int>::iterator i_atom = index_to_molecular_index_.begin();
       i_atom != index_to_molecular_index_.end(); ++i_atom) {
    // Decrements molecular indices to eliminate gap
    if ((*i_atom) > marked_molecular_index)
      (*i_atom) -= 1;
  }
  assert(!mark_for_rem.empty());
  molecular_index_to_mass_.erase(
      molecular_index_to_mass_.begin() +
      index_to_molecular_index_[mark_for_rem.front()]);
  molecular_index_to_molecule_.erase(
      molecular_index_to_molecule_.begin() +
      index_to_molecular_index_[mark_for_rem.front()]);
  // Save references to position/velocity for later replacement.
  old_com_position =
      com_positions_.row(index_to_molecular_index_[mark_for_rem.front()]);
  old_com_velocity =
      com_velocities_.row(index_to_molecular_index_[mark_for_rem.front()]);
  com_positions_.shed_row(index_to_molecular_index_[mark_for_rem.front()]);
  com_velocities_.shed_row(index_to_molecular_index_[mark_for_rem.front()]);
  for (std::vector<int>::reverse_iterator i_atom = mark_for_rem.rbegin();
       i_atom != mark_for_rem.rend(); i_atom++) {
    RemoveAtom(*i_atom);
  }
  num_molecules_--;
}


void AtomGroup::AddMolecule(const int molecule_id, const Molecule& molecule,
                            const arma::rowvec& position,
                            const arma::rowvec& velocity)
{
    // TODO(Zak): Allow for polyatomic molecules by randomly
    // orienting around center of mass position.
    // This assert is a placeholder until then.
    assert (molecule.num_atoms() == 1);
    molecule_names_.push_back(molecule.name());
    molecular_index_to_mass_.push_back(molecule.mass());
    molecular_index_to_molecule_.push_back(molecule_id);
    for (std::vector<Atom>::iterator i_atom = molecule.atoms().begin();
         i_atom != molecule.atoms().begin(); ++i_atom) {
      AddAtom(*i_atom, molecule_id, position, velocity);
    }
    num_molecules_++;
    com_positions_.insert_rows(com_positions_.n_rows, position);
    com_velocities_.insert_rows(com_velocities_.n_rows, velocity);
}

void AtomGroup::WriteGro(const std::string& gro,
    arma::rowvec box, std::string description) {
    std::ofstream gro_file;
    gro_file.open(gro.c_str());
    assert(gro_file.is_open());
    std::cout << description << std::endl;
    std::cout << group_size_ << std::endl;
    for (int i_atom = 0; i_atom < group_size_; i_atom++)
    {
        std::cout << boost::format("%5d%-5s%5s%5d%8.3f%8.3f%8.3f%8.3f%8.3f%8.3f") %
            index_to_molecular_index_[i_atom] %
            molecule_names_[i_atom] % atom_names_[i_atom] %
            indices_[i_atom] % positions_(i_atom,0) %
            positions_(i_atom,1) % positions_(i_atom,2) %
            velocities_(i_atom,0) % velocities_(i_atom,1) %
            velocities_(i_atom,2) << std::endl;
    }
    std::cout << box(0) << " " << box(1) << " " << box(2);
    gro_file.close();
}

void AtomGroup::UpdateCom() {
  ZeroCom();
  // Actually com_position*mass and com_velocity*mass
  for (int i_atom = 0; i_atom < group_size_; i_atom++) {
    com_positions_.row(index_to_molecular_index_[i_atom]) +=
        index_to_mass_[i_atom]*positions_.row(i_atom);
    com_velocities_.row(index_to_molecular_index_[i_atom]) +=
        index_to_mass_[i_atom]*velocities_.row(i_atom);
  }
}

void AtomGroup::ReadGro(const std::string& gro,
                        const std::vector<Molecule> molecules) {
  std::ifstream gro_file;
  std::string line;
  gro_file.open(gro.c_str());
  assert(gro_file.is_open());
  getline(gro_file, line);
  getline(gro_file, line);
  std::vector<double> index_to_mass;
  std::vector<int> index_to_molecule;
  std::vector<double> index_to_sigma;
  std::vector<double> index_to_epsilon;
  const int num_atoms = boost::lexical_cast<int>(Trim(line));
  Molecule mol_match;
  positions_.zeros(num_atoms,DIMS);
  velocities_.zeros(num_atoms,DIMS);
  int mol_number = std::numeric_limits<int>::max(), mol_number_old;
  for (int i_atom = 0; i_atom < num_atoms; i_atom++) {
    indices_.push_back(i_atom);
    std::string mol_name;
    assert(!gro_file.eof());
    getline(gro_file, line);
    std::vector<std::string> split_line = Split(line, ' ');
    std::istringstream iline(split_line[0]);
    mol_number_old = mol_number;
    iline >> mol_number >> mol_name;
    mol_number--;
    atom_names_.push_back(split_line[1]);
    if (mol_number_old != mol_number)
      molecule_names_.push_back(mol_name);
    index_to_molecule.push_back(mol_number);
    // Find matching molecule
    for (std::vector<Molecule>::const_iterator i_mol = molecules.begin();
         i_mol != molecules.end(); i_mol++) {
      if (mol_name == (*i_mol).name()) {
        mol_match = (*i_mol);
        break;
      }
    }
    // Find matching atom within molecule
    for (std::vector<Atom>::const_iterator i_at = mol_match.begin();
         i_at != mol_match.end(); ++i_at) {
      if (split_line[1] == (*i_at).name()) {
        index_to_mass.push_back((*i_at).mass());
        index_to_sigma.push_back((*i_at).sigma());
        index_to_epsilon.push_back((*i_at).epsilon());
        break;
      }
    }
    for (int i_dim = 0; i_dim<DIMS; i_dim++) {
      positions_(i_atom,i_dim) =
      boost::lexical_cast<double>(split_line[3+i_dim]);
    }
    if (split_line.size() == 9) {
      for (int i_dim = 0; i_dim<DIMS; i_dim++) {
        velocities_(i_atom,i_dim) =
        boost::lexical_cast<double>(split_line[6+i_dim]);
      }
    }
  }
  // Init takes care of initializing the rest of the class matrices
  // and vectors.
  Init("all", index_to_mass, index_to_molecule, index_to_sigma,
       index_to_epsilon);
  gro_file.close();
}

void AtomGroup::RemoveAtom(const int index) {
  indices_.erase(indices_.begin() + index);
  atom_names_.erase(atom_names_.begin() + index);
  index_to_mass_.erase(index_to_mass_.begin() + index);
  index_to_molecule_.erase(index_to_molecule_.begin() + index);
  index_to_sigma_.erase(index_to_sigma_.begin() + index);
  index_to_epsilon_.erase(index_to_epsilon_.begin() + index);
  index_to_molecular_index_.erase(
      index_to_molecular_index_.begin() + index);
  positions_.shed_row(index);
  velocities_.shed_row(index);
  group_size_--;
}

void AtomGroup::AddAtom(const Atom atom_to_add, const int molecule_id,
                        const arma::rowvec& position,
                        const arma::rowvec& velocity) {
  atom_names_.push_back(atom_to_add.name());
  index_to_mass_.push_back(atom_to_add.mass());
  index_to_molecule_.push_back(molecule_id);
  index_to_sigma_.push_back(atom_to_add.sigma());
  index_to_epsilon_.push_back(atom_to_add.epsilon());
  index_to_molecular_index_.push_back(num_molecules_);
  // TODO(Zak): Fix to work outside of all_atoms group.
  indices_.push_back(group_size_);
  group_size_++;
  positions_.insert_rows(positions_.n_rows, position);
  velocities_.insert_rows(velocities_.n_rows, velocity);
}

//TODO(Zak): incorporate this function into the class.
//int clusterCheck(const arma::mat& rii2, const double rminAC2,
//    arma::irowvec& complete, const int numIons, const int which, int size)
//{
//    for (int i1=0; i1<numIons; i1++)
//    {
//        if (complete(i1)) continue;
//        if (rii2(which,i1)>rminAC2)
//            continue;
//        else
//        {
//            complete(i1) = 1;
//            size++;
//            size = clusterCheck(rii2, rminAC2, complete, numIons, i1, size);
//        }
//    }
//    return size;
//}
