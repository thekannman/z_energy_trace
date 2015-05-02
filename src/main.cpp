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

#include "z_sim_params.hpp"
#include "z_string.hpp"
#include "z_vec.hpp"
#include "z_conversions.hpp"
#include "z_molecule.hpp"
#include "z_atom_group.hpp"
#include "z_gromacs.hpp"
#include "xdrfile_trr.h"
#include "boost/program_options.hpp"

namespace po = boost::program_options;

// Units are nm, ps.

int main (int argc, char *argv[]) {
  int st;
  SimParams params;
  int max_steps = std::numeric_limits<int>::max();
  enum TemperatureType {kNone, kMolCom, kAtom};

  double gdsTop = 3.2, gdsBottom = 8.75;
  std::string when_filename;

  po::options_description desc("Options");
  desc.add_options()
    ("help,h",  "Print help messages")
    ("group,g", po::value<std::string>()->default_value("He"),
     "Group for temperature profiles")
    ("liquid,l", po::value<std::string>()->default_value("OW"),
     "Group to use for calculation of surface")
    ("index,n", po::value<std::string>()->default_value("index.ndx"),
     ".ndx file containing atomic indices for groups")
    ("gro", po::value<std::string>()->default_value("conf.gro"),
     ".gro file containing list of atoms/molecules")
    ("top", po::value<std::string>()->default_value("topol.top"),
     ".top file containing atomic/molecular properties")
    ("temperature_type,T", po::value<std::string>()->default_value("none"),
     "Use atom or mol_com velocities for temperature profile")
    ("max_time,t",
     po::value<double>()->default_value(std::numeric_limits<double>::max()),
     "Maximum simulation time to use in calculations")
    ("split,s", "Use splitting from liquid as desorption event");

  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);

  if (vm.count("help")) {
    std::cout << desc << "\n";
    exit(EXIT_SUCCESS);
  }

  TemperatureType temperature_type;
  const std::string& vm_temperature_type =
      vm["temperature_type"].as<std::string>();
  if (vm_temperature_type == "mol_com")
    temperature_type = kMolCom;
  else if (vm_temperature_type == "atom")
    temperature_type = kAtom;
  else if (vm_temperature_type == "none" )
    temperature_type = kNone;
  else
    assert(false && "Unrecognized temperature_type option");


  std::map<std::string, std::vector<int> > groups;
  groups = ReadNdx(vm["index"].as<std::string>());

  std::vector<Molecule> molecules = GenMolecules(vm["top"].as<std::string>(),
                                                 params);
  AtomGroup all_atoms(vm["gro"].as<std::string>(), molecules);
  AtomGroup selected_group(vm["group"].as<std::string>(),
                           SelectGroup(groups, vm["group"].as<std::string>()),
                           all_atoms);
  AtomGroup liquid_group(vm["liquid"].as<std::string>(),
                         SelectGroup(groups, vm["liquid"].as<std::string>()),
                         all_atoms);

  bool split = vm.count("split") ? true : false;

  arma::irowvec when = arma::zeros<arma::irowvec >(selected_group.size());
  arma::irowvec stop_tracking =
    arma::zeros<arma::irowvec >(selected_group.size());
  rvec *x_in = NULL;
  matrix box_mat;
  arma::rowvec box = arma::zeros<arma::rowvec>(DIMS);
  std::string xtc_filename = "prod.xtc";
  std::string trr_filename = "prod.trr";
  XDRFILE *xtc_file, *trr_file;
  params.ExtractTrajMetadata(strdup(xtc_filename.c_str()), (&x_in), box);
  trr_file = xdrfile_open(strdup(trr_filename.c_str()), "r");
  xtc_file = xdrfile_open(strdup(xtc_filename.c_str()), "r");
  params.set_max_time(vm["max_time"].as<double>());
  double bin_width = 0.1;
  int num_bins = (int)(9.0/bin_width);
  if (split)
    when_filename = "whenSplit.dat";
  else
    when_filename = "when.dat";
  std::ifstream when_file;
  when_file.open(when_filename.c_str());
  assert(when_file.is_open());
  for (int i = 0; i < selected_group.size(); i++)
    when_file >> when(i);
  when_file.close();

  const int kStepsBefore = 100/params.dt();
  const int kStepsAfter = 50/params.dt();
  const int kStepsTotal = kStepsBefore + kStepsAfter;
  const double kUpperCutoff = 7.0;
  const double kLowerCutoff = 5.0;
  const double kCutoffRegionSize = 1.0;
  arma::rowvec tempxyTimeProfile = arma::zeros<arma::rowvec>(kStepsTotal);
  arma::rowvec tempzTimeProfile = arma::zeros<arma::rowvec>(kStepsTotal);
  arma::rowvec tempxySpaceProfile = arma::zeros<arma::rowvec>(num_bins);
  arma::rowvec tempzSpaceProfile = arma::zeros<arma::rowvec>(num_bins);
  arma::rowvec potTimeProfile = arma::zeros<arma::rowvec>(kStepsTotal);
  arma::rowvec forcexyTimeProfile = arma::zeros<arma::rowvec>(kStepsTotal);
  arma::rowvec forcezTimeProfile = arma::zeros<arma::rowvec>(kStepsTotal);
  arma::rowvec potSpaceProfile = arma::zeros<arma::rowvec>(num_bins);
  arma::rowvec forcezSpaceProfile = arma::zeros<arma::rowvec>(num_bins);
  arma::rowvec forcexySpaceProfile = arma::zeros<arma::rowvec>(num_bins);
  arma::irowvec countTime = arma::zeros<arma::irowvec>(kStepsTotal);
  arma::irowvec countSpace = arma::zeros<arma::irowvec>(num_bins);
  trr_file = xdrfile_open(strdup(trr_filename.c_str()), "r");
  xtc_file = xdrfile_open(strdup(xtc_filename.c_str()), "r");
  rvec *v_in = NULL;
  v_in = new rvec [params.num_atoms()];
  float time, lambda, prec;
  for (int step=0, steps=0; step<max_steps; step++) {
    steps++;
    if(read_xtc(xtc_file, params.num_atoms(), &st, &time, box_mat, x_in,
                &prec)) {
      break;
    }
    if(read_trr(trr_file, params.num_atoms(), &st, &time, &lambda, box_mat,
                NULL, v_in, NULL)) {
      break;
    }
    int i = 0;
    for (std::vector<int>::iterator i_atom = selected_group.begin();
         i_atom != selected_group.end(); i_atom++, i++) {
      selected_group.set_position(i, x_in[*i_atom]);
      selected_group.set_velocity(i, v_in[*i_atom]);
    }
    i = 0;
    for (std::vector<int>::iterator i_atom = liquid_group.begin();
         i_atom != liquid_group.end(); ++i_atom, ++i) {
      liquid_group.set_position(i, x_in[*i_atom]);
    }
    arma::rowvec dx;
    const int MIN_STEP = 100;
    if (step > MIN_STEP) {
      for (int i_atom = 0; i_atom < selected_group.size(); i_atom++) {
        if((when(i_atom) == 0) || (stop_tracking(i_atom) != 0))
          continue;
        int shifted_step = step-when(i_atom)+kStepsBefore;
        if(shifted_step<0 || shifted_step>=kStepsTotal) continue;
        double mass = selected_group.mass(i_atom);
        double x_vel = selected_group.velocity(i_atom,0);
        double y_vel = selected_group.velocity(i_atom,1);
        double z_vel = selected_group.velocity(i_atom,2);
        if (z_vel > 0.0) {
          if (selected_group.position(i_atom,2) > kUpperCutoff &&
              (selected_group.position(i_atom,2) < kUpperCutoff +
               kCutoffRegionSize)) {
            stop_tracking(i_atom) = 1;
          }
        } else {
          if (selected_group.position(i_atom,2) < kLowerCutoff &&
              (selected_group.position(i_atom,2) > kLowerCutoff -
               kCutoffRegionSize)) {
            stop_tracking(i_atom) = 1;
          }
        }

        double xy_temp = mass * (x_vel*x_vel + y_vel*y_vel);
        double z_temp = mass * z_vel * z_vel;
        // TODO(Zak): hard-coding potential/force calculation for now,
        //            will fix later.
        double potential = 0.0;
        double xy_force = 0.0;
        double z_force = 0.0;
        for (int i_liq = 0; i_liq < liquid_group.size(); i_liq++) {
          FindDxNoShift(dx, selected_group.position(i_atom),
                        liquid_group.position(i_liq), box);
          double r2 = arma::dot(dx,dx);
          dx = arma::normalise(dx);
          if (r2 > 1.0) continue;
          double sigma =
              (selected_group.sigma(i_atom)+liquid_group.sigma(i_liq))/2.0;
          double sigma2 = sigma*sigma;
          double epsilon = sqrt(selected_group.epsilon(i_atom)*
                                liquid_group.epsilon(i_liq));
          double r6_term = std::pow(sigma2/r2,3.0);
          double r12_term = std::pow(r6_term,2.0);
          potential += 4.0*epsilon*(r12_term - r6_term);
          double force_term = 48.0*epsilon/r2*(r12_term - 0.5*r6_term);
          xy_force += (dx(0)+dx(1))*force_term;
          z_force += dx(2)*force_term;
        }
        // Stop hard-coded temp here.
        tempxyTimeProfile(shifted_step) += xy_temp;
        tempzTimeProfile(shifted_step) += z_temp;
        potTimeProfile(shifted_step) += potential;
        forcexyTimeProfile(shifted_step) += xy_force;
        forcezTimeProfile(shifted_step) += z_force;
        int which_bin =
          static_cast<int>(selected_group.position(i_atom,2)/box(2)*num_bins);
        if(which_bin == num_bins)
          which_bin = 0;
        tempxySpaceProfile(which_bin) += xy_temp;
        tempzSpaceProfile(which_bin) += z_temp;
        potSpaceProfile(which_bin) += potential;
        forcexySpaceProfile(which_bin) += xy_force;
        forcezSpaceProfile(which_bin) += z_force;
        countTime(shifted_step)++;
        countSpace(which_bin)++;
      }
    }
  }
  xdrfile_close(xtc_file);
  xdrfile_close(trr_file);
  for (int i=0; i<kStepsTotal; i++) {
    if (countTime(i)) {
      tempxyTimeProfile(i) *=
          AMU_TO_KG*NM_TO_M/PS_TO_S*NM_TO_M/PS_TO_S/countTime(i)/2.0/KB;
      tempzTimeProfile(i) *=
          AMU_TO_KG*NM_TO_M/PS_TO_S*NM_TO_M/PS_TO_S/countTime(i)/KB;
      potTimeProfile(i) *= 1.0/countTime(i);
      forcexyTimeProfile(i) *= 1.0/countTime(i)/2.0;
      forcezTimeProfile(i) *= 1.0/countTime(i);
    }
  }

  std::string xy_temperature_time_filename =
      vm["group"].as<std::string>() + "_xytempTimeSplit.dat";
  std::ofstream xy_temperature_time_file;
  xy_temperature_time_file.open(xy_temperature_time_filename.c_str());
  for (int i=0; i<kStepsTotal; i++) {
    xy_temperature_time_file << (i-kStepsBefore)*params.dt() << " " <<
    tempxyTimeProfile(i) << std::endl;
  }
  xy_temperature_time_file.close();

  std::string z_temperature_time_filename =
      vm["group"].as<std::string>() + "_ztempTimeSplit.dat";
  std::ofstream z_temperature_time_file;
  z_temperature_time_file.open(z_temperature_time_filename.c_str());
  for (int i=0; i<kStepsTotal; i++) {
    z_temperature_time_file << (i-kStepsBefore)*params.dt() << " " <<
    tempzTimeProfile(i) << std::endl;
  }
  z_temperature_time_file.close();

  std::string potential_time_filename = vm["liquid"].as<std::string>() +
      "_To_" + vm["group"].as<std::string>() + "_potTimeSplit.dat";
  std::ofstream potential_time_file;
  potential_time_file.open(potential_time_filename.c_str());
  for (int i=0; i<kStepsTotal; i++) {
    potential_time_file << (i-kStepsBefore)*params.dt() << " " <<
    potTimeProfile(i) << std::endl;
  }
  potential_time_file.close();

  std::string xy_force_time_filename = vm["liquid"].as<std::string>() +
      "_To_" + vm["group"].as<std::string>() + "_xyforceTimeSplit.dat";
  std::ofstream xy_force_time_file;
  xy_force_time_file.open(xy_force_time_filename.c_str());
  for (int i=0; i<kStepsTotal; i++) {
    xy_force_time_file << (i-kStepsBefore)*params.dt() << " " <<
    forcexyTimeProfile(i) << std::endl;
  }
  xy_force_time_file.close();

  std::string z_force_time_filename = vm["liquid"].as<std::string>() +
      "_To_" + vm["group"].as<std::string>() + "_zforceTimeSplit.dat";
  std::ofstream z_force_time_file;
  z_force_time_file.open(z_force_time_filename.c_str());
  for (int i=0; i<kStepsTotal; i++) {
    z_force_time_file << (i-kStepsBefore)*params.dt() << " " <<
    forcezTimeProfile(i) << std::endl;
  }
  z_force_time_file.close();

  for (int i=0; i<num_bins; i++) {
    if (countSpace(i)) {
      tempxySpaceProfile(i) *= AMU_TO_KG*1000*1000/countSpace(i)/2.0/KB;
      tempzSpaceProfile(i) *= AMU_TO_KG*1000*1000/countSpace(i)/1.0/KB;
      potSpaceProfile(i) *= 1.0/countSpace(i);
    }
  }

  std::string xy_temperature_space_filename =
      vm["group"].as<std::string>() + "_xytempSpaceSplit.dat";
  std::ofstream xy_temperature_space_file;
  xy_temperature_space_file.open(xy_temperature_space_filename.c_str());
  for (int i=0; i<num_bins; i++) {
    xy_temperature_space_file << (i+0.5)*bin_width << " " <<
    tempxySpaceProfile(i) << std::endl;
  }
  xy_temperature_space_file.close();

  std::string z_temperature_space_filename =
      vm["group"].as<std::string>() + "_ztempSpaceSplit.dat";
  std::ofstream z_temperature_space_file;
  z_temperature_space_file.open(z_temperature_space_filename.c_str());
  for (int i=0; i<num_bins; i++) {
    z_temperature_space_file << (i+0.5)*bin_width << " " <<
    tempzSpaceProfile(i) << std::endl;
  }
  z_temperature_space_file.close();

  std::string potential_space_filename = vm["liquid"].as<std::string>() +
      "_To_" + vm["group"].as<std::string>() + "_potSpaceSplit.dat";
  std::ofstream potential_space_file;
  potential_space_file.open(potential_space_filename.c_str());
  for (int i=0; i<num_bins; i++) {
    potential_space_file << (i+0.5)*bin_width << " " <<
    potSpaceProfile(i) << std::endl;
  }
  potential_space_file.close();

  std::string xy_force_space_filename = vm["liquid"].as<std::string>() +
      "_To_" + vm["group"].as<std::string>() + "_xyforceSpaceSplit.dat";
  std::ofstream xy_force_space_file;
  xy_force_space_file.open(xy_force_space_filename.c_str());
  for (int i=0; i<num_bins; i++) {
    xy_force_space_file << (i+0.5)*bin_width << " " <<
    forcexySpaceProfile(i) << std::endl;
  }
  xy_force_space_file.close();

  std::string z_force_space_filename = vm["liquid"].as<std::string>() +
      "_To_" + vm["group"].as<std::string>() + "_zforceSpaceSplit.dat";
  std::ofstream z_force_space_file;
  z_force_space_file.open(z_force_space_filename.c_str());
  for (int i=0; i<num_bins; i++) {
    z_force_space_file << (i+0.5)*bin_width << " " <<
    forcezSpaceProfile(i) << std::endl;
  }
  z_force_space_file.close();

} // main
