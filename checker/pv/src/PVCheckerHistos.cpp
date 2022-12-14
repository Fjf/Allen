/*****************************************************************************\
* (c) Copyright 2018-2020 CERN for the benefit of the LHCb Collaboration      *
\*****************************************************************************/
#include "PVCheckerHistos.h"

namespace {
  double binomial_error(double k, double N) { return sqrt(k * (1 - k / N)) / N; }
} // namespace

PVCheckerHistos::PVCheckerHistos(
  CheckerInvoker const* invoker,
  std::string const& root_file,
  std::string const& directory) :
  m_directory {directory}
{
  m_file = invoker->root_file(root_file);
  auto* dir = static_cast<TDirectory*>(m_file->Get(m_directory.c_str()));
  if (!dir) {
    dir = m_file->mkdir(m_directory.c_str());
    dir = static_cast<TDirectory*>(m_file->Get(m_directory.c_str()));
  }
  dir->cd();

  eff_vs_z = std::make_unique<TH1F>("eff_vs_z", "eff_vs_z", m_bins_norm_z, -300, 300);
  eff_matched_vs_z = std::make_unique<TH1F>("eff_matched_z", "eff_matched_z", m_bins_norm_z, -300, 300);
  eff_vs_mult = std::make_unique<TH1F>("eff_vs_mult", "eff_vs_mult", m_bins_norm_mult, 0, 50);
  eff_matched_vs_mult = std::make_unique<TH1F>("eff_matched_mult", "eff_matched_mult", m_bins_norm_mult, 0, 50);
  eff_norm_z = std::make_unique<TH1F>("eff_norm_z", "eff_norm_z", m_bins_norm_z, -300, 300);
  eff_norm_mult = std::make_unique<TH1F>("eff_norm_mult", "eff_norm_mult", m_bins_norm_mult, 0, 50);
  fakes_vs_mult = std::make_unique<TH1F>("fakes_vs_mult", "fakes_vs_mult", m_bins_fake_mult, 0, 20);
  fakes_norm = std::make_unique<TH1F>("fakes_norm_mult", "fakes_norm_mult", m_bins_fake_mult, 0, 20);

  auto make_tree = [dir = m_directory](const std::string& name) {
    return std::make_unique<TTree>(name.c_str(), name.c_str());
  };

  m_tree = make_tree("PV_tree");
  m_tree->Branch("nmcpv", &m_nmcpv);
  m_tree->Branch("ntrinmcpv", &m_ntrinmcpv);

  m_tree->Branch("diff_x", &m_diff_x);
  m_tree->Branch("diff_y", &m_diff_y);
  m_tree->Branch("diff_z", &m_diff_z);
  m_tree->Branch("rec_x", &m_rec_x);
  m_tree->Branch("rec_y", &m_rec_y);
  m_tree->Branch("rec_z", &m_rec_z);

  m_tree->Branch("err_x", &m_err_x);
  m_tree->Branch("err_y", &m_err_y);
  m_tree->Branch("err_z", &m_err_z);

  m_mctree = make_tree("MC_tree");
  m_mctree->Branch("x", &m_mc_x);
  m_mctree->Branch("y", &m_mc_y);
  m_mctree->Branch("z", &m_mc_z);

  m_allPV = make_tree("allPV");
  m_allPV->Branch("x", &m_x);
  m_allPV->Branch("y", &m_y);
  m_allPV->Branch("z", &m_z);
  m_allPV->Branch("errx", &m_errx);
  m_allPV->Branch("erry", &m_erry);
  m_allPV->Branch("errz", &m_errz);
  m_allPV->Branch("isFake", &m_isFake);
}

void PVCheckerHistos::accumulate(
  gsl::span<const RecPVInfo> vec_all_rec,
  gsl::span<const double> vec_rec_x,
  gsl::span<const double> vec_rec_y,
  gsl::span<const double> vec_rec_z,
  gsl::span<const double> vec_diff_x,
  gsl::span<const double> vec_diff_y,
  gsl::span<const double> vec_diff_z,
  gsl::span<const double> vec_err_x,
  gsl::span<const double> vec_err_y,
  gsl::span<const double> vec_err_z,
  gsl::span<const int> vec_n_trinmcpv,
  gsl::span<const int> vec_n_mcpv,
  gsl::span<const int> vec_mcpv_recd,
  gsl::span<const int> vec_recpv_fake,
  gsl::span<const int> vec_mcpv_mult,
  gsl::span<const int> vec_recpv_mult,
  gsl::span<const double> vec_mcpv_zpos,
  gsl::span<const double> vec_mc_x,
  gsl::span<const double> vec_mc_y,
  gsl::span<const double> vec_mc_z)
{
  // save information about matched reconstructed PVs for pulls distributions
  for (size_t i = 0; i < vec_diff_x.size(); i++) {
    m_nmcpv = vec_n_mcpv[i];
    m_ntrinmcpv = vec_n_trinmcpv[i];
    m_diff_x = vec_diff_x[i];
    m_diff_y = vec_diff_y[i];
    m_diff_z = vec_diff_z[i];
    m_rec_x = vec_rec_x[i];
    m_rec_y = vec_rec_y[i];
    m_rec_z = vec_rec_z[i];

    m_err_x = vec_err_x[i];
    m_err_y = vec_err_y[i];
    m_err_z = vec_err_z[i];

    m_tree->Fill();
  }

  for (size_t i = 0; i < vec_recpv_mult.size(); i++) {
    fakes_vs_mult->Fill(vec_recpv_mult[i], vec_recpv_fake[i]);
    fakes_norm->Fill(vec_recpv_mult[i], 1);
  }

  for (size_t i = 0; i < vec_mcpv_mult.size(); i++) {
    eff_vs_z->Fill(vec_mcpv_zpos[i], vec_mcpv_recd[i]);
    eff_vs_mult->Fill(vec_mcpv_mult[i], vec_mcpv_recd[i]);
    if (vec_mcpv_recd[i]) {
      eff_matched_vs_z->Fill(vec_mcpv_zpos[i]);
      eff_matched_vs_mult->Fill(vec_mcpv_mult[i]);
    }
    eff_norm_z->Fill(vec_mcpv_zpos[i], 1);
    eff_norm_mult->Fill(vec_mcpv_mult[i], 1);
  }

  std::vector<double> binerrors_vs_z;
  std::vector<double> binerrors_vs_mult;

  // Proper uncertainties for efficiencies
  for (int i = 1; i <= m_bins_norm_z; i++) {
    auto N = eff_norm_z->GetBinContent(i);
    auto k = eff_vs_z->GetBinContent(i);
    if (k < N && N > 0) {
      binerrors_vs_z.push_back(binomial_error(k, N));
    }
    else
      binerrors_vs_z.push_back(0.);
  }
  for (int i = 1; i <= m_bins_norm_mult; i++) {
    auto N = eff_norm_mult->GetBinContent(i);
    auto k = eff_vs_mult->GetBinContent(i);
    if (k < N && N > 0) {
      binerrors_vs_mult.push_back(binomial_error(k, N));
    }
    else
      binerrors_vs_mult.push_back(0.);
  }

  eff_vs_z->Divide(eff_norm_z.get());
  for (int i = 1; i <= m_bins_norm_z; i++) {
    eff_vs_z->SetBinError(i, binerrors_vs_z[i - 1]);
  }
  eff_vs_mult->Divide(eff_norm_mult.get());
  for (int i = 1; i <= m_bins_norm_mult; i++) {
    eff_vs_mult->SetBinError(i, binerrors_vs_mult[i - 1]);
  }
  fakes_vs_mult->Divide(fakes_norm.get());

  for (size_t j = 0; j < vec_mc_x.size(); j++) {
    m_mc_x = vec_mc_x[j];
    m_mc_y = vec_mc_y[j];
    m_mc_z = vec_mc_z[j];
    m_mctree->Fill();
  }

  for (auto rec_pv : vec_all_rec) {
    m_x = rec_pv.x;
    m_y = rec_pv.y;
    m_z = rec_pv.z;
    m_errx = rec_pv.positionSigma.x;
    m_erry = rec_pv.positionSigma.y;
    m_errz = rec_pv.positionSigma.z;
    m_isFake = rec_pv.indexMCPVInfo < 0;
    m_allPV->Fill();
  }
}

void PVCheckerHistos::write()
{
  auto* dir = static_cast<TDirectory*>(m_file->Get(m_directory.c_str()));
  std::tuple to_write {std::ref(m_tree),
                       std::ref(m_mctree),
                       std::ref(m_allPV),
                       std::ref(eff_vs_z),
                       std::ref(eff_vs_mult),
                       std::ref(eff_matched_vs_z),
                       std::ref(eff_matched_vs_mult),
                       std::ref(eff_norm_z),
                       std::ref(eff_norm_mult),
                       std::ref(fakes_vs_mult),
                       std::ref(fakes_norm)};
  for_each(to_write, [dir](auto& o) {
    o.get()->SetDirectory(nullptr);
    dir->WriteTObject(o.get().get());
  });
}
