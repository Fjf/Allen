/*****************************************************************************\
* (c) Copyright 2018-2020 CERN for the benefit of the LHCb Collaboration      *
\*****************************************************************************/
#pragma once

#include <cstdio>
#include <math.h>
#include "TrackChecker.h"
#include "CheckerTypes.h"

namespace {
  using Checker::HistoCategory;
  using Checker::TrackEffReport;
} // namespace

namespace Categories {
  template<typename T>
  inline std::vector<TrackEffReport> make_track_eff_report_vector();

  template<typename T>
  inline std::vector<HistoCategory> make_histo_category_vector();

  template<>
  inline std::vector<TrackEffReport> make_track_eff_report_vector<Checker::Subdetector::Velo>()
  {
    return std::vector<TrackEffReport> {
      {// define which categories to monitor
       TrackEffReport({
         "01_velo",
         [](MCParticles::const_reference& mcp) { return mcp.hasVelo && !mcp.isElectron() && mcp.inEta2_5(); },
       }),
       TrackEffReport({
         "02_long",
         [](MCParticles::const_reference& mcp) { return mcp.isLong && !mcp.isElectron() && mcp.inEta2_5(); },
       }),
       TrackEffReport({
         "03_long_P>5GeV",
         [](MCParticles::const_reference& mcp) {
           return mcp.isLong && !mcp.isElectron() && mcp.p > 5e3f && mcp.inEta2_5();
         },
       }),
       TrackEffReport({
         "04_long_strange",
         [](MCParticles::const_reference& mcp) {
           return mcp.isLong && !mcp.isElectron() && mcp.fromStrangeDecay && mcp.inEta2_5();
         },
       }),
       TrackEffReport({
         "05_long_strange_P>5GeV",
         [](MCParticles::const_reference& mcp) {
           return mcp.isLong && !mcp.isElectron() && mcp.fromStrangeDecay && mcp.p > 5e3f && mcp.inEta2_5();
         },
       }),
       TrackEffReport({
         "06_long_fromB",
         [](MCParticles::const_reference& mcp) {
           return mcp.isLong && !mcp.isElectron() && mcp.fromBeautyDecay && mcp.inEta2_5();
         },
       }),
       TrackEffReport({
         "07_long_fromB_P>5GeV",
         [](MCParticles::const_reference& mcp) {
           return mcp.isLong && !mcp.isElectron() && mcp.fromBeautyDecay && mcp.p > 5e3f && mcp.inEta2_5();
         },
       }),
       TrackEffReport({
         "08_long_electrons",
         [](MCParticles::const_reference& mcp) { return mcp.isLong && mcp.isElectron() && mcp.inEta2_5(); },
       }),
       TrackEffReport({
         "09_long_fromB_electrons",
         [](MCParticles::const_reference& mcp) {
           return mcp.isLong && mcp.isElectron() && mcp.fromBeautyDecay && mcp.inEta2_5();
         },
       }),
       TrackEffReport({
         "10_long_fromB_electrons_P>5GeV",
         [](MCParticles::const_reference& mcp) {
           return mcp.isLong && mcp.isElectron() && mcp.fromBeautyDecay && mcp.p > 5e3f && mcp.inEta2_5();
         },
       }),
       TrackEffReport({
         "11_long_fromSignal",
         [](MCParticles::const_reference& mcp) { return mcp.isLong && mcp.fromSignal && mcp.inEta2_5(); },
       })}};
  }

  /* const std::vector<TrackEffReport>& Velo { */
  /*   {// define which categories to monitor */
  /*    TrackEffReport({ */
  /*      "Electrons long eta25", */
  /*      [](MCParticles::const_reference& mcp) { return mcp.isLong && mcp.isElectron() && mcp.inEta2_5(); }, */
  /*    }), */
  /*    TrackEffReport({ */
  /*      "Electrons long fromB eta25", */
  /*      [](MCParticles::const_reference& mcp) { */
  /*        return mcp.isLong && mcp.fromBeautyDecay && mcp.isElectron() && mcp.inEta2_5(); */
  /*      }, */
  /*    }), */
  /*    TrackEffReport({ */
  /*      "Electrons long fromB eta25 p<5GeV", */
  /*      [](MCParticles::const_reference& mcp) { */
  /*        return mcp.isLong && mcp.fromBeautyDecay && mcp.isElectron() && mcp.inEta2_5() && mcp.p < 5e3f; */
  /*      }, */
  /*    }), */
  /*    TrackEffReport({ */
  /*      "Electrons long fromB eta25 p>3GeV pt>400MeV", */
  /*      [](MCParticles::const_reference& mcp) { */
  /*        return mcp.isLong && mcp.fromBeautyDecay && mcp.isElectron() && mcp.inEta2_5() && mcp.p > 3e3f && */
  /*               mcp.pt > 400.f; */
  /*      }, */
  /*    }), */
  /*    TrackEffReport({ */
  /*      "Electrons long fromB eta25 p>5GeV", */
  /*      [](MCParticles::const_reference& mcp) { */
  /*        return mcp.isLong && mcp.fromBeautyDecay && mcp.isElectron() && mcp.inEta2_5() && mcp.p > 5e3f; */
  /*      }, */
  /*    }), */
  /*    TrackEffReport({ */
  /*      "Electrons long fromD eta25", */
  /*      [](MCParticles::const_reference& mcp) { */
  /*        return mcp.isLong && mcp.fromCharmDecay && mcp.isElectron() && mcp.inEta2_5(); */
  /*      }, */
  /*    }), */
  /*    TrackEffReport({ */
  /*      "Electrons long fromD eta25 p<5GeV", */
  /*      [](MCParticles::const_reference& mcp) { */
  /*        return mcp.isLong && mcp.fromCharmDecay && mcp.isElectron() && mcp.inEta2_5() && mcp.p < 5e3f; */
  /*      }, */
  /*    }), */
  /*    TrackEffReport({ */
  /*      "Electrons long fromD eta25 p>3GeV pt>400MeV", */
  /*      [](MCParticles::const_reference& mcp) { */
  /*        return mcp.isLong && mcp.fromCharmDecay && mcp.isElectron() && mcp.inEta2_5() && mcp.p > 3e3f && */
  /*               mcp.pt > 400.f; */
  /*      }, */
  /*    }), */
  /*    TrackEffReport({ */
  /*      "Electrons long fromD eta25 p>5GeV", */
  /*      [](MCParticles::const_reference& mcp) { */
  /*        return mcp.isLong && mcp.fromCharmDecay && mcp.isElectron() && mcp.inEta2_5() && mcp.p > 5e3f; */
  /*      }, */
  /*    }), */
  /*    TrackEffReport({ */
  /*      "Electrons long eta25 p<5GeV", */
  /*      [](MCParticles::const_reference& mcp) { */
  /*        return mcp.isLong && mcp.isElectron() && mcp.inEta2_5() && mcp.p < 5e3f; */
  /*      }, */
  /*    }), */
  /*    TrackEffReport({ */
  /*      "Electrons long eta25 p>3GeV pt>400MeV", */
  /*      [](MCParticles::const_reference& mcp) { */
  /*        return mcp.isLong && mcp.isElectron() && mcp.inEta2_5() && mcp.p > 3e3f && mcp.pt > 400.f; */
  /*      }, */
  /*    }), */
  /*    TrackEffReport({ */
  /*      "Electrons long eta25 p>5GeV", */
  /*      [](MCParticles::const_reference& mcp) { */
  /*        return mcp.isLong && mcp.isElectron() && mcp.inEta2_5() && mcp.p > 5e3f; */
  /*      }, */
  /*    }), */
  /*    TrackEffReport({ */
  /*      "Electrons long strange eta25", */
  /*      [](MCParticles::const_reference& mcp) { */
  /*        return mcp.isLong && mcp.fromStrangeDecay && mcp.isElectron() && mcp.inEta2_5(); */
  /*      }, */
  /*    }), */
  /*    TrackEffReport({ */
  /*      "Electrons long strange eta25 p<5GeV", */
  /*      [](MCParticles::const_reference& mcp) { */
  /*        return mcp.isLong && mcp.fromStrangeDecay && mcp.isElectron() && mcp.inEta2_5() && mcp.p < 5e3f; */
  /*      }, */
  /*    }), */
  /*    TrackEffReport({ */
  /*      "Electrons long strange eta25 p>3GeV pt>400MeV", */
  /*      [](MCParticles::const_reference& mcp) { */
  /*        return mcp.isLong && mcp.fromStrangeDecay && mcp.isElectron() && mcp.inEta2_5() && mcp.p > 3e3f && */
  /*               mcp.pt > 400.f; */
  /*      }, */
  /*    }), */
  /*    TrackEffReport({ */
  /*      "Electrons long strange eta25 p>5GeV", */
  /*      [](MCParticles::const_reference& mcp) { */
  /*        return mcp.isLong && mcp.fromStrangeDecay && mcp.isElectron() && mcp.inEta2_5() && mcp.p > 5e3f; */
  /*      }, */
  /*    }), */
  /*    TrackEffReport({ */
  /*      "Electrons Velo", */
  /*      [](MCParticles::const_reference& mcp) { return mcp.hasVelo && mcp.isElectron(); }, */
  /*    }), */
  /*    TrackEffReport({ */
  /*      "Electrons Velo backward", */
  /*      [](MCParticles::const_reference& mcp) { return mcp.hasVelo && mcp.isElectron() && mcp.eta < 0; }, */
  /*    }), */
  /*    TrackEffReport({ */
  /*      "Electrons Velo forward", */
  /*      [](MCParticles::const_reference& mcp) { return mcp.hasVelo && mcp.isElectron() && mcp.eta > 0; }, */
  /*    }), */
  /*    TrackEffReport({ */
  /*      "Electrons Velo eta25", */
  /*      [](MCParticles::const_reference& mcp) { return mcp.hasVelo && mcp.isElectron() && mcp.inEta2_5(); }, */
  /*    }), */
  /*    TrackEffReport({ */
  /*      "Not electron long eta25", */
  /*      [](MCParticles::const_reference& mcp) { return mcp.isLong && !mcp.isElectron() && mcp.inEta2_5(); }, */
  /*    }), */
  /*    TrackEffReport({ */
  /*      "Not electron long fromB eta25", */
  /*      [](MCParticles::const_reference& mcp) { */
  /*        return mcp.isLong && mcp.fromBeautyDecay && !mcp.isElectron() && mcp.inEta2_5(); */
  /*      }, */
  /*    }), */
  /*    TrackEffReport({ */
  /*      "Not electron long fromB eta25 p<5GeV", */
  /*      [](MCParticles::const_reference& mcp) { */
  /*        return mcp.isLong && mcp.fromBeautyDecay && !mcp.isElectron() && mcp.inEta2_5() && mcp.p < 5e3f; */
  /*      }, */
  /*    }), */
  /*    TrackEffReport({ */
  /*      "Not electron long fromB eta25 p>3GeV pt>400MeV", */
  /*      [](MCParticles::const_reference& mcp) { */
  /*        return mcp.isLong && mcp.fromBeautyDecay && !mcp.isElectron() && mcp.inEta2_5() && mcp.p > 3e3f && */
  /*               mcp.pt > 400.f; */
  /*      }, */
  /*    }), */
  /*    TrackEffReport({ */
  /*      "Not electron long fromB eta25 p>5GeV", */
  /*      [](MCParticles::const_reference& mcp) { */
  /*        return mcp.isLong && mcp.fromBeautyDecay && !mcp.isElectron() && mcp.inEta2_5() && mcp.p > 5e3f; */
  /*      }, */
  /*    }), */
  /*    TrackEffReport({ */
  /*      "Not electron long fromD eta25", */
  /*      [](MCParticles::const_reference& mcp) { */
  /*        return mcp.isLong && mcp.fromCharmDecay && !mcp.isElectron() && mcp.inEta2_5(); */
  /*      }, */
  /*    }), */
  /*    TrackEffReport({ */
  /*      "Not electron long fromD eta25 p<5GeV", */
  /*      [](MCParticles::const_reference& mcp) { */
  /*        return mcp.isLong && mcp.fromCharmDecay && !mcp.isElectron() && mcp.inEta2_5() && mcp.p < 5e3f; */
  /*      }, */
  /*    }), */
  /*    TrackEffReport({ */
  /*      "Not electron long fromD eta25 p>3GeV pt>400MeV", */
  /*      [](MCParticles::const_reference& mcp) { */
  /*        return mcp.isLong && mcp.fromCharmDecay && !mcp.isElectron() && mcp.inEta2_5() && mcp.p > 3e3f && */
  /*               mcp.pt > 400.f; */
  /*      }, */
  /*    }), */
  /*    TrackEffReport({ */
  /*      "Not electron long fromD eta25 p>5GeV", */
  /*      [](MCParticles::const_reference& mcp) { */
  /*        return mcp.isLong && mcp.fromCharmDecay && !mcp.isElectron() && mcp.inEta2_5() && mcp.p > 5e3f; */
  /*      }, */
  /*    }), */
  /*    TrackEffReport({ */
  /*      "Not electron long eta25 p<5GeV", */
  /*      [](MCParticles::const_reference& mcp) { */
  /*        return mcp.isLong && !mcp.isElectron() && mcp.inEta2_5() && mcp.p < 5e3f; */
  /*      }, */
  /*    }), */
  /*    TrackEffReport({ */
  /*      "Not electron long eta25 p>3GeV pt>400MeV", */
  /*      [](MCParticles::const_reference& mcp) { */
  /*        return mcp.isLong && !mcp.isElectron() && mcp.inEta2_5() && mcp.p > 3e3f && mcp.pt > 400.f; */
  /*      }, */
  /*    }), */
  /*    TrackEffReport({ */
  /*      "Not electron long eta25 p>5GeV", */
  /*      [](MCParticles::const_reference& mcp) { */
  /*        return mcp.isLong && !mcp.isElectron() && mcp.inEta2_5() && mcp.p > 5e3f; */
  /*      }, */
  /*    }), */
  /*    TrackEffReport({ */
  /*      "Not electron long strange eta25", */
  /*      [](MCParticles::const_reference& mcp) { */
  /*        return mcp.isLong && mcp.fromStrangeDecay && !mcp.isElectron() && mcp.inEta2_5(); */
  /*      }, */
  /*    }), */
  /*    TrackEffReport({ */
  /*      "Not electron long strange eta25 p<5GeV", */
  /*      [](MCParticles::const_reference& mcp) { */
  /*        return mcp.isLong && mcp.fromStrangeDecay && !mcp.isElectron() && mcp.inEta2_5() && mcp.p < 5e3f; */
  /*      }, */
  /*    }), */
  /*    TrackEffReport({ */
  /*      "Not electron long strange eta25 p>3GeV pt>400MeV", */
  /*      [](MCParticles::const_reference& mcp) { */
  /*        return mcp.isLong && mcp.fromStrangeDecay && !mcp.isElectron() && mcp.inEta2_5() && mcp.p > 3e3f && */
  /*               mcp.pt > 400.f; */
  /*      }, */
  /*    }), */
  /*    TrackEffReport({ */
  /*      "Not electron long strange eta25 p>5GeV", */
  /*      [](MCParticles::const_reference& mcp) { */
  /*        return mcp.isLong && mcp.fromStrangeDecay && !mcp.isElectron() && mcp.inEta2_5() && mcp.p > 5e3f; */
  /*      }, */
  /*    }), */
  /*    TrackEffReport({ */
  /*      "Not electron Velo", */
  /*      [](MCParticles::const_reference& mcp) { return mcp.hasVelo && !mcp.isElectron(); }, */
  /*    }), */
  /*    TrackEffReport({ */
  /*      "Not electron Velo backward", */
  /*      [](MCParticles::const_reference& mcp) { return mcp.hasVelo && !mcp.isElectron() && mcp.eta < 0; }, */
  /*    }), */
  /*    TrackEffReport({ */
  /*      "Not electron Velo forward", */
  /*      [](MCParticles::const_reference& mcp) { return mcp.hasVelo && !mcp.isElectron() && mcp.eta > 0; }, */
  /*    }), */
  /*    TrackEffReport({ */
  /*      "Not electron Velo eta25", */
  /*      [](MCParticles::const_reference& mcp) { return mcp.hasVelo && !mcp.isElectron() && mcp.inEta2_5(); }, */
  /*    })}}; */

  template<>
  inline std::vector<HistoCategory> make_histo_category_vector<Checker::Subdetector::Velo>()
  {
    return std::vector<HistoCategory> {
      {// define which categories to create histograms for
       HistoCategory({
         "VeloTracks_electrons",
         [](MCParticles::const_reference& mcp) { return mcp.hasVelo && mcp.isElectron(); },
       }),
       HistoCategory({
         "VeloTracks_eta25_electrons",
         [](MCParticles::const_reference& mcp) { return mcp.hasVelo && mcp.isElectron() && mcp.inEta2_5(); },
       }),
       HistoCategory({
         "LongFromB_eta25_electrons",
         [](MCParticles::const_reference& mcp) {
           return mcp.isLong && mcp.fromBeautyDecay && mcp.isElectron() && mcp.inEta2_5();
         },
       }),
       HistoCategory({
         "LongFromD_eta25_electrons",
         [](MCParticles::const_reference& mcp) {
           return mcp.isLong && mcp.fromCharmDecay && mcp.isElectron() && mcp.inEta2_5();
         },
       }),
       HistoCategory({
         "LongStrange_eta25_electrons",
         [](MCParticles::const_reference& mcp) {
           return mcp.isLong && mcp.fromStrangeDecay && mcp.isElectron() && mcp.inEta2_5();
         },
       }),
       HistoCategory({
         "VeloTracks_notElectrons",
         [](MCParticles::const_reference& mcp) { return mcp.hasVelo && !mcp.isElectron(); },
       }),
       HistoCategory({
         "VeloTracks_eta25_notElectrons",
         [](MCParticles::const_reference& mcp) { return mcp.hasVelo && !mcp.isElectron() && mcp.inEta2_5(); },
       }),
       HistoCategory({
         "LongFromB_eta25_notElectrons",
         [](MCParticles::const_reference& mcp) {
           return mcp.isLong && mcp.fromBeautyDecay && !mcp.isElectron() && mcp.inEta2_5();
         },
       }),
       HistoCategory({
         "LongFromD_eta25_notElectrons",
         [](MCParticles::const_reference& mcp) {
           return mcp.isLong && mcp.fromCharmDecay && !mcp.isElectron() && mcp.inEta2_5();
         },
       }),
       HistoCategory({
         "LongStrange_eta25_notElectrons",
         [](MCParticles::const_reference& mcp) {
           return mcp.isLong && mcp.fromStrangeDecay && !mcp.isElectron() && mcp.inEta2_5();
         },
       })}};
  }

  template<>
  inline std::vector<TrackEffReport> make_track_eff_report_vector<Checker::Subdetector::UT>()
  {
    return std::vector<TrackEffReport> {{
      TrackEffReport({
        "01_velo",
        [](MCParticles::const_reference& mcp) { return mcp.hasVelo && !mcp.isElectron() && mcp.inEta2_5(); },
      }),
      TrackEffReport({
        "02_velo+UT",
        [](MCParticles::const_reference& mcp) {
          return mcp.hasVelo && mcp.hasUT && !mcp.isElectron() && mcp.inEta2_5();
        },
      }),
      TrackEffReport({
        "03_velo+UT_P>5GeV",
        [](MCParticles::const_reference& mcp) {
          return mcp.hasVelo && mcp.hasUT && !mcp.isElectron() && mcp.p > 5e3f && mcp.inEta2_5();
        },
      }),
      TrackEffReport({
        "04_velo+notLong",
        [](MCParticles::const_reference& mcp) {
          return mcp.hasVelo && !mcp.isElectron() && !mcp.isLong && mcp.inEta2_5();
        },
      }),
      TrackEffReport({
        "05_velo+UT+notLong",
        [](MCParticles::const_reference& mcp) {
          return mcp.hasVelo && mcp.hasUT && !mcp.isElectron() && !mcp.isLong && mcp.inEta2_5();
        },
      }),
      TrackEffReport({
        "06_velo+UT+notLong_P>5GeV",
        [](MCParticles::const_reference& mcp) {
          return mcp.hasVelo && mcp.hasUT && !mcp.isElectron() && !mcp.isLong && mcp.p > 5e3f && mcp.inEta2_5();
        },
      }),
      TrackEffReport({
        "07_long",
        [](MCParticles::const_reference& mcp) { return mcp.isLong && !mcp.isElectron() && mcp.inEta2_5(); },
      }),
      TrackEffReport({
        "08_long_P>5GeV",
        [](MCParticles::const_reference& mcp) {
          return mcp.isLong && !mcp.isElectron() && mcp.p > 5e3f && mcp.inEta2_5();
        },
      }),
      TrackEffReport({
        "09_long_fromB",
        [](MCParticles::const_reference& mcp) {
          return mcp.isLong && !mcp.isElectron() && mcp.fromBeautyDecay && mcp.inEta2_5();
        },
      }),
      TrackEffReport({
        "10_long_fromB_P>5GeV",
        [](MCParticles::const_reference& mcp) {
          return mcp.isLong && !mcp.isElectron() && mcp.fromBeautyDecay && mcp.p > 5e3f && mcp.inEta2_5();
        },
      }),
      TrackEffReport({
        "11_long_electrons",
        [](MCParticles::const_reference& mcp) { return mcp.isLong && mcp.isElectron() && mcp.inEta2_5(); },
      }),
      TrackEffReport({
        "12_long_fromB_electrons",
        [](MCParticles::const_reference& mcp) {
          return mcp.isLong && mcp.isElectron() && mcp.fromBeautyDecay && mcp.inEta2_5();
        },
      }),
      TrackEffReport({
        "13_long_fromB_electrons_P>5GeV",
        [](MCParticles::const_reference& mcp) {
          return mcp.isLong && mcp.isElectron() && mcp.fromBeautyDecay && mcp.p > 5e3f && mcp.inEta2_5();
        },
      })

      // define which categories to monitor
      /* TrackEffReport({ */
      /*   "Velo", */
      /*   [](MCParticles::const_reference& mcp) { return mcp.hasVelo && !mcp.isElectron() && mcp.inEta2_5(); }, */
      /* }), */
      /* TrackEffReport({ */
      /*   "Velo+UT", */
      /*   [](MCParticles::const_reference& mcp) { return mcp.hasVelo && mcp.hasUT && !mcp.isElectron() &&
         mcp.inEta2_5();
         }, */
      /* }), */
      /* TrackEffReport({ */
      /*   "Velo+UT, p > 5 GeV", */
      /*   [](MCParticles::const_reference& mcp) { */
      /*     return mcp.hasVelo && mcp.hasUT && mcp.p > 5e3f && !mcp.isElectron() && mcp.inEta2_5(); */
      /*   }, */
      /* }), */
      /* TrackEffReport({ */
      /*   "Velo, not long", */
      /*   [](MCParticles::const_reference& mcp) { */
      /*     return mcp.hasVelo && !mcp.isLong && !mcp.isElectron() && mcp.inEta2_5(); */
      /*   }, */
      /* }), */
      /* TrackEffReport({ */
      /*   "Velo+UT, not long", */
      /*   [](MCParticles::const_reference& mcp) { */
      /*     return mcp.hasVelo && mcp.hasUT && !mcp.isLong && !mcp.isElectron() && mcp.inEta2_5(); */
      /*   }, */
      /* }), */
      /* TrackEffReport({ */
      /*   "Velo+UT, not long, p > 5 GeV", */
      /*   [](MCParticles::const_reference& mcp) { */
      /*     return mcp.hasVelo && mcp.hasUT && !mcp.isLong && mcp.p > 5e3f && !mcp.isElectron() && mcp.inEta2_5(); */
      /*   }, */
      /* }), */
      /* TrackEffReport({ */
      /*   "Long", */
      /*   [](MCParticles::const_reference& mcp) { return mcp.isLong && !mcp.isElectron() && mcp.inEta2_5(); }, */
      /* }), */
      /* TrackEffReport({ */
      /*   "Long, p > 5 GeV", */
      /*   [](MCParticles::const_reference& mcp) { */
      /*     return mcp.isLong && mcp.p > 5e3f && !mcp.isElectron() && mcp.inEta2_5(); */
      /*   }, */
      /* }), */
      /* TrackEffReport({ */
      /*   "Long, pt > 20 GeV", */
      /*   [](MCParticles::const_reference& mcp) { */
      /*     return mcp.isLong && mcp.pt > 20e3f && !mcp.isElectron() && mcp.inEta2_5(); */
      /*   }, */
      /* }), */
      /* TrackEffReport({ */
      /*   "Long strange", */
      /*   [](MCParticles::const_reference& mcp) { */
      /*     return mcp.isLong && mcp.fromStrangeDecay && !mcp.isElectron() && mcp.inEta2_5(); */
      /*   }, */
      /* }), */
      /* TrackEffReport({ */
      /*   "Long strange, p > 5 GeV", */
      /*   [](MCParticles::const_reference& mcp) { */
      /*     return mcp.isLong && mcp.fromStrangeDecay && !mcp.isElectron() && mcp.p > 5e3f && mcp.inEta2_5(); */
      /*   }, */
      /* }), */
      /* TrackEffReport({ */
      /*   "Long strange muons", */
      /*   [](MCParticles::const_reference& mcp) { */
      /*     return mcp.isLong && mcp.fromStrangeDecay && mcp.isMuon() && mcp.inEta2_5(); */
      /*   }, */
      /* }), */
      /* TrackEffReport({ */
      /*   "Long strange muons, p > 3 GeV", */
      /*   [](MCParticles::const_reference& mcp) { */
      /*     return mcp.isLong && mcp.fromStrangeDecay && mcp.isMuon() && mcp.p > 3e3f && mcp.inEta2_5(); */
      /*   }, */
      /* }), */
      /* TrackEffReport({ */
      /*   "Long from B", */
      /*   [](MCParticles::const_reference& mcp) { */
      /*     return mcp.isLong && mcp.fromBeautyDecay && !mcp.isElectron() && mcp.inEta2_5(); */
      /*   }, */
      /* }), */
      /* TrackEffReport({ */
      /*   "Long from B, p > 5 GeV", */
      /*   [](MCParticles::const_reference& mcp) { */
      /*     return mcp.isLong && mcp.fromBeautyDecay && mcp.p > 5e3f && !mcp.isElectron() && mcp.inEta2_5(); */
      /*   }, */
      /* }), */
      /* TrackEffReport({ */
      /*   "Long electrons", */
      /*   [](MCParticles::const_reference& mcp) { return mcp.isLong && mcp.isElectron() && mcp.inEta2_5(); }, */
      /* }), */
      /* TrackEffReport({ */
      /*   "Long from B electrons", */
      /*   [](MCParticles::const_reference& mcp) { */
      /*     return mcp.isLong && mcp.fromBeautyDecay && mcp.isElectron() && mcp.inEta2_5(); */
      /*   }, */
      /* }), */
      /* TrackEffReport({ */
      /*   "Long from B electrons, p > 5 GeV", */
      /*   [](MCParticles::const_reference& mcp) { */
      /*     return mcp.isLong && mcp.fromBeautyDecay && mcp.isElectron() && mcp.p > 5e3f && mcp.inEta2_5(); */
      /*   }, */
      /* }), */
      /* TrackEffReport({ */
      /*   "Long from D electrons, p > 3 GeV, pt > 0.5 GeV", */
      /*   [](MCParticles::const_reference& mcp) { */
      /*     return mcp.isLong && mcp.fromCharmDecay && mcp.isElectron() && mcp.p > 3e3f && mcp.pt > 0.5e3f && */
      /*            mcp.inEta2_5(); */
      /*   }, */
      /* }), */
      /* TrackEffReport({ */
      /*   "Long from D, p > 3 GeV, pt > 0.5 GeV", */
      /*   [](MCParticles::const_reference& mcp) { */
      /*     return mcp.isLong && mcp.fromCharmDecay && !mcp.isElectron() && mcp.p > 3e3f && mcp.pt > 0.5e3f && */
      /*            mcp.inEta2_5(); */
      /*   }, */
      /* }), */
      /* TrackEffReport({ */
      /*   "Long from B electrons, p > 3 GeV, pt > 0.5 GeV", */
      /*   [](MCParticles::const_reference& mcp) { */
      /*     return mcp.isLong && mcp.fromBeautyDecay && mcp.isElectron() && mcp.p > 3e3f && mcp.pt > 0.5e3f && */
      /*            mcp.inEta2_5(); */
      /*   }, */
      /* }), */
      /* TrackEffReport({ */
      /*   "Long from B, p > 3 GeV, pt > 0.5 GeV", */
      /*   [](MCParticles::const_reference& mcp) { */
      /*     return mcp.isLong && mcp.fromBeautyDecay && !mcp.isElectron() && mcp.p > 3e3f && mcp.pt > 0.5e3f && */
      /*            mcp.inEta2_5(); */
      /*   }, */
      /* }), */
      /* TrackEffReport({ */
      /*   "Long from B, p > 3 GeV, pt > 0.5 GeV, eta < 2.5, (phi-pi/2)<0.8", */
      /*   [](MCParticles::const_reference& mcp) { */
      /*     return mcp.isLong && mcp.fromBeautyDecay && !mcp.isElectron() && mcp.p > 3e3f && mcp.pt > 0.5e3f && */
      /*            mcp.inEta2_5() && mcp.eta < 2.5f && std::fabs(std::fabs(mcp.phi) - 1.57f) < 0.8f; */
      /*   }, */
      /* }), */
      /* TrackEffReport({ */
      /*   "Long from B, p > 3 GeV, pt > 0.5 GeV, eta > 2.5, (phi-pi/2)<0.8", */
      /*   [](MCParticles::const_reference& mcp) { */
      /*     return mcp.isLong && mcp.fromBeautyDecay && !mcp.isElectron() && mcp.p > 3e3f && mcp.pt > 0.5e3f && */
      /*            mcp.inEta2_5() && mcp.eta > 2.5f && std::fabs(std::fabs(mcp.phi) - 1.57f) < 0.8f; */
      /*   }, */
      /* }), */
      /* TrackEffReport({ */
      /*   "Long from B, p > 3 GeV, pt > 0.5 GeV, eta < 2.5, (phi-pi/2)>0.8", */
      /*   [](MCParticles::const_reference& mcp) { */
      /*     return mcp.isLong && mcp.fromBeautyDecay && !mcp.isElectron() && mcp.p > 3e3f && mcp.pt > 0.5e3f && */
      /*            mcp.inEta2_5() && mcp.eta < 2.5f && std::fabs(std::fabs(mcp.phi) - 1.57f) > 0.8f; */
      /*   }, */
      /* }), */
      /* TrackEffReport({ */
      /*   "Long from B, p > 3 GeV, pt > 0.5 GeV, eta > 2.5, (phi-pi/2)>0.8", */
      /*   [](MCParticles::const_reference& mcp) { */
      /*     return mcp.isLong && mcp.fromBeautyDecay && !mcp.isElectron() && mcp.p > 3e3f && mcp.pt > 0.5e3f && */
      /*            mcp.inEta2_5() && mcp.eta > 2.5f && std::fabs(std::fabs(mcp.phi) - 1.57f) > 0.8f; */
      /*   }, */
      /* }),  */
    }};
  }

  template<>
  inline std::vector<HistoCategory> make_histo_category_vector<Checker::Subdetector::UT>()
  {
    return std::vector<HistoCategory> {
      {// define which categories to create histograms for
       HistoCategory({
         "VeloUTTracks_eta25_electrons",
         [](MCParticles::const_reference& mcp) {
           return mcp.hasVelo && mcp.hasUT && mcp.isElectron() && mcp.inEta2_5();
         },
       }),
       HistoCategory({
         "LongFromB_eta25_electrons",
         [](MCParticles::const_reference& mcp) {
           return mcp.isLong && mcp.fromBeautyDecay && mcp.isElectron() && mcp.inEta2_5();
         },
       }),
       HistoCategory({
         "LongFromD_eta25_electrons",
         [](MCParticles::const_reference& mcp) {
           return mcp.isLong && mcp.fromCharmDecay && mcp.isElectron() && mcp.inEta2_5();
         },
       }),
       HistoCategory({
         "LongStrange_eta25_electrons",
         [](MCParticles::const_reference& mcp) {
           return mcp.isLong && mcp.fromStrangeDecay && mcp.isElectron() && mcp.inEta2_5();
         },
       }),
       HistoCategory({
         "VeloUTTracks_eta25_notElectrons",
         [](MCParticles::const_reference& mcp) {
           return mcp.hasVelo && mcp.hasUT && !mcp.isElectron() && mcp.inEta2_5();
         },
       }),
       HistoCategory({
         "LongFromB_eta25_notElectrons",
         [](MCParticles::const_reference& mcp) {
           return mcp.isLong && mcp.fromBeautyDecay && !mcp.isElectron() && mcp.inEta2_5();
         },
       }),
       HistoCategory({
         "LongFromD_eta25_notElectrons",
         [](MCParticles::const_reference& mcp) {
           return mcp.isLong && mcp.fromCharmDecay && !mcp.isElectron() && mcp.inEta2_5();
         },
       }),
       HistoCategory({
         "LongStrange_eta25_notElectrons",
         [](MCParticles::const_reference& mcp) {
           return mcp.isLong && mcp.fromStrangeDecay && !mcp.isElectron() && mcp.inEta2_5();
         },
       })}};
  }

  template<>
  inline std::vector<TrackEffReport> make_track_eff_report_vector<Checker::Subdetector::SciFi>()
  {
    return std::vector<TrackEffReport> {{
      // define which categories to monitor
      TrackEffReport({
        "01_long",
        [](MCParticles::const_reference& mcp) { return mcp.isLong && !mcp.isElectron() && mcp.inEta2_5(); },
      }),
      TrackEffReport({
        "02_long_P>5GeV",
        [](MCParticles::const_reference& mcp) {
          return mcp.isLong && !mcp.isElectron() && mcp.p > 5e3f && mcp.inEta2_5();
        },
      }),
      TrackEffReport({
        "03_long_strange",
        [](MCParticles::const_reference& mcp) {
          return mcp.isLong && !mcp.isElectron() && mcp.fromStrangeDecay && mcp.inEta2_5();
        },
      }),
      TrackEffReport({
        "04_long_strange_P>5GeV",
        [](MCParticles::const_reference& mcp) {
          return mcp.isLong && !mcp.isElectron() && mcp.fromStrangeDecay && mcp.p > 5e3f && mcp.inEta2_5();
        },
      }),
      TrackEffReport({
        "05_long_fromB",
        [](MCParticles::const_reference& mcp) {
          return mcp.isLong && !mcp.isElectron() && mcp.fromBeautyDecay && mcp.inEta2_5();
        },
      }),
      TrackEffReport({
        "06_long_fromB_P>5GeV",
        [](MCParticles::const_reference& mcp) {
          return mcp.isLong && !mcp.isElectron() && mcp.fromBeautyDecay && mcp.p > 5e3f && mcp.inEta2_5();
        },
      }),
      TrackEffReport({
        "07_long_electrons",
        [](MCParticles::const_reference& mcp) { return mcp.isLong && mcp.isElectron() && mcp.inEta2_5(); },
      }),
      TrackEffReport({
        "08_long_electrons_P>5GeV",
        [](MCParticles::const_reference& mcp) {
          return mcp.isLong && mcp.isElectron() && mcp.p > 5e3f && mcp.inEta2_5();
        },
      }),
      TrackEffReport({
        "09_long_fromB_electrons",
        [](MCParticles::const_reference& mcp) {
          return mcp.isLong && mcp.isElectron() && mcp.fromBeautyDecay && mcp.inEta2_5();
        },
      }),
      TrackEffReport({
        "10_long_fromB_electrons_P>5GeV",
        [](MCParticles::const_reference& mcp) {
          return mcp.isLong && mcp.isElectron() && mcp.fromBeautyDecay && mcp.p > 5e3f && mcp.inEta2_5();
        },
      }),
      TrackEffReport({
        "long_P>5GeV_AND_Pt>1GeV",
        [](MCParticles::const_reference& mcp) {
          return mcp.isLong && !mcp.isElectron() && mcp.p > 5e3f && mcp.pt > 1e3f && mcp.inEta2_5();
        },
      }),
      TrackEffReport({
        "long_fromB_P>5GeV_AND_Pt>1GeV",
        [](MCParticles::const_reference& mcp) {
          return mcp.isLong && !mcp.isElectron() && mcp.fromBeautyDecay && mcp.p > 5e3f && mcp.pt > 1e3f &&
                 mcp.inEta2_5();
        },
      }),
      TrackEffReport({
        "11_noVelo_UT",
        [](MCParticles::const_reference& mcp) {
          return !mcp.hasVelo && mcp.hasUT && mcp.hasSciFi && mcp.isElectron() && mcp.inEta2_5();
        },
      }),
      TrackEffReport({
        "12_noVelo_UT_P>5GeV",
        [](MCParticles::const_reference& mcp) {
          return !mcp.hasVelo && mcp.hasUT && mcp.hasSciFi && mcp.isElectron() && mcp.inEta2_5() && mcp.p > 5e3f;
        },
      }),
      TrackEffReport({
        "13_long_PT>2GeV",
        [](MCParticles::const_reference& mcp) {
          return mcp.isLong && !mcp.isElectron() && mcp.pt > 2e3f && mcp.inEta2_5();
        },
      }),
      TrackEffReport({
        "14_long_from_B_PT>2GeV",
        [](MCParticles::const_reference& mcp) {
          return mcp.isLong && mcp.fromBeautyDecay && !mcp.isElectron() && mcp.pt > 2e3f && mcp.inEta2_5();
        },
      }),
      TrackEffReport({
        "15_long_strange_P>5GeV",
        [](MCParticles::const_reference& mcp) {
          return mcp.isLong && !mcp.isElectron() && mcp.fromStrangeDecay && mcp.p > 5e3f && mcp.inEta2_5();
        },
      }),
      TrackEffReport({
        "16_long_strange_P>5GeV_PT>500MeV",
        [](MCParticles::const_reference& mcp) {
          return mcp.isLong && !mcp.isElectron() && mcp.fromStrangeDecay && mcp.p > 5e3f && mcp.inEta2_5() &&
                 mcp.pt > 5e2f;
        },
      }),
      TrackEffReport({
        "17_long_fromSignal",
        [](MCParticles::const_reference& mcp) { return mcp.isLong && mcp.fromSignal && mcp.inEta2_5(); },
      }),

      /* TrackEffReport({ */
      /*   "Long", */
      /*   [](MCParticles::const_reference& mcp) { return mcp.isLong && !mcp.isElectron() && mcp.inEta2_5(); }, */
      /* }), */
      /* TrackEffReport({ */
      /*   "Long, p > 5 GeV", */
      /*   [](MCParticles::const_reference& mcp) { */
      /*     return mcp.isLong && mcp.p > 5e3f && !mcp.isElectron() && mcp.inEta2_5(); */
      /*   }, */
      /* }), */
      /* TrackEffReport({ */
      /*   "Long, pt > 20 GeV", */
      /*   [](MCParticles::const_reference& mcp) { */
      /*     return mcp.isLong && mcp.pt > 20e3f && !mcp.isElectron() && mcp.inEta2_5(); */
      /*   }, */
      /* }), */
      /* TrackEffReport({ */
      /*   "Long strange", */
      /*   [](MCParticles::const_reference& mcp) { */
      /*     return mcp.isLong && mcp.fromStrangeDecay && !mcp.isElectron() && mcp.inEta2_5(); */
      /*   }, */
      /* }), */
      /* TrackEffReport({ */
      /*   "Long strange, p > 5 GeV", */
      /*   [](MCParticles::const_reference& mcp) { */
      /*     return mcp.isLong && mcp.fromStrangeDecay && !mcp.isElectron() && mcp.p > 5e3f && mcp.inEta2_5(); */
      /*   }, */
      /* }), */
      /* TrackEffReport({ */
      /*   "Long strange muons", */
      /*   [](MCParticles::const_reference& mcp) { */
      /*     return mcp.isLong && mcp.fromStrangeDecay && mcp.isMuon() && mcp.inEta2_5(); */
      /*   }, */
      /* }), */
      /* TrackEffReport({ */
      /*   "Long strange muons, p > 3 GeV", */
      /*   [](MCParticles::const_reference& mcp) { */
      /*     return mcp.isLong && mcp.fromStrangeDecay && mcp.isMuon() && mcp.p > 3e3f && mcp.inEta2_5(); */
      /*   }, */
      /* }), */
      /* TrackEffReport({ */
      /*   "Long from B", */
      /*   [](MCParticles::const_reference& mcp) { */
      /*     return mcp.isLong && mcp.fromBeautyDecay && !mcp.isElectron() && mcp.inEta2_5(); */
      /*   }, */
      /* }), */
      /* TrackEffReport({ */
      /*   "Long from B, p > 5 GeV", */
      /*   [](MCParticles::const_reference& mcp) { */
      /*     return mcp.isLong && mcp.fromBeautyDecay && mcp.p > 5e3f && !mcp.isElectron() && mcp.inEta2_5(); */
      /*   }, */
      /* }), */
      /* TrackEffReport({ */
      /*   "Long electrons", */
      /*   [](MCParticles::const_reference& mcp) { return mcp.isLong && mcp.isElectron() && mcp.inEta2_5(); }, */
      /* }), */
      /* TrackEffReport({ */
      /*   "Long electrons from B", */
      /*   [](MCParticles::const_reference& mcp) { */
      /*     return mcp.isLong && mcp.fromBeautyDecay && mcp.isElectron() && mcp.inEta2_5(); */
      /*   }, */
      /* }), */
      /* TrackEffReport({ */
      /*   "Long electrons from B, p > 5 GeV", */
      /*   [](MCParticles::const_reference& mcp) { */
      /*     return mcp.isLong && mcp.fromBeautyDecay && mcp.p > 5e3f && mcp.isElectron() && mcp.inEta2_5(); */
      /*   }, */
      /* }), */
      /* TrackEffReport({ */
      /*   "Long from D electrons, p > 3 GeV, pt > 0.3 GeV", */
      /*   [](MCParticles::const_reference& mcp) { */
      /*     return mcp.isLong && mcp.fromCharmDecay && mcp.isElectron() && mcp.p > 3e3f && mcp.pt > 0.3e3f && */
      /*            mcp.inEta2_5(); */
      /*   }, */
      /* }), */
      /* TrackEffReport({ */
      /*   "Long from D electrons, p > 3 GeV, pt > 0.5 GeV", */
      /*   [](MCParticles::const_reference& mcp) { */
      /*     return mcp.isLong && mcp.fromCharmDecay && mcp.isElectron() && mcp.p > 3e3f && mcp.pt > 0.5e3f && */
      /*            mcp.inEta2_5(); */
      /*   }, */
      /* }), */
      /* TrackEffReport({ */
      /*   "Long from D, p > 3 GeV, pt > 0.3 GeV", */
      /*   [](MCParticles::const_reference& mcp) { */
      /*     return mcp.isLong && mcp.fromCharmDecay && !mcp.isElectron() && mcp.p > 3e3f && mcp.pt > 0.3e3f && */
      /*            mcp.inEta2_5(); */
      /*   }, */
      /* }), */
      /* TrackEffReport({ */
      /*   "Long from D, p > 3 GeV, pt > 0.5 GeV", */
      /*   [](MCParticles::const_reference& mcp) { */
      /*     return mcp.isLong && mcp.fromCharmDecay && !mcp.isElectron() && mcp.p > 3e3f && mcp.pt > 0.5e3f && */
      /*            mcp.inEta2_5(); */
      /*   }, */
      /* }), */
      /* TrackEffReport({ */
      /*   "Long from B electrons, p > 3 GeV, pt > 0.3 GeV", */
      /*   [](MCParticles::const_reference& mcp) { */
      /*     return mcp.isLong && mcp.fromBeautyDecay && mcp.isElectron() && mcp.p > 3e3f && mcp.pt > 0.3e3f && */
      /*            mcp.inEta2_5(); */
      /*   }, */
      /* }), */
      /* TrackEffReport({ */
      /*   "Long from B electrons, p > 3 GeV, pt > 0.5 GeV", */
      /*   [](MCParticles::const_reference& mcp) { */
      /*     return mcp.isLong && mcp.fromBeautyDecay && mcp.isElectron() && mcp.p > 3e3f && mcp.pt > 0.5e3f && */
      /*            mcp.inEta2_5(); */
      /*   }, */
      /* }), */
      /* TrackEffReport({ */
      /*   "Long from B, p > 3 GeV, pt > 0.3 GeV", */
      /*   [](MCParticles::const_reference& mcp) { */
      /*     return mcp.isLong && mcp.fromBeautyDecay && !mcp.isElectron() && mcp.p > 3e3f && mcp.pt > 0.3e3f && */
      /*            mcp.inEta2_5(); */
      /*   }, */
      /* }), */
      /* TrackEffReport({ */
      /*   "Long from B, p > 3 GeV, pt > 0.5 GeV", */
      /*   [](MCParticles::const_reference& mcp) { */
      /*     return mcp.isLong && mcp.fromBeautyDecay && !mcp.isElectron() && mcp.p > 3e3f && mcp.pt > 0.5e3f && */
      /*            mcp.inEta2_5(); */
      /*   }, */
      /* }), */
      /* TrackEffReport({ */
      /*   "Long from B, p > 3 GeV, pt > 0.5 GeV, eta < 2.5, (phi-pi/2)<0.8", */
      /*   [](MCParticles::const_reference& mcp) { */
      /*     return mcp.isLong && mcp.fromBeautyDecay && !mcp.isElectron() && mcp.p > 3e3f && mcp.pt > 0.5e3f && */
      /*            mcp.inEta2_5() && mcp.eta < 2.5f && std::fabs(std::fabs(mcp.phi) - 1.57f) < 0.8f; */
      /*   }, */
      /* }), */
      /* TrackEffReport({ */
      /*   "Long from B, p > 3 GeV, pt > 0.5 GeV, eta > 2.5, (phi-pi/2)<0.8", */
      /*   [](MCParticles::const_reference& mcp) { */
      /*     return mcp.isLong && mcp.fromBeautyDecay && !mcp.isElectron() && mcp.p > 3e3f && mcp.pt > 0.5e3f && */
      /*            mcp.inEta2_5() && mcp.eta > 2.5f && std::fabs(std::fabs(mcp.phi) - 1.57f) < 0.8f; */
      /*   }, */
      /* }), */
      /* TrackEffReport({ */
      /*   "Long from B, p > 3 GeV, pt > 0.5 GeV, eta < 2.5, (phi-pi/2)>0.8", */
      /*   [](MCParticles::const_reference& mcp) { */
      /*     return mcp.isLong && mcp.fromBeautyDecay && !mcp.isElectron() && mcp.p > 3e3f && mcp.pt > 0.5e3f && */
      /*            mcp.inEta2_5() && mcp.eta < 2.5f && std::fabs(std::fabs(mcp.phi) - 1.57f) > 0.8f; */
      /*   }, */
      /* }), */
      /* TrackEffReport({ */
      /*   "Long from B, p > 3 GeV, pt > 0.5 GeV, eta > 2.5, (phi-pi/2)>0.8", */
      /*   [](MCParticles::const_reference& mcp) { */
      /*     return mcp.isLong && mcp.fromBeautyDecay && !mcp.isElectron() && mcp.p > 3e3f && mcp.pt > 0.5e3f && */
      /*            mcp.inEta2_5() && mcp.eta > 2.5f && std::fabs(std::fabs(mcp.phi) - 1.57f) > 0.8f; */
      /*   }, */
      /* }) */
    }};
  }

  template<>
  inline std::vector<HistoCategory> make_histo_category_vector<Checker::Subdetector::SciFi>()
  {
    return std::vector<HistoCategory> {
      {// define which categories to create histograms for
       HistoCategory({
         "Long_eta25_electrons",
         [](MCParticles::const_reference& mcp) { return mcp.isLong && mcp.isElectron() && mcp.inEta2_5(); },
       }),
       HistoCategory({
         "LongFromB_eta25_electrons",
         [](MCParticles::const_reference& mcp) {
           return mcp.isLong && mcp.fromBeautyDecay && mcp.isElectron() && mcp.inEta2_5();
         },
       }),
       HistoCategory({
         "LongFromD_eta25_electrons",
         [](MCParticles::const_reference& mcp) {
           return mcp.isLong && mcp.fromCharmDecay && mcp.isElectron() && mcp.inEta2_5();
         },
       }),
       HistoCategory({
         "LongStrange_eta25_electrons",
         [](MCParticles::const_reference& mcp) {
           return mcp.isLong && mcp.fromStrangeDecay && mcp.isElectron() && mcp.inEta2_5();
         },
       }),
       HistoCategory({
         "LongStrange_eta25_muons",
         [](MCParticles::const_reference& mcp) {
           return mcp.isLong && mcp.fromStrangeDecay && mcp.isMuon() && mcp.inEta2_5();
         },
       }),
       HistoCategory({
         "LongFromD_eta25_p_gt_3_pt_gt_0p5_electrons",
         [](MCParticles::const_reference& mcp) {
           return mcp.isLong && mcp.fromCharmDecay && mcp.isElectron() && mcp.inEta2_5() && mcp.p > 3e3f &&
                  mcp.pt > 0.5e3f;
         },
       }),
       HistoCategory({
         "LongFromD_eta25_p_gt_3_pt_gt_0p3_electrons",
         [](MCParticles::const_reference& mcp) {
           return mcp.isLong && mcp.fromCharmDecay && mcp.isElectron() && mcp.inEta2_5() && mcp.p > 3e3f &&
                  mcp.pt > 0.3e3f;
         },
       }),
       HistoCategory({
         "LongFromB_eta25_p_gt_3_pt_gt_0p5_electrons",
         [](MCParticles::const_reference& mcp) {
           return mcp.isLong && mcp.fromBeautyDecay && mcp.isElectron() && mcp.inEta2_5() && mcp.p > 3e3f &&
                  mcp.pt > 0.5e3f;
         },
       }),
       HistoCategory({
         "LongFromB_eta25_p_gt_3_pt_gt_0p3_electrons",
         [](MCParticles::const_reference& mcp) {
           return mcp.isLong && mcp.fromBeautyDecay && mcp.isElectron() && mcp.inEta2_5() && mcp.p > 3e3f &&
                  mcp.pt > 0.3e3f;
         },
       }),
       HistoCategory({
         "Long_eta25_notElectrons",
         [](MCParticles::const_reference& mcp) { return mcp.isLong && !mcp.isElectron() && mcp.inEta2_5(); },
       }),
       HistoCategory({
         "LongFromB_eta25_notElectrons",
         [](MCParticles::const_reference& mcp) {
           return mcp.isLong && mcp.fromBeautyDecay && !mcp.isElectron() && mcp.inEta2_5();
         },
       }),
       HistoCategory({
         "LongFromD_eta25_p_gt_3_pt_gt_0p5_notElectrons",
         [](MCParticles::const_reference& mcp) {
           return mcp.isLong && mcp.fromCharmDecay && !mcp.isElectron() && mcp.inEta2_5() && mcp.p > 3e3f &&
                  mcp.pt > 0.5e3f;
         },
       }),
       HistoCategory({
         "LongFromD_eta25_p_gt_3_pt_gt_0p3_notElectrons",
         [](MCParticles::const_reference& mcp) {
           return mcp.isLong && mcp.fromCharmDecay && !mcp.isElectron() && mcp.inEta2_5() && mcp.p > 3e3f &&
                  mcp.pt > 0.3e3f;
         },
       }),
       HistoCategory({
         "LongFromB_eta25_p_gt_3_pt_gt_0p5_notElectrons",
         [](MCParticles::const_reference& mcp) {
           return mcp.isLong && mcp.fromBeautyDecay && !mcp.isElectron() && mcp.inEta2_5() && mcp.p > 3e3f &&
                  mcp.pt > 0.5e3f;
         },
       }),
       HistoCategory({
         "LongFromB_eta25_p_gt_3_pt_gt_0p3_notElectrons",
         [](MCParticles::const_reference& mcp) {
           return mcp.isLong && mcp.fromBeautyDecay && !mcp.isElectron() && mcp.inEta2_5() && mcp.p > 3e3f &&
                  mcp.pt > 0.3e3f;
         },
       }),
       HistoCategory({
         "LongFromD_eta25_notElectrons",
         [](MCParticles::const_reference& mcp) {
           return mcp.isLong && mcp.fromCharmDecay && !mcp.isElectron() && mcp.inEta2_5();
         },
       }),
       HistoCategory({
         "LongStrange_eta25_notElectrons",
         [](MCParticles::const_reference& mcp) {
           return mcp.isLong && mcp.fromStrangeDecay && !mcp.isElectron() && mcp.inEta2_5();
         },
       })}};
  }

  template<>
  inline std::vector<TrackEffReport> make_track_eff_report_vector<Checker::Subdetector::SciFiSeeding>()
  {
    constexpr float pi = static_cast<float>(M_PI);
    constexpr auto base = [](MCParticles::const_reference& mcp) {
      return mcp.isLong && !mcp.isElectron() && mcp.inEta2_5();
    };
    return std::vector<TrackEffReport> {{
      // define which categories to monitor
      TrackEffReport({
        "00_P>3Gev_Pt>0.5",
        [](MCParticles::const_reference& mcp) {
          return mcp.isLong && !mcp.isElectron() && mcp.inEta2_5() && mcp.p > 3e3f && mcp.pt > 500.f;
        },
      }),
      TrackEffReport({
        "01_long",
        [](MCParticles::const_reference& mcp) { return mcp.isLong && !mcp.isElectron() && mcp.inEta2_5(); },
      }),
      TrackEffReport({
        "---1. phi quadrant",
        [](MCParticles::const_reference& mcp) {
          return mcp.isLong && !mcp.isElectron() && mcp.inEta2_5() && mcp.phi > 0 && mcp.phi < pi / 2.f;
        },
      }),
      TrackEffReport({
        "---2. phi quadrant",
        [](MCParticles::const_reference& mcp) {
          return mcp.isLong && !mcp.isElectron() && mcp.inEta2_5() && mcp.phi > pi / 2.f && mcp.phi < pi;
        },
      }),
      TrackEffReport({
        "---3. phi quadrant",
        [](MCParticles::const_reference& mcp) {
          return mcp.isLong && !mcp.isElectron() && mcp.inEta2_5() && mcp.phi > -pi && mcp.phi < -pi / 2.f;
        },
      }),
      TrackEffReport({
        "---4. phi quadrant",
        [](MCParticles::const_reference& mcp) {
          return mcp.isLong && !mcp.isElectron() && mcp.inEta2_5() && mcp.phi > -pi / 2.f && mcp.phi < 0.f;
        },
      }),
      TrackEffReport({
        "---eta < 2.5, small x, large y ",
        [&base](MCParticles::const_reference& mcp) {
          return base(mcp) && mcp.eta < 2.5f &&
                 ((mcp.phi > pi / 3.f && mcp.phi < 2.f * pi / 3.f) ||
                  (mcp.phi > -2.f * pi / 3.f && mcp.phi < -pi / 3.f));
        },
      }),
      TrackEffReport({
        "---eta < 2.5, large x, small y ",
        [&base](MCParticles::const_reference& mcp) {
          return base(mcp) && mcp.eta < 2.5f &&
                 ((mcp.phi > 2.f * pi / 3.f) || (mcp.phi < -2.f * pi / 3.f) ||
                  (mcp.phi > -pi / 3.f && mcp.phi < pi / 3.f));
        },
      }),
      TrackEffReport({
        "---eta > 2.5, small x, large y ",
        [&base](MCParticles::const_reference& mcp) {
          return base(mcp) && mcp.eta > 2.5f &&
                 ((mcp.phi > pi / 3.f && mcp.phi < 2 * pi / 3.f) || (mcp.phi > -2 * pi / 3.f && mcp.phi < -pi / 3.f));
        },
      }),
      TrackEffReport({
        "---eta > 2.5, large x, small y ",
        [&base](MCParticles::const_reference& mcp) {
          return base(mcp) && mcp.eta > 2.5f &&
                 ((mcp.phi > 2 * pi / 3.f) || (mcp.phi < -2 * pi / 3.f) || (mcp.phi > -pi / 3.f && mcp.phi < pi / 3.f));
        },
      }),
      TrackEffReport({
        "02_long_P>5GeV",
        [&base](MCParticles::const_reference& mcp) { return base(mcp) && mcp.p > 5e3f; },
      }),
      TrackEffReport({
        "02_long_P>5GeV, eta > 4",
        [&base](MCParticles::const_reference& mcp) { return base(mcp) && mcp.p > 5e3f && mcp.eta > 4.f; },
      }),
      TrackEffReport({
        "---eta < 2.5, small x, large y ",
        [&base](MCParticles::const_reference& mcp) {
          return base(mcp) && mcp.p > 5e3f && mcp.eta < 2.5f &&
                 ((mcp.phi > pi / 3.f && mcp.phi < 2.f * pi / 3.f) ||
                  (mcp.phi > -2.f * pi / 3.f && mcp.phi < -pi / 3.f));
        },
      }),
      TrackEffReport({
        "---eta < 2.5, large x, small y ",
        [&base](MCParticles::const_reference& mcp) {
          return base(mcp) && mcp.p > 5e3f && mcp.eta < 2.5f &&
                 ((mcp.phi > 2 * pi / 3.f) || (mcp.phi < -2 * pi / 3.f) || (mcp.phi > -pi / 3.f && mcp.phi < pi / 3.f));
        },
      }),
      TrackEffReport({
        "---eta > 2.5, small x, large y ",
        [&base](MCParticles::const_reference& mcp) {
          return base(mcp) && mcp.p > 5e3f && mcp.eta > 2.5f &&
                 ((mcp.phi > pi / 3.f && mcp.phi < 2.f * pi / 3.f) ||
                  (mcp.phi > -2.f * pi / 3.f && mcp.phi < -pi / 3.f));
        },
      }),
      TrackEffReport({
        "---eta > 2.5, large x, small y ",
        [&base](MCParticles::const_reference& mcp) {
          return base(mcp) && mcp.p > 5e3f && mcp.eta > 2.5f &&
                 ((mcp.phi > 2.f * pi / 3.f) || (mcp.phi < -2.f * pi / 3.f) ||
                  (mcp.phi > -pi / 3.f && mcp.phi < pi / 3.f));
        },
      }),
      TrackEffReport({
        "03_long_P>3GeV",
        [&base](MCParticles::const_reference& mcp) { return base(mcp) && mcp.p > 3e3f; },
      }),
      TrackEffReport({
        "04_long_P>0.5GeV",
        [&base](MCParticles::const_reference& mcp) { return base(mcp) && mcp.p > 5e2f; },
      }),
      TrackEffReport({
        "05_long_from_B",
        [&base](MCParticles::const_reference& mcp) { return base(mcp) && mcp.fromBeautyDecay; },
      }),
      TrackEffReport({
        "06_long_from_B_P>5GeV",
        [&base](MCParticles::const_reference& mcp) { return base(mcp) && mcp.fromBeautyDecay && mcp.p > 5e3f; },
      }),
      TrackEffReport({
        "07_long_from_B_P>3GeV",
        [](MCParticles::const_reference& mcp) {
          return mcp.isLong && mcp.fromBeautyDecay && !mcp.isElectron() && mcp.p > 3e3f && mcp.inEta2_5();
        },
      }),
      TrackEffReport({
        "08_UT+SciFi",
        [](MCParticles::const_reference& mcp) {
          return mcp.hasUT && mcp.hasSciFi && !mcp.hasVelo && !mcp.isElectron() && mcp.inEta2_5();
        },
      }),
      TrackEffReport({
        "09_UT+SciFi_P>5GeV",
        [](MCParticles::const_reference& mcp) {
          return mcp.hasUT && mcp.hasSciFi && !mcp.hasVelo && !mcp.isElectron() && mcp.p > 5e3f && mcp.inEta2_5();
        },
      }),
      TrackEffReport({
        "10_UT+SciFi_P>3GeV",
        [](MCParticles::const_reference& mcp) {
          return mcp.hasUT && mcp.hasSciFi && !mcp.hasVelo && !mcp.isElectron() && mcp.p > 3e3f && mcp.inEta2_5();
        },
      }),
      TrackEffReport({
        "11_UT+SciFi_fromStrange",
        [](MCParticles::const_reference& mcp) {
          return mcp.hasUT && mcp.hasSciFi && !mcp.hasVelo && mcp.fromStrangeDecay && !mcp.isElectron() &&
                 mcp.inEta2_5();
        },
      }),
      TrackEffReport({
        "12_UT+SciFi_fromStrange_P>5GeV",
        [](MCParticles::const_reference& mcp) {
          return mcp.hasUT && mcp.hasSciFi && !mcp.hasVelo && mcp.fromStrangeDecay && mcp.p > 5e3f &&
                 !mcp.isElectron() && mcp.inEta2_5();
        },
      }),
      TrackEffReport({
        "13_UT+SciFi_fromStrange_P>3GeV",
        [](MCParticles::const_reference& mcp) {
          return mcp.hasUT && mcp.hasSciFi && !mcp.hasVelo && mcp.fromStrangeDecay && mcp.p > 3e3f &&
                 !mcp.isElectron() && mcp.inEta2_5();
        },
      }),
      TrackEffReport({
        "14_long_electrons",
        [](MCParticles::const_reference& mcp) { return mcp.isLong && mcp.isElectron() && mcp.inEta2_5(); },
      }),
      TrackEffReport({
        "15_long_electrons_P>5GeV",
        [](MCParticles::const_reference& mcp) {
          return mcp.isLong && mcp.isElectron() && mcp.p > 5e3f && mcp.inEta2_5();
        },
      }),
      TrackEffReport({
        "16_long_electrons_P>3GeV",
        [](MCParticles::const_reference& mcp) {
          return mcp.isLong && mcp.isElectron() && mcp.p > 3e3f && mcp.inEta2_5();
        },
      }),
      TrackEffReport({
        "17_long_fromB_electrons",
        [](MCParticles::const_reference& mcp) {
          return mcp.isLong && mcp.isElectron() && mcp.fromBeautyDecay && mcp.inEta2_5();
        },
      }),
      TrackEffReport({
        "18_long_fromB_electrons_P>5GeV",
        [](MCParticles::const_reference& mcp) {
          return mcp.isLong && mcp.isElectron() && mcp.fromBeautyDecay && mcp.p > 5e3f && mcp.inEta2_5();
        },
      }),
      TrackEffReport({
        "19_long_PT>2GeV",
        [](MCParticles::const_reference& mcp) {
          return mcp.isLong && !mcp.isElectron() && mcp.pt > 2e3f && mcp.inEta2_5();
        },
      }),
      TrackEffReport({
        "20_long_from_B_PT>2GeV",
        [](MCParticles::const_reference& mcp) {
          return mcp.isLong && mcp.fromBeautyDecay && !mcp.isElectron() && mcp.pt > 2e3f && mcp.inEta2_5();
        },
      }),
      TrackEffReport({
        "21_long_strange_P>5GeV",
        [](MCParticles::const_reference& mcp) {
          return mcp.isLong && !mcp.isElectron() && mcp.fromStrangeDecay && mcp.p > 5e3f && mcp.inEta2_5();
        },
      }),
      TrackEffReport({
        "22_long_strange_P>5GeV_PT>500MeV",
        [](MCParticles::const_reference& mcp) {
          return mcp.isLong && !mcp.isElectron() && mcp.fromStrangeDecay && mcp.p > 5e3f && mcp.inEta2_5() &&
                 mcp.pt > 5e2f;
        },
      }),
    }};
  }

  template<>
  inline std::vector<HistoCategory> make_histo_category_vector<Checker::Subdetector::SciFiSeeding>()
  {
    return std::vector<HistoCategory> {
      {// define which categories to create histograms for
       HistoCategory({
         "Long_eta25_electrons",
         [](MCParticles::const_reference& mcp) { return mcp.isLong && mcp.isElectron() && mcp.inEta2_5(); },
       }),
       HistoCategory({
         "LongFromB_eta25_electrons",
         [](MCParticles::const_reference& mcp) {
           return mcp.isLong && mcp.fromBeautyDecay && mcp.isElectron() && mcp.inEta2_5();
         },
       }),
       HistoCategory({
         "LongFromD_eta25_electrons",
         [](MCParticles::const_reference& mcp) {
           return mcp.isLong && mcp.fromCharmDecay && mcp.isElectron() && mcp.inEta2_5();
         },
       }),
       HistoCategory({
         "LongStrange_eta25_electrons",
         [](MCParticles::const_reference& mcp) {
           return mcp.isLong && mcp.fromStrangeDecay && mcp.isElectron() && mcp.inEta2_5();
         },
       }),
       HistoCategory({
         "LongStrange_eta25_muons",
         [](MCParticles::const_reference& mcp) {
           return mcp.isLong && mcp.fromStrangeDecay && mcp.isMuon() && mcp.inEta2_5();
         },
       }),
       HistoCategory({
         "LongFromD_eta25_p_gt_3_pt_gt_0p5_electrons",
         [](MCParticles::const_reference& mcp) {
           return mcp.isLong && mcp.fromCharmDecay && mcp.isElectron() && mcp.inEta2_5() && mcp.p > 3e3f &&
                  mcp.pt > 0.5e3f;
         },
       }),
       HistoCategory({
         "LongFromD_eta25_p_gt_3_pt_gt_0p3_electrons",
         [](MCParticles::const_reference& mcp) {
           return mcp.isLong && mcp.fromCharmDecay && mcp.isElectron() && mcp.inEta2_5() && mcp.p > 3e3f &&
                  mcp.pt > 0.3e3f;
         },
       }),
       HistoCategory({
         "LongFromB_eta25_p_gt_3_pt_gt_0p5_electrons",
         [](MCParticles::const_reference& mcp) {
           return mcp.isLong && mcp.fromBeautyDecay && mcp.isElectron() && mcp.inEta2_5() && mcp.p > 3e3f &&
                  mcp.pt > 0.5e3f;
         },
       }),
       HistoCategory({
         "LongFromB_eta25_p_gt_3_pt_gt_0p3_electrons",
         [](MCParticles::const_reference& mcp) {
           return mcp.isLong && mcp.fromBeautyDecay && mcp.isElectron() && mcp.inEta2_5() && mcp.p > 3e3f &&
                  mcp.pt > 0.3e3f;
         },
       }),
       HistoCategory({
         "Long_eta25_notElectrons",
         [](MCParticles::const_reference& mcp) { return mcp.isLong && !mcp.isElectron() && mcp.inEta2_5(); },
       }),
       HistoCategory({
         "LongFromB_eta25_notElectrons",
         [](MCParticles::const_reference& mcp) {
           return mcp.isLong && mcp.fromBeautyDecay && !mcp.isElectron() && mcp.inEta2_5();
         },
       }),
       HistoCategory({
         "LongFromD_eta25_p_gt_3_pt_gt_0p5_notElectrons",
         [](MCParticles::const_reference& mcp) {
           return mcp.isLong && mcp.fromCharmDecay && !mcp.isElectron() && mcp.inEta2_5() && mcp.p > 3e3f &&
                  mcp.pt > 0.5e3f;
         },
       }),
       HistoCategory({
         "LongFromD_eta25_p_gt_3_pt_gt_0p3_notElectrons",
         [](MCParticles::const_reference& mcp) {
           return mcp.isLong && mcp.fromCharmDecay && !mcp.isElectron() && mcp.inEta2_5() && mcp.p > 3e3f &&
                  mcp.pt > 0.3e3f;
         },
       }),
       HistoCategory({
         "LongFromB_eta25_p_gt_3_pt_gt_0p5_notElectrons",
         [](MCParticles::const_reference& mcp) {
           return mcp.isLong && mcp.fromBeautyDecay && !mcp.isElectron() && mcp.inEta2_5() && mcp.p > 3e3f &&
                  mcp.pt > 0.5e3f;
         },
       }),
       HistoCategory({
         "LongFromB_eta25_p_gt_3_pt_gt_0p3_notElectrons",
         [](MCParticles::const_reference& mcp) {
           return mcp.isLong && mcp.fromBeautyDecay && !mcp.isElectron() && mcp.inEta2_5() && mcp.p > 3e3f &&
                  mcp.pt > 0.3e3f;
         },
       }),
       HistoCategory({
         "LongFromD_eta25_notElectrons",
         [](MCParticles::const_reference& mcp) {
           return mcp.isLong && mcp.fromCharmDecay && !mcp.isElectron() && mcp.inEta2_5();
         },
       }),
       HistoCategory({
         "LongStrange_eta25_notElectrons",
         [](MCParticles::const_reference& mcp) {
           return mcp.isLong && mcp.fromStrangeDecay && !mcp.isElectron() && mcp.inEta2_5();
         },
       })}};
  }

  template<>
  inline std::vector<TrackEffReport> make_track_eff_report_vector<Checker::Subdetector::Muon>()
  {
    return std::vector<TrackEffReport> {};
  }

  template<>
  inline std::vector<HistoCategory> make_histo_category_vector<Checker::Subdetector::Muon>()
  {
    return std::vector<HistoCategory> {};
  }

} // namespace Categories
