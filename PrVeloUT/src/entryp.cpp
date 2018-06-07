#include "../include/VeloTypes.h"
#include "../src/PrVeloUT.h"

int main() {

  // Create fake tracks
  const int nb_tracks = 10;
  const int nb_states = 20;

  std::vector<TrackVelo> tracks;
  for (int i=0; i<nb_tracks; ++i) {
    VeloUTTracking::TrackVelo tr;
    for (int j=0; j<nb_states; ++j) {
      VeloUTTracking::VeloState st;
      tr.emplace_back(st);
    }
    tracks.emplace_back(tr);
  }

  // Call the veloUT
  PrVeloUT velout;
  if ( velout.initialize() ) {
    velout(tracks);    
  }
 
}
