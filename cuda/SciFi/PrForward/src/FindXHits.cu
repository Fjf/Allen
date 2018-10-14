#include "FindXHits.cuh"



//=========================================================================
// From LHCb Forward tracking description
//
// Collect all X hits, within a window defined by the minimum Pt.
// Better restrictions possible, if we use the momentum of the input track.
// Ask for the presence of a stereo hit in the same biLayer compatible.
// This reduces the efficiency. X-alone hits to be re-added later in the processing
//
// side = 1  -> upper y half
// side = -1 -> lower y half
//=========================================================================
//
__host__ __device__ void collectAllXHits(
  const SciFi::SciFiHits& scifi_hits,
  const SciFi::SciFiHitCount& scifi_hit_count,
  int allXHits[SciFi::Tracking::max_x_hits],
  int& n_x_hits,
  float coordX[SciFi::Tracking::max_x_hits],
  const float xParams_seed[4], 
  const float yParams_seed[4],
  SciFi::Tracking::Arrays* constArrays,
  const MiniState& velo_state,
  const float qOverP,
  int side)
{
  // Find size of search window on reference plane, using Velo slopes and min pT as input  
  float dxRef = 0.9 * calcDxRef(SciFi::Tracking::minPt, velo_state);
  // find position within magnet where bending happens
  float zMag = zMagnet(velo_state, constArrays);
 
  const float q = qOverP > 0.f ? 1.f :-1.f;
  const float dir = q*SciFi::Tracking::magscalefactor*(-1.f);

  float slope2 = velo_state.tx*velo_state.tx + velo_state.ty*velo_state.ty; 
  const float pt = sqrtf( fabsf(1./ (qOverP*qOverP) ) ) * (slope2) / (1. + slope2);
  const bool wSignTreatment = SciFi::Tracking::useWrongSignWindow && pt > SciFi::Tracking::wrongSignPT;

  float dxRefWS = 0.0; 
  if( wSignTreatment ){
    // DvB: what happens if we use the acual momentum from VeloUT here instead of a constant?
    dxRefWS = 0.9 * calcDxRef(SciFi::Tracking::wrongSignPT, velo_state); //make windows a bit too small - FIXME check effect of this, seems wrong
  }

  int iZoneEnd[7]; //6 x planes
  iZoneEnd[0] = 0; 
  int cptZone = 1; 

  int iZoneStartingPoint = side > 0 ? constArrays->zoneoffsetpar : 0;

  for(unsigned int iZone = iZoneStartingPoint; iZone < iZoneStartingPoint + constArrays->zoneoffsetpar; iZone++) {
    assert ( iZone-iZoneStartingPoint < SciFi::Constants::n_zones );
    assert( iZone-iZoneStartingPoint < 12 );
    const float zZone   = constArrays->xZone_zPos[iZone-iZoneStartingPoint];
    const float xInZone = straightLineExtend(xParams_seed,zZone);
    const float yInZone = straightLineExtend(yParams_seed,zZone);

    // Now the code checks if the x and y are in the zone limits. I am really not sure
    // why this is done here, surely could just check if within limits for the last zone
    // in T3 and go from there? Need to think more about this.
    //
    // Here for now I assume the same min/max x and y for all stations, this again needs to
    // be read from some file blablabla although actually I suspect having some general tolerances
    // here is anyway good enough since we are doing a straight line extrapolation in the first place
    // check (roughly) whether the extrapolated velo track is within the current zone
    if (side > 0) {
      if (!isInside(xInZone,SciFi::Tracking::xLim_Min,SciFi::Tracking::xLim_Max)
          || !isInside(yInZone,SciFi::Tracking::yLim_Min,SciFi::Tracking::yLim_Max)) continue;
    } else {
      if (!isInside(xInZone,SciFi::Tracking::xLim_Min,SciFi::Tracking::xLim_Max)
          || !isInside(yInZone,side*SciFi::Tracking::yLim_Max,side*SciFi::Tracking::yLim_Min)) continue;
    }

    // extrapolate dxRef (x window on reference plane) to plane of current zone
    const float xTol  = ( zZone < SciFi::Tracking::zReference ) ? dxRef * zZone / SciFi::Tracking::zReference :  dxRef * (zZone - zMag) / ( SciFi::Tracking::zReference - zMag );
    float xMin        = xInZone - xTol;
    float xMax        = xInZone + xTol;

    if( SciFi::Tracking::useMomentumEstimate ) { //For VeloUT tracks, suppress check if track actually has qOverP set, get the option right!
      float xTolWS = 0.0;
      if( wSignTreatment ){
        xTolWS  = ( zZone < SciFi::Tracking::zReference ) ? dxRefWS * zZone / SciFi::Tracking::zReference :  dxRefWS * (zZone - zMag) / ( SciFi::Tracking::zReference - zMag );
      }
      if(dir > 0){
        xMin = xInZone - xTolWS;
      }else{
        xMax = xInZone + xTolWS;
      }
    }

    // Get the hits within the bounds
    assert ( iZone < SciFi::Constants::n_layers );
    assert ( constArrays->xZones[iZone] < SciFi::Constants::n_zones );
    int x_zone_offset_begin = scifi_hit_count.layer_offsets[constArrays->xZones[iZone]];
    int x_zone_offset_end = x_zone_offset_begin + scifi_hit_count.n_hits_layers[constArrays->xZones[iZone]];
    const int itH   = getLowerBound(scifi_hits.x0,xMin,x_zone_offset_begin,x_zone_offset_end); 
    const int itEnd = getLowerBound(scifi_hits.x0,xMax,x_zone_offset_begin,x_zone_offset_end);
    assert( itH >=  x_zone_offset_begin && itH <= x_zone_offset_end );
    assert( itEnd >=  x_zone_offset_begin && itEnd <= x_zone_offset_end );

    // Skip making range but continue if the end is before or equal to the start
    if (!(itEnd > itH)) continue; 
 
    // Now match the stereo hits
    const float this_uv_z   = constArrays->uvZone_zPos[iZone-iZoneStartingPoint];
    const float xInUv       = straightLineExtend(xParams_seed,this_uv_z);
    const float zRatio      = ( this_uv_z - zMag ) / ( zZone - zMag );
    const float dx          = yInZone * constArrays->uvZone_dxdy[iZone-iZoneStartingPoint];
    const float xCentral    = xInZone + dx;
    const float xPredUv     = xInUv + ( scifi_hits.x0[itH] - xInZone) * zRatio - dx;
    const float maxDx       = SciFi::Tracking::tolYCollectX + ( fabsf( scifi_hits.x0[itH] - xCentral ) + fabsf( yInZone ) ) * SciFi::Tracking::tolYSlopeCollectX;
    const float xMinUV      = xPredUv - maxDx;

    // Get bounds in UV layers
    // do one search on the same side as the x module
    // if we are close to y = 0, also look within a region on the other side module ("triangle search")
    assert( constArrays->uvZones[iZone] < SciFi::Constants::n_zones );
    const int uv_zone_offset_begin = scifi_hit_count.layer_offsets[constArrays->uvZones[iZone]];
    const int uv_zone_offset_end   = uv_zone_offset_begin + scifi_hit_count.n_hits_layers[constArrays->uvZones[iZone]];
    const int triangleOffset       = side > 0 ? -1 : 1;
    assert( constArrays->uvZones[iZone + constArrays->zoneoffsetpar*triangleOffset] < SciFi::Constants::n_zones );
    const int triangle_zone_offset_begin = scifi_hit_count.layer_offsets[constArrays->uvZones[iZone + constArrays->zoneoffsetpar*triangleOffset]];
    assert( constArrays->uvZones[iZone + constArrays->zoneoffsetpar*triangleOffset] < SciFi::Constants::n_zones );
    const int triangle_zone_offset_end   = triangle_zone_offset_begin + scifi_hit_count.n_hits_layers[constArrays->uvZones[iZone + constArrays->zoneoffsetpar*triangleOffset]];
    int itUV1                = getLowerBound(scifi_hits.x0,xMinUV,uv_zone_offset_begin,uv_zone_offset_end);    
    int itUV2                = getLowerBound(scifi_hits.x0,xMinUV,triangle_zone_offset_begin,triangle_zone_offset_end);

    const float xPredUVProto =  xInUv - xInZone * zRatio - dx;
    const float maxDxProto   =  SciFi::Tracking::tolYCollectX + fabsf( yInZone ) * SciFi::Tracking::tolYSlopeCollectX;

    const bool withTriangleSearch = fabsf(yInZone) < SciFi::Tracking::tolYTriangleSearch;
    for (int xHit = itH; xHit < itEnd; ++xHit) { //loop over all xHits in a layer between xMin and xMax
      const float xPredUv = xPredUVProto + scifi_hits.x0[xHit]* zRatio;
      const float maxDx   = maxDxProto   + fabsf( scifi_hits.x0[xHit] -xCentral )* SciFi::Tracking::tolYSlopeCollectX;
      const float xMinUV  = xPredUv - maxDx;
      const float xMaxUV  = xPredUv + maxDx;
      
      if ( matchStereoHit( itUV1, uv_zone_offset_end, scifi_hits, xMinUV, xMaxUV) //) {
           || ( withTriangleSearch && matchStereoHitWithTriangle(itUV2, triangle_zone_offset_end, yInZone, scifi_hits, xMinUV, xMaxUV, side ) ) ) {
        if ( n_x_hits >= SciFi::Tracking::max_x_hits )
          break;
        assert( n_x_hits < SciFi::Tracking::max_x_hits );
        allXHits[n_x_hits++] = xHit;
      }
    }
    
    
    const int iStart = iZoneEnd[cptZone-1];
    const int iEnd = n_x_hits;
    
    assert( cptZone < 7 );
    iZoneEnd[cptZone++] = iEnd;

    // project x of all hits to reference plane
    // save it in coordX
    // -> first step of 1D Hough transform,
    // select clusters of x hits on reference plane in selectXCandidates
    if ( iStart < iEnd ) {
      xAtRef_SamePlaneHits(
        scifi_hits, allXHits,
        n_x_hits, coordX, xParams_seed, constArrays,
        velo_state, zMag, iStart, iEnd); 
    }
    if ( n_x_hits >= SciFi::Tracking::max_x_hits )
      break; 
  }

  // Sort hits by coord
  // not using thrust::sort due to "temporary_buffer::allocate: get_temporary_buffer failed" error
  // every time thrust::sort is called, cudaMalloc is called, apparently there can be trouble
  // doing this many times
  //thrust::sort_by_key(thrust::seq, coordX, coordX + n_x_hits, allXHits); 
  sortHitsByKey<SciFi::Tracking::max_x_hits>( coordX, n_x_hits, allXHits );
}



//=========================================================================
//  Select the zones in the allXHits array where we can have a track
//=========================================================================
__host__ __device__ void selectXCandidates(
  const SciFi::SciFiHits& scifi_hits,
  const SciFi::SciFiHitCount& scifi_hit_count,
  int allXHits[SciFi::Tracking::max_x_hits],
  int& n_x_hits,
  bool usedHits[SciFi::Tracking::max_x_hits],
  float coordX[SciFi::Tracking::max_x_hits],
  const VeloUTTracking::TrackUT& veloUTTrack,
  SciFi::Tracking::Track candidate_tracks[SciFi::Tracking::max_candidate_tracks],
  int& n_candidate_tracks,
  const float zRef_track, 
  const float xParams_seed[4],
  const float yParams_seed[4],
  const MiniState& velo_state,
  SciFi::Tracking::HitSearchCuts& pars,
  SciFi::Tracking::Arrays* constArrays,
  int side,
  const bool secondLoop)
{
  if ( n_x_hits < pars.minXHits ) return;
  if ( secondLoop )
    if ( n_candidate_tracks >= SciFi::Tracking::max_tracks_second_loop ) return;
  if ( !secondLoop )
    if ( n_candidate_tracks >= SciFi::Tracking::max_candidate_tracks ) return;
  int itEnd = n_x_hits;
  const float xStraight = straightLineExtend(xParams_seed,SciFi::Tracking::zReference);
  int it1 = 0;
  int it2 = 0; 
  pars.minStereoHits = 0;

  PlaneCounter planeCounter;

  while (it1 < itEnd) {
    //find next unused Hits
    assert( it1 < n_x_hits );
    while ( it1+pars.minXHits - 1 < itEnd && usedHits[it1] ) ++it1;
    it2 = it1 + pars.minXHits;
    if (it2 > itEnd) break;
    assert( it2-1 < n_x_hits );
    while (it2 <= itEnd && usedHits[it2-1] ) ++it2;
    if (it2 > itEnd) break;

    //define search window for Cluster
    //TODO better xWindow calculation?? how to tune this???
    const float xWindow = pars.maxXWindow + (fabsf(coordX[it1]) + 
                                             fabsf(coordX[it1] - xStraight)
                                             ) * pars.maxXWindowSlope;
    
    if ( (coordX[it2 - 1] - coordX[it1]) > xWindow ) {
      ++it1;
      continue;
    }
 
    // Cluster candidate found, now count planes
    planeCounter.clear();
    for (int itH = it1; itH != it2; ++itH) {
      assert( itH < n_x_hits );
      if (!usedHits[itH]) {
        const int plane = scifi_hits.planeCode[allXHits[itH]]/2;
        planeCounter.addHit( plane );
      }
    }   
    // Improve cluster (at the moment only add hits to the right)
    int itLast = it2 - 1;
    while (it2 < itEnd) {
      assert( it2 < n_x_hits );
      if (usedHits[it2]) {
        ++it2;
        continue;
      } 
      //now  the first and last+1 hit exist and are not used!
      
      //Add next hit,
      // if there is only a small gap between the hits
      //    or inside window and plane is still empty
      assert( it2 < itEnd );
      if ( ( coordX[it2] < coordX[itLast] + pars.maxXGap ) || 
           ( (coordX[it2] - coordX[it1] < xWindow) && 
             (planeCounter.nbInPlane( scifi_hits.planeCode[allXHits[it2]]/2 )  == 0)
             ) 
         ) {
        planeCounter.addHit( scifi_hits.planeCode[allXHits[it2]]/2 );
        itLast = it2; 
        ++it2;
        continue;
      }   
      //Found nothing to improve
      else {
        break;
      }
    }

    //if not enough different planes, start again from the very beginning with next right hit
    if (planeCounter.nbDifferent < pars.minXHits) {
      ++it1;
      continue;
    }
    //====================================================================
    //  Now we have a (rather) clean candidate, do best hit selection
    //  Two possibilities:
    //  1) If there are enough planes with only one hit, do a straight
    //      line fit through these and select good matching others
    //  2) Do some magic
    //==================================================================== 
 
    // The line fitter was a standalone helper class in our framework, here I've ripped
    // its functionality out, basically have a struct which contains all the parameters
    // and some light helper math functions operating on these.
  
    SciFi::Tracking::LineFitterPars lineFitParameters;
    lineFitParameters.m_z0 = SciFi::Tracking::zReference;
    float xAtRef = 0.;
    const unsigned int nbSingle = planeCounter.nbSingle();
    int coordToFit[SciFi::Tracking::max_coordToFit];
    int n_coordToFit = 0;
    
    if ( nbSingle >= SciFi::Tracking::minSingleHits && nbSingle != planeCounter.nbDifferent ) {
      //1) we have enough single planes (thus two) to make a straight line fit
      int otherHits[SciFi::Constants::n_layers][SciFi::Tracking::max_other_hits] = {0};
      int nOtherHits[SciFi::Constants::n_layers] = {0};
      
      //seperate single and double hits
      for(auto itH = it1; it2 > itH; ++itH ){
        assert( itH < n_x_hits );
        if( usedHits[itH] ) continue;
        int planeCode = scifi_hits.planeCode[allXHits[itH]]/2;
        if( planeCounter.nbInPlane(planeCode) == 1 ){
          incrementLineFitParameters(lineFitParameters, scifi_hits, coordX, allXHits, itH);
        }else{
          if ( nOtherHits[planeCode] < SciFi::Tracking::max_other_hits ) {
            assert( nOtherHits[planeCode] < SciFi::Tracking::max_other_hits );
            otherHits[planeCode][ nOtherHits[planeCode]++ ] = itH;
          }
        }
      }
      solveLineFit(lineFitParameters);

      //select best other hits (only best other hit is enough!)
      for(int i = 0; i < SciFi::Constants::n_layers; i++){ 
        if(nOtherHits[i] == 0) continue;
        
        float bestChi2 = 1e9f;
      
        int best = -1;
        for( int hit = 0; hit < nOtherHits[i]; ++hit ){
          const float chi2 = getLineFitChi2(lineFitParameters, scifi_hits, coordX, allXHits, otherHits[i][hit] );
          if( chi2 < bestChi2 ){
            bestChi2 = chi2;
            best = hit;
          }
        }
        if ( best > -1 ) {
          incrementLineFitParameters(lineFitParameters, scifi_hits, coordX, allXHits,  otherHits[i][best]);
        }
      }
      solveLineFit(lineFitParameters);
      
      xAtRef = lineFitParameters.m_c0; //Used to be a helper function now a straight access
    } else { // start magical second part
      // 2) Try to find a small distance containing at least 5(4) different planes
      //    Most of the time do nothing
      const unsigned int nPlanes =  fminf(planeCounter.nbDifferent,uint{5});
      int itWindowStart = it1; 
      int itWindowEnd   = it1 + nPlanes; //pointing at last+1
      //Hit is used, go to next unused one
      while( itWindowEnd<=it2  &&  usedHits[itWindowEnd-1] && (itWindowEnd-1) < n_x_hits ) ++itWindowEnd;
      if( itWindowEnd > it2) continue; //start from very beginning
    
      float minInterval = 1.e9f;
      int best     = itWindowStart;
      int bestEnd  = itWindowEnd;

      PlaneCounter lplaneCounter;
      for (int itH = itWindowStart; itH != itWindowEnd; ++itH) {
        assert( itH < n_x_hits );
        if (!usedHits[itH]) {
          lplaneCounter.addHit( scifi_hits.planeCode[allXHits[itH]]/2 );
        }
      } 
      while ( itWindowEnd <= it2 ) {
        if ( lplaneCounter.nbDifferent >= nPlanes ) {
          //have nPlanes, check x distance
          assert( itWindowEnd-1 < SciFi::Tracking::max_x_hits );
          assert( itWindowStart < SciFi::Tracking::max_x_hits );
          const float dist = coordX[itWindowEnd-1] - coordX[itWindowStart];
          if ( dist < minInterval ) {
            minInterval = dist;
            best    = itWindowStart;
            bestEnd = itWindowEnd;
          }    
        } else {
          //too few planes, add one hit
          ++itWindowEnd;
          if ( itWindowEnd > it2 ) break;
          assert( itWindowEnd <= n_x_hits );
          while( itWindowEnd<=it2  &&  usedHits[itWindowEnd-1] && itWindowEnd < SciFi::Tracking::max_x_hits ) ++itWindowEnd;
          if( itWindowEnd > it2) break;
          lplaneCounter.addHit( scifi_hits.planeCode[allXHits[itWindowEnd-1]]/2 );
          continue;
        } 
        // move on to the right
        
        lplaneCounter.removeHit( scifi_hits.planeCode[allXHits[itWindowStart]]/2 );
        ++itWindowStart;
        assert( itWindowStart < itEnd );
        while( itWindowStart<itWindowEnd && usedHits[itWindowStart] && itWindowStart < n_x_hits) ++itWindowStart;
        //last hit guaranteed to be not used. Therefore there is always at least one hit to go to. No additional if required.
      }
      //TODO tune minInterval cut value
      if ( minInterval < 1.f ) {
        it1 = best;
        it2 = bestEnd;
      }
      assert( it2 <= itEnd );
      //Fill coords and compute average x at reference
      for ( int itH = it1; it2 != itH; ++itH ) {
        assert( itH < itEnd );
        assert( itH < n_x_hits );
        if (!usedHits[itH]) {
          if ( n_coordToFit >= SciFi::Tracking::max_coordToFit )
            break;
          assert( n_coordToFit < SciFi::Tracking::max_coordToFit );
          coordToFit[n_coordToFit++] = allXHits[itH];
          usedHits[itH] = true;
          xAtRef += coordX[ itH ];
        }
      }
      xAtRef /= ((float)n_coordToFit);
    } // end of magical second part
    //=== We have a candidate :)
  
    planeCounter.clear();
    for ( int j = 0; j < n_coordToFit; ++j ) {
      planeCounter.addHit( scifi_hits.planeCode[ coordToFit[j] ] / 2 );
    }
    // Only unused(!) hits in coordToFit now
 
    bool ok = planeCounter.nbDifferent > 3;
    float trackParameters[SciFi::Tracking::nTrackParams];
    if(ok){
      getTrackParameters(xAtRef, velo_state, constArrays, trackParameters);
      // to do: remove outlier from usedHits
      fastLinearFit( scifi_hits, trackParameters, coordToFit, n_coordToFit, planeCounter,pars);
      // to do: do we have to mark these as used as well?
      addHitsOnEmptyXLayers(
        scifi_hits, scifi_hit_count, trackParameters,
        xParams_seed, yParams_seed,
        false, coordToFit,n_coordToFit,
        constArrays, planeCounter, pars, side);
      
      ok = planeCounter.nbDifferent > 3;
    }
    // == Fit and remove hits...
    // to do: remove outlier from usedHits
    if (ok) ok = fitXProjection(scifi_hits, trackParameters, coordToFit, n_coordToFit, planeCounter, pars);
    if (ok) ok = trackParameters[7]/trackParameters[8] < SciFi::Tracking::maxChi2PerDoF;
    if (ok )
      ok = addHitsOnEmptyXLayers(
        scifi_hits, scifi_hit_count, trackParameters,
        xParams_seed, yParams_seed,
        true, coordToFit, n_coordToFit, 
        constArrays, planeCounter, pars, side);
    if (ok) {
      //set ModPrHits used , challenge in c++: we don't have the link any more!
      //here it is fairly trivial... :)
      //Do we really need isUsed in Forward? We can otherwise speed up the search quite a lot!
      // --> we need it for the second loop
      SciFi::Tracking::Track track;
      track.state_endvelo.x = velo_state.x;
      track.state_endvelo.y = velo_state.y;
      track.state_endvelo.z = velo_state.z;
      track.state_endvelo.tx = velo_state.tx;
      track.state_endvelo.ty = velo_state.ty;
      track.chi2 = trackParameters[7];
      for (int k=0;k<7;++k){
        track.trackParams[k] = trackParameters[k];
      }
      
      for ( int i_hit = 0; i_hit < n_coordToFit; ++i_hit ) {
        int hit = coordToFit[i_hit];
        assert( track.hitsNum < SciFi::Tracking::max_scifi_hits );
        track.addHit( hit );
      }
      if ( !secondLoop ) {
        
        assert( n_candidate_tracks < SciFi::Tracking::max_candidate_tracks );
        candidate_tracks[n_candidate_tracks++] = track;
      }  
      else if ( secondLoop ) {
        assert( n_candidate_tracks < SciFi::Tracking::max_tracks_second_loop );
        candidate_tracks[n_candidate_tracks++] = track;
      }
    }
    if ( secondLoop ) {
      if ( n_candidate_tracks >= SciFi::Tracking::max_tracks_second_loop ) break;
      assert( n_candidate_tracks < SciFi::Tracking::max_tracks_second_loop );
    }
    else if ( !secondLoop ) {
      if ( n_candidate_tracks >= SciFi::Tracking::max_candidate_tracks ) break;
      assert( n_candidate_tracks < SciFi::Tracking::max_candidate_tracks );
    }
    
    ++it1;   
  } 
    
} 


__host__ __device__ bool addHitsOnEmptyXLayers(
  const SciFi::SciFiHits& scifi_hits,
  const SciFi::SciFiHitCount& scifi_hit_count,
  float trackParameters[SciFi::Tracking::nTrackParams],
  const float xParams_seed[4],
  const float yParams_seed[4],
  bool fullFit,
  int coordToFit[SciFi::Tracking::max_coordToFit],
  int& n_coordToFit,
  SciFi::Tracking::Arrays* constArrays,
  PlaneCounter& planeCounter,
  SciFi::Tracking::HitSearchCuts& pars,
  int side)
{
  //is there an empty plane? otherwise skip here!
  if (planeCounter.nbDifferent > 11) return true;
  bool  added = false;
  const float x1 = trackParameters[0];
  const float xStraight = straightLineExtend(xParams_seed,SciFi::Tracking::zReference);
  const float xWindow = pars.maxXWindow + ( fabsf( x1 ) + fabsf( x1 - xStraight ) ) * pars.maxXWindowSlope;

  int iZoneStartingPoint = side > 0 ? constArrays->zoneoffsetpar : 0;

  for(unsigned int iZone = iZoneStartingPoint; iZone < iZoneStartingPoint + constArrays->zoneoffsetpar; iZone++) {
    assert( constArrays->xZones[iZone]/2 < SciFi::Constants::n_layers );
    if (planeCounter.nbInPlane( constArrays->xZones[iZone]/2 ) != 0) continue;

    const float parsX[4] = {trackParameters[0],
                            trackParameters[1],
                            trackParameters[2],
                            trackParameters[3]};

    assert( iZone-iZoneStartingPoint < SciFi::Constants::n_zones );
    const float zZone  = constArrays->xZone_zPos[iZone-iZoneStartingPoint];
    const float xPred  = straightLineExtend(parsX,zZone);
    const float minX = xPred - xWindow;
    const float maxX = xPred + xWindow;
    float bestChi2 = 1.e9f;
    int best = -1;

    // -- Use a search to find the lower bound of the range of x values
    assert( constArrays->xZones[iZone] < SciFi::Constants::n_zones );
    int x_zone_offset_begin = scifi_hit_count.layer_offsets[constArrays->xZones[iZone]];
    int x_zone_offset_end   = x_zone_offset_begin + scifi_hit_count.n_hits_layers[constArrays->xZones[iZone]];
    int itH   = getLowerBound(scifi_hits.x0,minX,x_zone_offset_begin,x_zone_offset_end);
    int itEnd = x_zone_offset_end;
    
    for ( ; itEnd != itH; ++itH ) {
      if( scifi_hits.x0[itH] > maxX ) break;
      const float d = scifi_hits.x0[itH] - xPred; //fast distance good enough at this point (?!)
      const float chi2 = d*d * scifi_hits.w[itH];
      if ( chi2 < bestChi2 ) {
        bestChi2 = chi2;
        best = itH;
      }    
    }    
    if ( best > -1 ) {
      if ( n_coordToFit >= SciFi::Tracking::max_coordToFit )
        break;
      assert( n_coordToFit < SciFi::Tracking::max_coordToFit );
      coordToFit[n_coordToFit++] = best; // add the best hit here
      planeCounter.addHit( scifi_hits.planeCode[best]/2 );
      added = true;
    }    
  }
  if ( !added ) return true;
  if ( fullFit ) {
    return fitXProjection(scifi_hits, trackParameters, coordToFit, n_coordToFit, planeCounter, pars);
  }
  fastLinearFit( scifi_hits, trackParameters, coordToFit, n_coordToFit, planeCounter, pars);
  return true;
}
