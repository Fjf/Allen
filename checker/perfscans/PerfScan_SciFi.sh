outloc=../output/perfscans
cd ../..
cp cuda/SciFi/looking_forward/common/include/LookingForwardConstants.cuh bkpfile.bkp

for par in 'maximum_number_of_candidates_per_ut_track' 
do
  for val in '32' '48' '64' '80' '96' '112' '128'
  do
    echo 'Scanning SciFi parameters' $par $val
    sed -i s/$par\ =\ \[^\;\]\*\;/$par\ =\ $val\;/g cuda/SciFi/looking_forward/common/include/LookingForwardConstants.cuh
    cd build
    make -j 8 >& /tmp/WTF
    ./Allen -f /data/gligorov/BsPhiPhi -c 1 -t 1 -r 1 -m 3000 >& $outloc\/SciFi-$par-$val-scan.stdout
    cp $outloc\/PrCheckerPlots.root $outloc\/PrChk-SciFi-$par-$val-scan.root
    cp $outloc\/KalmanIPCheckerOutput.root $outloc\/KFChk-SciFi-$par-$val-scan.root
    ./Allen -f /data/gligorov/minbias -c 0 -n 10000 -t 3 -r 1 -m 4000 >& $outloc\/SciFi-$par-$val-tptscan.stdout    
    cd ..
    cp bkpfile.bkp cuda/SciFi/looking_forward/common/include/LookingForwardConstants.cuh
  done
done

for par in 'maximum_number_of_candidates_per_ut_track_after_x_filter' 
do
  for val in '2' '3' '4'
  do
    echo 'Scanning SciFi parameters' $par $val
    sed -i s/$par\ =\ \[^\;\]\*\;/$par\ =\ $val\;/g cuda/SciFi/looking_forward/common/include/LookingForwardConstants.cuh
    cd build
    make -j 8 >& /tmp/WTF
    ./Allen -f /data/gligorov/BsPhiPhi -c 1 -t 1 -r 1 -m 3000 >& $outloc\/SciFi-$par-$val-scan.stdout
    cp $outloc\/PrCheckerPlots.root $outloc\/PrChk-SciFi-$par-$val-scan.root
    cp $outloc\/KalmanIPCheckerOutput.root $outloc\/KFChk-SciFi-$par-$val-scan.root
    ./Allen -f /data/gligorov/minbias -c 0 -n 10000 -t 3 -r 1 -m 4000 >& $outloc\/SciFi-$par-$val-tptscan.stdout    
    cd ..
    cp bkpfile.bkp cuda/SciFi/looking_forward/common/include/LookingForwardConstants.cuh
  done
done

for par in 'track_min_quality' 
do
  for val in '0.05' '0.075' '0.10' '0.125' '0.15'
  do
    echo 'Scanning SciFi parameters' $par $val
    sed -i s/$par\ =\ \[^\;\]\*\;/$par\ =\ $val\f\;/g cuda/SciFi/looking_forward/common/include/LookingForwardConstants.cuh
    cd build
    make -j 8 >& /tmp/WTF
    ./Allen -f /data/gligorov/BsPhiPhi -c 1 -t 1 -r 1 -m 3000 >& $outloc\/SciFi-$par-$val-scan.stdout
    cp $outloc\/PrCheckerPlots.root $outloc\/PrChk-SciFi-$par-$val-scan.root
    cp $outloc\/KalmanIPCheckerOutput.root $outloc\/KFChk-SciFi-$par-$val-scan.root
    ./Allen -f /data/gligorov/minbias -c 0 -n 10000 -t 3 -r 1 -m 4000 >& $outloc\/SciFi-$par-$val-tptscan.stdout    
    cd ..
    cp bkpfile.bkp cuda/SciFi/looking_forward/common/include/LookingForwardConstants.cuh
  done
done

cp bkpfile.bkp cuda/SciFi/looking_forward/common/include/LookingForwardConstants.cuh
rm bkpfile.bkp
