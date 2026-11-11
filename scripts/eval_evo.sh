#!/bin/bash
# scores every method against gps with evo
for m in p2p p2plane gicp; do
  echo "===== $m ====="
  echo -n "ATE unaligned (m): "; evo_ape kitti results/gt.txt results/$m.txt | grep rmse
  echo -n "ATE aligned   (m): "; evo_ape kitti results/gt.txt results/$m.txt -a | grep rmse
  echo -n "RPE per step  (m): "; evo_rpe kitti results/gt.txt results/$m.txt --delta 1 --delta_unit f | grep rmse
  echo -n "RPE per 10 m  (m): "; evo_rpe kitti results/gt.txt results/$m.txt --delta 10 --delta_unit m | grep rmse
done
