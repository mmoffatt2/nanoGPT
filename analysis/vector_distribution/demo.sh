#!/bin/bash

echo "Exhaustive: Integers"
for (( i = 3; i < 8; i++ )); do
python3 ./vector_distribution_analysis.py  \
  --format "int$i" \
  --mode exhaustive  \
  --healpix  \
  --nside 300  \
  --out3d images/healpix_int"$i"_exhaustive.html
done

echo "Exhaustive: e_m_"

for (( e = 0; e < 4; e++ )); do
  for (( m = 2; m < 3; m++ )); do
  python3 ./vector_distribution_analysis.py  \
    --format fp16 -e "$e" -m "$m" \
    --mode exhaustive  \
    --healpix  \
    --nside 300  \
    --out3d images/healpix_e"$e"m"$m"_exhaustive.html
  done
done

echo "Gaussian: int4"
python3 ./vector_distribution_analysis.py  \
  --format int4 \
  --mode gaussian  \
  --num 100000  \
  --std 10 \
  --healpix  \
  --nside 300  \
  --out3d images/healpix_int4_gaussian_std10_100000_healpix_500.html

echo "Gaussian: e3m2"
python3 ./vector_distribution_analysis.py  \
  --format fp16 -e 3 -m 2  \
  --mode gaussian  \
  --num 100000  \
  --healpix  \
  --nside 300  \
  --out3d images/healpix_e3m2_gaussian_100000_healpix_500.html
