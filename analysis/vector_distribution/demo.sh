#!/bin/bash
echo "Exhaustive: Integers"
for (( n = 32; n <= 128; n+=32 )); do
  for (( i = 3; i <= 8; i++ )); do
    python3 ./vector_distribution_analysis.py  \
      --format "int$i" \
      --mode exhaustive  \
      --healpix  \
      --nside "$n"  \
      --out3d "images/healpix_int${i}_nside${n}_exhaustive.html"
  done
done

echo "Exhaustive: e_m_"

for (( n = 32; n <= 128; n+=32 )); do
  for (( e = 1; e <= 4; e++ )); do
    for (( m = 2; m <= 3; m++ )); do
      python3 ./vector_distribution_analysis.py  \
        --format fp16 -e "$e" -m "$m" \
        --mode exhaustive  \
        --healpix  \
        --nside "$n"  \
        --out3d "images/healpix_e${e}m${m}_nside${n}_exhaustive.html"
    done
  done
done

for (( n = 32; n <= 128; n+=32 )); do
  for (( i = 3; i <= 8; i++ )); do
    for (( std = 10; std <= 30; std++ )); do
      echo "Gaussian: int${i}"
      python3 ./vector_distribution_analysis.py  \
        --format "int${i}" \
        --mode gaussian  \
        --num 1000000  \
        --std 10 \
        --healpix  \
        --nside "$n"  \
        --out3d "images/healpix_int${i}_gaussian_std${std}_1000000_healpix_${n}.html"
    done
  done
done

for (( n = 32; n <= 128; n+=32 )); do
  for (( e = 0; e <= 4; e++ )); do
    for (( m = 2; m <= 3; m++ )); do
      echo "Gaussian: e${e}m${m}"
      python3 ./vector_distribution_analysis.py  \
        --format fp16 -e "$e" -m "$m"  \
        --mode gaussian  \
        --num 1000000  \
        --healpix  \
        --nside 300  \
        --out3d "images/healpix_e${e}m${m}_gaussian_1000000_healpix_${n}.html"
    done
  done
done
