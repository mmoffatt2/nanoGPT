# python s2_plotly_rotation_demo.py \
#   --metric-mode merge \
#   --mode gen --nA 100 --nB 300 --gamma 0.1 \
#   --beta 12 --theta-deg 20 --alpha-chord 0.8 \
#   --fib-N 256 --healpix-nside 192 --angle-step 10 \
#   --out-html s2_merge_plotly.html --out-csv s2_merge_metrics.csv

python s2_plotly_rotation_demo.py \
  --metric-mode merge \
  --mode gen --nA 120 --nB 120 \
  --gamma1 0.1 --gamma2 0.2 \
  --beta 12 --theta-deg 20 --alpha-chord 0.9 \
  --plot-subset 1000 --marker-size 1 \
  --color-A "#E74C3C" --color-B "#2ECC71" \
  --out-html s2_merge_plotly.html --out-csv s2_merge_metrics.csv

