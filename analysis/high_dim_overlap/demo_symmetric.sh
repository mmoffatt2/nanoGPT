# python s2_plotly_rotation_demo.py \
#   --metric-mode overlap \
#   --mode gen --nA 100 --nB 100 --gamma 0.2 \
#   --beta 12 --theta-deg 20 --alpha-chord 0.9 \
#   --out-html s2_overlap_plotly.html --out-csv s2_overlap_metrics.csv

# python s2_plotly_rotation_demo.py \
#   --metric-mode overlap \
#   --mode gen --nA 200 --nB 200 \
#   --gamma 0.2 --theta-deg 10 \
#   --fig-height 1200 --scene-frac 0.6 \
#   --plot-subset 1000 --marker-size 1 \
#   --fib-N 256 --healpix-nside 192 --angle-step 10 \
#   --out-html s2_overlap_plotly.html --out-csv s2_overlap_metrics.csv

python s2_plotly_rotation_demo.py \
  --metric-mode overlap \
  --mode gen --nA 1000 --nB 1000 \
  --gamma 0.01 \
  --theta-deg 20 --prox-deg 10 \
  --fig-height 1200 --scene-frac 0.6 \
  --plot-subset 1000 --marker-size 1 \
  --out-html s2_overlap_plotly.html --out-csv s2_overlap_metrics.csv

