# 💰 Budget Table on AWS

Below is a table summarizing the estimated costs for running our examples on a NVIDIA L4 instance `g6.2xlarge` at Amazon Web Services US regions (`us-east-1` and `us-east-2`).

- 💰 Uptime cost is approximately $1 per hour.
- ⏳ Deployment time is approximately 8 minutes ($0.13). Instance loading takes 3 minutes, and Docker pull & load takes 5 minutes.
- 🎮 The NVIDIA L4 delivers [30.3 TFLOPS for FP32](https://www.nvidia.com/en-us/data-center/l4/), offering approximately 36% of the [performance of an RTX 4090](https://www.nvidia.com/en-us/geforce/graphics-cards/40-series/rtx-4090/).
- 🎥 Video frame rate is 60fps.

| **Example** | **Cost** | **Time** | **#Frame** | **#Vert** | **#Face** | **#Tet** | **#Rod** | **Max Strain** |
|--------------|-------|-------|-----|--------|--------|--------|---------|-----|
| trapped      | $0.08 | 4.66m | 300 | 135.4K | 216.5K | 416.5K | ```N/A```     | ```N/A``` |
| twist        | $0.37 | 22.4m | 430 | 204.0K | 406.8K | ```N/A```    | ```N/A```     | ```N/A``` |
| stack        | $0.28 | 16.5m | 120 | 167.4K | 327.7K | 12.7K  | ```N/A```     | 5%  |
| trampoline   | $0.13 | 8.03m | 60  | 36.4K  | 56.8K  | 78.2K  | ```N/A```     | 1%  |
| needle       | $0.20 | 11.7m | 120 | 86.8K  | 169.0K | 12.7K  | ```N/A```     | 5%  |
| cards        | $0.08 | 4.54m | 180 | 9.4K   | 13.9K  | 6.0K   | ```N/A```     | ```N/A``` |
| domino       | $0.06 | 3.38m | 250 | 0.5K   | 0.8K   | 0.3K   | ```N/A```     | ```N/A``` |
| drape        | $0.04 | 2.67m | 100 | 81.9K  | 161.3K | ```N/A```    | ```N/A```     | 5% |
| curtain      | $0.18 | 10.6m | 300 | 64.0K  | 124.2K | ```N/A```    | ```N/A```     | 5% |
| friction     | $0.16 | 9.63m | 850 | 3.3K   | 4.3K   | 12.0K  | ```N/A```     | ```N/A``` |
| hang         | $0.14 | 8.54m | 200 | 16.4K  | 32.3K  | ```N/A```    | ```N/A```     | 1%  |
| belt         | $0.08 | 5.09m | 200 | 12.3K  | 23.3K  | ```N/A```    | ```N/A```     | 5%  |
| codim        | $0.11 | 6.80m | 240 | 61.6K  | 73.6K  | 234.5K | 1.3K    | ```N/A``` |
| fishingknot  | $0.08 | 4.89m | 350 | 19.6K  | 36.9K  | ```N/A```    | ```N/A```     | 5%  |
| fitting      | $0.02 | 1.26m | 240 | 28.5K  | 54.9K  | ```N/A```    | ```N/A```     | 10% |
| noodle       | $0.09 | 5.25m | 240 | 116.3K | ```N/A```    | ```N/A```    | 116.2K  | ```N/A``` |
| ribbon       | $0.25 | 14.8m | 480 | 35.6K  | 53.0K  | 12.6K  | ```N/A```     | ```N/A``` |
| woven        | $0.39 | 23.2m | 450 | 115.6K | ```N/A```    | ```N/A```    | 115.4K  | ```N/A``` |
| yarn         | $0.01 | 0.25m | 120 | 28.6K  | ```N/A```    | ```N/A```    | 28.5K   | ```N/A``` |
| roller       | $0.04 | 2.64m | 180 | 6.2K   | 11.0K  | 7.1K   | ```N/A```     | ```N/A``` |
