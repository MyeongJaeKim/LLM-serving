# i : GPU index
# l : interval
# f : file output

# utilization.memory : memory read/write
# pstate : p0 fast
# PCIe generation configured
# pcie.link.gen.max,pcie.link.gen.current

nvidia-smi -i 1 -l 1 -f ./Data.csv --format=csv --query-gpu=timestamp,utilization.gpu,utilization.memory,pstate