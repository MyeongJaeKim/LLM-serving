# i : GPU index
# l : interval
# f : file output

# utilization.memory : memory read/write
# pstate : p0 fast
# PCIe generation configured
# pcie.link.gen.max,pcie.link.gen.current

nvidia-smi -i 1 -l 1 -f ./Data.csv --format=csv --query-gpu=timestamp,utilization.gpu,utilization.memory,pstate

# Device monitoring

    # [-s | --select]:      One or more metrics [default=puc]
    #                       Can be any of the following:
    #                           p - Power Usage and Temperature
    #                           u - Utilization
    #                           c - Proc and Mem Clocks
    #                           v - Power and Thermal Violations
    #                           m - FB, Bar1 and CC Protected Memory
    #                           e - ECC Errors and PCIe Replay errors
    #                           t - PCIe Rx and Tx Throughput

nvidia-smi dmon -s uvt -i 7 -f gpu.log && tail -f gpu.log
nvidia-smi dmon -s uvtp -i 7 &