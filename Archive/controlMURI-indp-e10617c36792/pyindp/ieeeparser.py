# Parses IEEE CDF (for power networks) into a dictionary keyed
# by Bus Number (Column 1). Values are dicts which correspond to labels in CDF template
# (omitting spaces).

import os

cdf_bus_template={1: "Name",
                  2: "LoadFlowAreaNumber",
                  3: "LoadZoneNumber",
                  4: "Type",
                  5: "FinalVoltage",
                  6: "FinalAngle",
                  7: "LoadMW",
                  8: "LoadMVAR",
                  9: "GenerationMW",
                  10:"GenerationMVAR",
                  11:"BaseKV",
                  12:"DesiredVolts",
                  13:"MaximumMVAR",
                  14:"MinimumMVAR",
                  15:"ShuntConductance",
                  16:"ShuntSusceptance",
                  17:"RemoteControlledBusNumber"}

def parse_bus(filename):
    dct={}
    with open(filename) as f:
        lines=f.readlines()[2:]
        for line in lines:
            if line[0:4] != "-999":
                bus_id=int(line[0:4].strip())
                bus_name=line[4:15].strip()
                count_spaces=sum([not c.isalnum() for c in bus_name])
                split=line.split()
                dct[bus_id]={}
                dct[bus_id]["Name"]=bus_name
                dct[bus_id]["LoadFlowAreaNumber"]=split[2+count_spaces]
                for i in range(3+count_spaces,18+count_spaces):
                    dct[bus_id][cdf_bus_template[i-count_spaces]]=float(split[i])
            else:
                break
    return dct
    
buses=parse_bus("../data/ieee14cdf.txt")
for k,v in buses.iteritems():
    print k,v["FinalVoltage"],v["FinalVoltage"]  